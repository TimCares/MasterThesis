import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
import numpy as np
from omegaconf import OmegaConf
import os
import logging
import pytorch_lightning as L
import json
from timm.models.vision_transformer import LayerScale
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from transformers import BertModel
from . import MODEL_REGISTRY
from modules import Block, ClipLoss
from utils import freeze_module, load_beit2_teacher, load_pretrained_d2v_model, MaskingGenerator
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling

logger = logging.getLogger(__name__)

class Sx3HRePreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = Sx3HRe(cfg=self.cfg.model)

        beit2_kwargs = OmegaConf.to_container(self.cfg.teacher, resolve=True)
        sd_path = beit2_kwargs.pop("model_path")
        sd_name = beit2_kwargs.pop("model_name")
        beit_path = os.path.join(sd_path, sd_name)

        self.teacher:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        freeze_module(self.teacher)

        self.lm_head = nn.Linear(self.model.cfg.embed_dim, 30522)
        self.lm_head.apply(self.teacher._init_weights)
        self.mim_head = nn.Linear(self.model.cfg.embed_dim, self.model.cfg.embed_dim)
        self.mim_head.apply(self.teacher._init_weights)

        self.masked_position_generator = MaskingGenerator(
            input_size=(14, 14),
            num_masking_patches=75,
            max_num_patches=None,
            min_num_patches=16,
        )

        self.mim_logit_scale = nn.Parameter(torch.ones([0]) * np.log(1 / 0.07))

        self.save_hyperparameters()

    def on_train_start(self):
        logger.info(f'World size: {self.trainer.world_size}')
        self.clip_loss = ClipLoss(
            cache_labels=True,
            rank=self.trainer.local_rank,
            world_size=self.trainer.world_size
        )

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):
        if 'target' in batch:
            batch.pop('target') # unused, layer activations are the targets

        image_teacher = batch.pop('image_teacher')

        # no mask
        bool_masked_pos = torch.zeros((image_teacher.shape[0], self.teacher.patch_embed.num_patches), 
                                      dtype=torch.bool).to(image_teacher.device)

        with torch.no_grad():
            target = self.teacher.forward_features(
                x=image_teacher,
                bool_masked_pos=bool_masked_pos,
            )[:, 0]
        
        output_dict = self(batch) # call "forward"
        input_text = output_dict['encoder_out_text']
        input_image = output_dict['encoder_out_image']

        kd_loss1 = F.mse_loss(input=input_text, target=target)
        self.log(f"{stage}/kd_text_loss", kd_loss1)
        kd_loss2 = F.mse_loss(input=input_image, target=target)
        self.log(f"{stage}/kd_image_loss", kd_loss2)
        kd_loss = (kd_loss1 + kd_loss2) / 2
        self.log(f"{stage}/kd_loss", kd_loss)

        self.model.logit_scales.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo

        if stage == 'train':
            itc_loss = 0
            for i, key_prefix in enumerate(['x_interm', 'x']):
                itc_out = self.clip_loss(
                    image_features=output_dict[key_prefix + '_image'],
                    text_features=output_dict[key_prefix + '_text'],
                    logit_scale=self.model.logit_scales[i].exp(),
                )
                self.log_itc_acc(itc_out['logits_per_image'], itc_out['logits_per_text'], itc_out['targets'], stage, key_prefix=key_prefix)
                itc_loss += itc_out['loss']
            itc_loss /= 2
            self.log(f"{stage}/itc_loss", itc_loss)
        else:
            itc_loss = 0
        
        mlm_loss = self.mlm_forward(stage, batch)
        mim_loss = self.mim_forward(stage, batch, image_teacher, bool_masked_pos)
        mdm_loss = 1/3 * (mlm_loss + mim_loss)
        self.log(f"{stage}/mdm_loss", mdm_loss)

        loss = kd_loss + itc_loss + mdm_loss

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def mlm_forward(self, stage, batch):
        x_text_only = self.model.text_model(
            input_ids=batch['masked_text'],
            attention_mask=1-batch['padding_mask'],
        ).last_hidden_state
        x_text_only = self.lm_head(x_text_only)

        input = x_text_only.view(-1, 30522)
        target = batch['targets'].view(-1)
        
        mlm_loss = F.cross_entropy(
            input=input,
            target=target,
            ignore_index=-100,
        )

        input_ = input.argmax(dim=-1)[target != -100]
        target_ = target[target != -100]
        mlm_acc = (input_ == target_).float().mean()
        self.log(f"{stage}/mlm_acc", mlm_acc)

        self.log(f"{stage}/mlm_loss", mlm_loss)

        return mlm_loss
    
    def mim_forward(self, stage, batch, image_teacher, zero_masked_pos):
        bool_masked_pos = self.masked_position_generator()
        x_image_only = self.model.image_model.extract_features(
            source=batch['image'],
            remove_extra_tokens=False,
        )['x']

        with torch.no_grad():
            x_teacher = self.teacher.patch_embed(image_teacher, bool_masked_pos=zero_masked_pos).view(-1, 768)

        bool_masked_pos = bool_masked_pos.view(-1)
        x_masked = x_image_only[:, 1:].view(-1, 768)[bool_masked_pos]

        scores = self.mim_logit_scale * self.mim_head(x_masked) @ x_teacher.T
        target = bool_masked_pos.nonzero().squeeze()

        mim_loss = F.cross_entropy(
            input=scores,
            target=target,
        )

        input_ = scores.argmax(dim=-1)
        mlm_acc = (input_ == target).float().mean()
        self.log(f"{stage}/mlm_acc", mlm_acc)

        self.log(f"{stage}/mim_loss", mim_loss)

        return mim_loss

    
    def log_itc_acc(self, logits_per_image, logits_per_text, target, stage, key_prefix=""):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        if key_prefix != "":
            key_prefix = key_prefix + "_"
        self.log(f"{stage}/{key_prefix}itc_text_acc", text_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_image_acc", img_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

    def configure_optimizers(self):
        ws = torch.cuda.device_count()
        logger.info(f"[Optimizer]: World size is {ws}")
        if 'lr' in self.cfg.optimizer:
            learning_rate = self.cfg.optimizer.lr
        else:
            assert 'base_lr' in self.cfg.optimizer
            learning_rate = self.cfg.optimizer.base_lr * (self.cfg.data.dataloader.batch_size*ws) / 256
            logger.info(f"[Optimizer]: Base Learning rate is {self.cfg.optimizer.base_lr}")
        logger.info(f"[Optimizer]: Learning rate is {learning_rate}")
        param_groups = self._get_param_groups(lr=learning_rate)
        optim_args = {
            "params": param_groups,
            "lr": learning_rate,
            "betas": tuple(self.cfg.optimizer.betas),
            "eps": self.cfg.optimizer.eps,
        }
            
        optimizer = torch.optim.AdamW(**optim_args)
        
        max_steps = int(self.cfg.optimizer.max_steps / ws)
        warmup_steps = int(max_steps * 0.1)
        logger.info(f"[Scheduler]: Max steps is {max_steps}")
        logger.info(f"[Scheduler]: Warmup steps is {warmup_steps}")
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": 'cosine'}]
        
    def _get_param_groups(self, lr):
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "fixed_positional_encoder.positions" in name or "extra_tokens" in name or 'embeddings' in name:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = self.cfg.optimizer.weight_decay

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": lr
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": lr
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        logger.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
        return list(parameter_group_vars.values())
    
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith('teacher.'):
                del sd[k]
        return sd
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class BEiTv2Config():
    model_path:str = "/workspace/models"
    model_name:str = "beitv2_base_patch16_224_pt1k.pth"
    drop_path_rate: float = 0.05
    use_shared_rel_pos_bias:bool = True
    use_abs_pos_emb: bool = False
    vocab_size: int = 8192
    init_values: float =  0.1

@dataclass
class Sx3HReConfig():
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)

    embed_dim: int = 768
    depth: int = 6

class Sx3HRe(nn.Module):
    def __init__(self,
                 cfg: Sx3HReConfig,
                 ):
        super(Sx3HRe, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=1e-6)

        self.token_type_embeddings = nn.Embedding(2, self.cfg.embed_dim)
        self.tte_scale = LayerScale(self.cfg.embed_dim, init_values=1e-5)

        self.shared = Block(
            dim=self.cfg.embed_dim,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            init_values=0.2,
            norm_layer=make_layer_norm,
        )

        self.fc_norm = make_layer_norm(self.cfg.embed_dim)
        self.head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim)

        self.apply(init_bert_params)

        self.logit_scales = nn.Parameter(torch.ones([2]) * np.log(1 / 0.07))

        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_model.encoder.layer = self.text_model.encoder.layer[:self.cfg.depth]
        self.text_model.pooler = None # remove pooler
        
        self.image_model = load_pretrained_d2v_model(state_dict_path='/workspace/models/base_imagenet.pt')
        self.image_model.blocks = self.image_model.blocks[:self.cfg.depth]

    def forward(
        self,
        image:torch.Tensor,
        text:torch.Tensor,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None,
        modality:Modality=None,
    ):
        out = dict()
        if modality is None:
            modality = Modality.VL
        
        if modality==Modality.IMAGE or modality==Modality.VL:
            img_out = self.encode_image(image)
            out.update({k+'_image': v for k, v in img_out.items()})
        if modality==Modality.TEXT or modality==Modality.VL:
            text_out = self.encode_text(text, padding_mask)
            out.update({k+'_text': v for k, v in text_out.items()})
        return out
    
    def encode_image(self, image):
        x = self.image_model.extract_features(
            source=image,
            remove_extra_tokens=False,
        )['x']

        img_tte = self.token_type_embeddings(
            torch.ones_like(x[:, :, 0], dtype=torch.long)
        )
        img_tte = self.tte_scale(img_tte)
        x = x + img_tte
        
        return self.encode_shared(x)

    def encode_text(self, text, padding_mask):
        x = self.text_model(
            input_ids=text,
            attention_mask=1-padding_mask,
        ).last_hidden_state

        text_tte = self.token_type_embeddings(
            torch.zeros_like(padding_mask)
        )
        text_tte = self.tte_scale(text_tte)
        x = x + text_tte
        
        return self.encode_shared(x, padding_mask)
    
    def encode_shared(self, x, mask=None):
        out_dict = dict()
        x_interm, x = self.shared(x=x, mask=mask)
        x_interm = x_interm[:, 0]
        x = x[:, 0]

        out_dict["encoder_out"] = self.head(self.fc_norm(x))
        x = x / x.norm(dim=-1, keepdim=True)
        out_dict["x"] = x
        x_interm = x_interm / x_interm.norm(dim=-1, keepdim=True)
        out_dict["x_interm"] = x_interm
        return out_dict

MODEL_REGISTRY['Sx3HRe'] = {
    'cfg': Sx3HReConfig,
    'module': Sx3HRePreTrainingLightningModule
}
