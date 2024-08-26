import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
import os
import numpy as np
from transformers import BertModel
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from omegaconf import OmegaConf
from . import MODEL_REGISTRY
from modules import Block, ClipLoss, KDClipMomentumMemoryBankLoss, KDClipLoss
from timm.models.vision_transformer import LayerScale
from utils import freeze_module, load_beit2_teacher
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
        self.teacher.norm = nn.Identity()
        freeze_module(self.teacher)

        self.logit_scale_target = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.save_hyperparameters()

    def on_train_start(self):
        logger.info(f'World size: {self.trainer.world_size}')
        logger.info(f'Local rank: {self.trainer.local_rank}')
        self.clip_loss = ClipLoss(
            cache_labels=True,
            rank=self.trainer.local_rank,
            world_size=self.trainer.world_size
        )
        self.kd_loss_clip = KDClipLoss(
            cache_labels=True,
            rank=self.trainer.local_rank,
            world_size=self.trainer.world_size
        )

        self.kd_loss_mb = KDClipMomentumMemoryBankLoss(
            embed_size=self.cfg.model.embed_dim,
            size=65536,
            device=self.device,
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

        image = batch['image']

        # no mask
        bool_masked_pos = torch.zeros((image.shape[0], self.teacher.patch_embed.num_patches), 
                                      dtype=torch.bool).to(image.device)

        with torch.no_grad():
            target = self.teacher.forward_features(
                x=image,
                bool_masked_pos=bool_masked_pos,
            )[:, 0]
        
        output_dict = self(batch) # call "forward"
        input_text = output_dict['encoder_out_text']
        input_image = output_dict['encoder_out_image']

        input_text = input_text / input_text.norm(dim=-1, keepdim=True)
        input_image = input_image / input_image.norm(dim=-1, keepdim=True)
        target = target / target.norm(dim=-1, keepdim=True)

        if stage == 'train':
            kd_out = self.kd_loss_mb(
                input_image=input_image,
                input_text=input_text,
                target=target,
                logit_scale=self.logit_scale_target.exp(),
            )
            self.kd_loss_mb._update(target=target)
        else:
            kd_out = self.kd_loss_clip(
                input_image=input_image,
                input_text=input_text,
                target=target,
                logit_scale=self.logit_scale_target.exp(),
            )


        self.log(f"{stage}/kd_text_loss", kd_out['text_loss'])
        self.log(f"{stage}/kd_image_loss", kd_out['image_loss'])
        kd_loss = kd_out['loss']
        self.log(f"{stage}/kd_loss", kd_loss)

        self.log_kd_acc(kd_out['logits_per_image'], kd_out['logits_per_text'], kd_out['targets'], stage)

        self.model.logit_scale_interm.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo
        self.model.logit_scale_out.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo

        if stage == 'train':
            itc_out1 = self.clip_loss(
                image_features=output_dict['x_interm_image'],
                text_features=output_dict['x_interm_text'],
                logit_scale=self.model.logit_scale_interm.exp(),
            )

            itc_out2 = self.clip_loss(
                image_features=output_dict['x_image'],
                text_features=output_dict['x_text'],
                logit_scale=self.model.logit_scale_out.exp(),
            )

            self.log_itc_acc(itc_out1['logits_per_image'], itc_out1['logits_per_text'], itc_out1['targets'], stage, key_prefix="interm")
            self.log_itc_acc(itc_out2['logits_per_image'], itc_out2['logits_per_text'], itc_out2['targets'], stage)
                
            itc_loss = (itc_out1['loss'] + itc_out2['loss']) / 2
            self.log(f"{stage}/itc_loss", itc_loss)
        else:
            itc_loss = 0
        
        loss = kd_loss + itc_loss

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def log_itc_acc(self, logits_per_image, logits_per_text, target, stage, key_prefix=""):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        if key_prefix != "":
            key_prefix = key_prefix + "_"
        self.log(f"{stage}/{key_prefix}itc_text_acc", text_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_image_acc", img_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

    def log_kd_acc(self, logits_per_image, logits_per_text, target, stage):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        
        self.log(f"{stage}/kd_text_acc", text_itc_acc)
        self.log(f"{stage}/kd_image_acc", img_itc_acc)
        self.log(f"{stage}/kd_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

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
        wd_params, non_wd_params = self._get_param_groups()
        assert len(wd_params) + len(non_wd_params) == len(list(self.model.parameters()))
        optim_args = {
            "params": [
                {"params": wd_params, "weight_decay": self.cfg.optimizer.weight_decay},
                {"params": non_wd_params, "weight_decay": 0}
            ],
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
        
    def _get_param_groups(self):
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name or 'embeddings' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params
    
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
    num_heads: int = 12
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_init_scale: float = 0.2

class Sx3HRe(nn.Module):
    def __init__(self,
                 cfg: Sx3HReConfig,
                 ):
        super(Sx3HRe, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)

        self.token_type_embeddings = nn.Embedding(2, self.cfg.embed_dim)
        self.tte_scale = LayerScale(self.cfg.embed_dim, init_values=1e-5)

        self.proj_norm = make_layer_norm(self.cfg.embed_dim)
        self.proj_head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim)

        self.shared = Block(
            dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=True,
            init_values=self.cfg.layer_init_scale,
            norm_layer=make_layer_norm,
        )

        self.apply(init_bert_params)

        self.logit_scale_interm = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_out = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_model.encoder.layer = self.text_model.encoder.layer[:self.cfg.depth]
        self.text_model.pooler = None # remove pooler

        beit2_kwargs = OmegaConf.to_container(self.cfg.beitv2, resolve=True)
        sd_path = beit2_kwargs.pop("model_path")
        sd_name = beit2_kwargs.pop("model_name")
        beit_path = os.path.join(sd_path, sd_name)

        self.image_model:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        self.image_model.blocks = self.image_model.blocks[:self.cfg.depth]
        self.image_model.norm = nn.Identity()

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
        # no mask
        bool_masked_pos = torch.zeros((image.shape[0], self.image_model.patch_embed.num_patches), 
                                      dtype=torch.bool).to(image.device)
        x = self.image_model.forward_features(
            x=image,
            bool_masked_pos=bool_masked_pos,
        )

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
        
        return self.encode_shared(x, mask=padding_mask)
    
    def encode_shared(self, x, mask=None):
        out_dict = dict()
        x_interm, x = self.shared(x=x, mask=mask)
        x_interm = x_interm[:, 0]
        x = x[:, 0]

        out_dict["encoder_out"] = self.proj_head(self.proj_norm(x))
        x = x / x.norm(dim=-1, keepdim=True)
        out_dict["x"] = x
        x_interm = x_interm / x_interm.norm(dim=-1, keepdim=True)
        out_dict["x_interm"] = x_interm
        return out_dict

MODEL_REGISTRY['Sx3HRe'] = {
    'cfg': Sx3HReConfig,
    'module': Sx3HRePreTrainingLightningModule
}
