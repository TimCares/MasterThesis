import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
from omegaconf import OmegaConf
import os
import logging
import pytorch_lightning as L
import json
import numpy as np
from timm.models.vision_transformer import LayerScale
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from transformers import BertModel
from . import MODEL_REGISTRY
from modules import ClipLoss
from modules.layers import Pooler, MoMEBlock
from utils import freeze_module, load_beit2_teacher
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling

logger = logging.getLogger(__name__)

class SSMKECPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = SSMKEC(cfg=self.cfg.model)

        self.lm_head = nn.Linear(self.model.cfg.embed_dim, 30522)
        self.lm_head.apply(init_bert_params)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
        unimodal_output = self.model.encode_unimodal(
            image=batch['image'],
            text=batch['text'],
            padding_mask=batch['padding_mask'],
        )

        mlm_output = self.mlm_forward(stage, batch, unimodal_output)
        itc_output = self.itc_forward(stage, batch, unimodal_output)
        itm_output = self.itm_forward(
            stage=stage,
            batch=batch,
            unimodal_output=unimodal_output,
            logits_per_image=itc_output['logits_per_image'],
            logits_per_text=itc_output['logits_per_text']
        )

        loss = mlm_output['mlm_loss'] + itc_output['loss'] + itm_output['loss']
        
        return loss
    
    def mlm_forward(self, stage, batch, unimodal_output):
        x = self.model.encode_shared_only(
            x=unimodal_output['x'],
            padding_mask=unimodal_output['padding_mask'],
        )['x'][:, :batch['masked_text'].size(1)] # 64 is the max text length

        x = self.lm_head(x)

        input = x.view(-1, 30522)
        target = batch['targets'].view(-1)
        
        mlm_loss = F.cross_entropy(
            input=input,
            target=target,
            ignore_index=-100,
        )

        input_ = input.argmax(dim=-1)[target != -100]
        target_ = target[target != -100]
        mlm_acc = (input_ == target_).float().mean()
        
        output_dict = {
            "loss": mlm_loss,
            "acc": mlm_acc,
        }
        self.log(f"{stage}/'mlm_loss", mlm_loss)
        self.log(f"{stage}/'mlm_acc", mlm_acc, prog_bar=True)
        
        return output_dict
    
    def itc_forward(self, stage, batch, unimodal_output):
        img_embed = self.model.encode_shared_only(
            x=unimodal_output['x_tte_image'],
            modality_type='image',
        )['img_embed']
        text_embed = self.model.encode_shared_only(
            x=unimodal_output['x_tte_text'],
            padding_mask=batch['padding_mask'],
            modality_type='text',
        )['text_embed']
        
        self.logit_scale.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo

        itc_out = self.clip_loss(
            image_features=img_embed,
            text_features=text_embed,
            logit_scale=self.logit_scale.exp(),
        )
        self.log_acc(itc_out['logits_per_image'], itc_out['logits_per_text'], itc_out['targets'], stage, key_prefix='itc_')
        return itc_out
    
    def itm_forward(self, stage, batch, unimodal_output, logits_per_image=None, logits_per_text=None):
        device = self.device
        bsz = batch['image'].shape[0]
        itm_labels = torch.cat([
            torch.ones(bsz), 
            torch.zeros(bsz), 
            torch.zeros(bsz)]).to(device)

        if logits_per_image is not None and logits_per_text is not None: # ... then hard-negative mining
            with torch.no_grad():       
                weights_i2t = F.softmax(logits_per_image[:bsz, :bsz].float(), dim=1)
                weights_t2i = F.softmax(logits_per_text[:bsz, :bsz].float(), dim=1)

                weights_i2t.fill_diagonal_(0)
                weights_t2i.fill_diagonal_(0)
        else: # ... then random sampling
            weights_i2t = torch.ones(bsz, bsz)
            weights_i2t.fill_diagonal_(0)
            weights_t2i = weights_i2t

        neg_text_idx = torch.multinomial(weights_i2t, 1).squeeze()
        neg_image_idx = torch.multinomial(weights_t2i, 1).squeeze()

        pos_pred = self.model.encode_shared_only(
            x=unimodal_output['x'],
            padding_mask=unimodal_output['padding_mask'],
        )['matched']

        x_text = unimodal_output['x_tte_text'][neg_text_idx]
        x_image = unimodal_output['x_tte_image']
        x = torch.cat([x_text, x_image], dim=1)
        neg_text_pred = self.model.encode_shared_only(
            x=x,
            padding_mask=unimodal_output['padding_mask'],
        )['matched']

        x_text = unimodal_output['x_tte_text']
        x_image = unimodal_output['x_tte_image'][neg_image_idx]
        x = torch.cat([x_text, x_image], dim=1)
        neg_img_pred = self.model.encode_shared_only(
            x=x,
            padding_mask=unimodal_output['padding_mask'],
        )['matched']

        input = torch.cat([pos_pred, neg_text_pred, neg_img_pred], dim=0)

        result_dict = {
            'loss': F.cross_entropy(input, itm_labels.long()),
            'logits': input,
            'targets': itm_labels,
        }
        self.log(f"{stage}/'itm_loss", result_dict['loss'])
        acc = (input.argmax(dim=1) == itm_labels).float().mean()
        self.log(f"{stage}/'itm_acc", acc, prog_bar=True)

        return result_dict
    
    def log_acc(self, logits_per_image, logits_per_text, target, stage, key_prefix=""):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        if key_prefix != "":
            key_prefix = key_prefix + "_"
        self.log(f"{stage}/{key_prefix}text_acc", text_itc_acc)
        self.log(f"{stage}/{key_prefix}image_acc", img_itc_acc)
        self.log(f"{stage}/{key_prefix}acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

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
class SSMKECConfig():
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)

    embed_dim: int = 768
    depth: int = 6
    shared_depth: int = 2

class SSMKEC(nn.Module):
    def __init__(self,
                 cfg: SSMKECConfig,
                 ):
        super(SSMKEC, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=1e-6)
        self.post_norm = make_layer_norm(cfg.embed_dim)

        self.token_type_embeddings = nn.Embedding(2, self.cfg.embed_dim)
        self.tte_scale = LayerScale(self.cfg.embed_dim, init_values=1e-5)

        self.pooler = Pooler(self.cfg.embed_dim)

        self.itc_img_head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=False)
        self.itc_text_head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=False)

        self.itm_head = nn.Linear(self.cfg.embed_dim, 2)

        self.shared_blocks = nn.ModuleList([
            MoMEBlock(
                dim=self.cfg.embed_dim,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                layer_scale_init_values=0.2,
                norm_layer=make_layer_norm,
                max_text_len=64,
            ) for _ in range(self.cfg.shared_depth)
        ])

        self.apply(init_bert_params)

        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_model.encoder.layer = self.text_model.encoder.layer[:self.cfg.depth]
        self.text_model.pooler = None # remove pooler
        freeze_module(self.text_model)
        
        beit2_kwargs = OmegaConf.to_container(self.cfg.beitv2, resolve=True)
        sd_path = beit2_kwargs.pop("model_path")
        sd_name = beit2_kwargs.pop("model_name")
        beit_path = os.path.join(sd_path, sd_name)

        self.image_model:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        self.image_model.blocks = self.image_model.blocks[:self.cfg.depth]
        freeze_module(self.image_model)
    
    def encode_image_only(self, image, bool_masked_pos=None):
        x = self.image_model.forward_features(
            x=image,
            bool_masked_pos=bool_masked_pos,
        )

        img_tte = self.token_type_embeddings(
            torch.ones_like(x[:, :, 0], dtype=torch.long)
        )
        img_tte = self.tte_scale(img_tte)
        x_tte = x + img_tte

        return {
            'x': x,
            'x_tte': x_tte,
        }

    def encode_text_only(self, text, padding_mask):
        x = self.text_model(
            input_ids=text,
            attention_mask=1-padding_mask,
        ).last_hidden_state

        text_tte = self.token_type_embeddings(
            torch.zeros_like(padding_mask)
        )
        text_tte = self.tte_scale(text_tte)
        x_tte = x + text_tte

        return {
            'x': x,
            'x_tte': x_tte,
        }
    
    def encode_unimodal(self, image, text, padding_mask, bool_masked_pos=None):
        result_dict = dict()
        img_out = self.encode_image_only(image, bool_masked_pos=bool_masked_pos)
        result_dict.update({k+'_image': v for k, v in img_out.items()})
        text_out = self.encode_text_only(text, padding_mask)
        result_dict.update({k+'_text': v for k, v in text_out.items()})

        x_image = img_out['x_tte']
        x_text = text_out['x_tte']
        x = torch.cat([x_text, x_image], dim=1)
        image_mask = torch.zeros(image.size(0), image.size(1)).to(padding_mask.device)
        padding_mask = torch.cat([padding_mask, image_mask], dim=1)
        result_dict.update({
            'x': x,
            'padding_mask': padding_mask,
        })

        return result_dict
    
    def forward(self, image=None, text=None, padding_mask=None, bool_masked_pos=None):
        return self.encode_shared(image, text, padding_mask, bool_masked_pos)
    
    def encode_shared(self, image=None, text=None, padding_mask=None, bool_masked_pos=None):
        modality_type = None
        if text is None:
            x = self.encode_image_only(image, bool_masked_pos=bool_masked_pos)['x_tte']
            modality_type = "image"
        if image is None:
            x = self.encode_text_only(text, padding_mask)['x_tte']
            modality_type = "text"

        if modality_type is None:
            x_image = self.encode_image_only(image, bool_masked_pos=bool_masked_pos)['x_tte']
            x_text = self.encode_text_only(text, padding_mask)['x_tte']
            modality_type = 'vl'

            x = torch.cat([x_text, x_image], dim=1)
            image_mask = torch.zeros(image.size(0), image.size(1)).to(padding_mask.device)
            padding_mask = torch.cat([padding_mask, image_mask], dim=1)

        return self.encode_shared_only(x, padding_mask, modality_type)

    
    def encode_shared_only(self, x, padding_mask=None, modality_type=None):
        modality_type = modality_type if modality_type is not None else 'vl'

        for block in self.shared_blocks:
            x = block(x=x, mask=padding_mask, modality_type=modality_type)
        x = self.post_norm(x)

        matched = self.itm_head(self.pooler(x))

        img_embed = self.itc_img_head(x[:, 0])
        text_embed = self.itc_text_head(x[:, 0])

        result_dict = {
            'x': x,
            'img_embed': img_embed,
            'text_embed': text_embed,
            'matched': matched,
        }
        return result_dict
        

MODEL_REGISTRY['SSMKEC'] = {
    'cfg': SSMKECConfig,
    'module': SSMKECPreTrainingLightningModule
}
