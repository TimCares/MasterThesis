import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
import numpy as np
from utils import load_pretrained_d2v_model
import logging
import pytorch_lightning as L
import json
from dataclasses import dataclass
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from transformers import BertModel
import timm
from . import MODEL_REGISTRY
from modules import Block, ClipLoss
from utils import freeze_module

logger = logging.getLogger(__name__)

class SHRePreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = SHRe(cfg=self.cfg.model)

        self.teacher = timm.create_model('vit_base_patch16_rope_reg1_gap_256.sbb_in1k', pretrained=True)
        freeze_module(self.teacher)

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
        with torch.no_grad():
            target = self.teacher(batch.pop('image_teacher'))
        
        output_dict = self(batch) # call "forward"
        target = torch.nn.functional.log_softmax(target, dim=1)
        input_text = torch.nn.functional.log_softmax(output_dict['encoder_out_text'], dim=1)
        input_image = torch.nn.functional.log_softmax(output_dict['encoder_out_image'], dim=1)

        kl_loss1 = F.kl_div(input=input_text, target=target, log_target=True, reduction='batchmean')
        self.log(f"{stage}/kl_text_loss", kl_loss1)
        kl_loss2 = F.kl_div(input=input_image, target=target, log_target=True, reduction='batchmean')
        self.log(f"{stage}/kl_image_loss", kl_loss2)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        self.log(f"{stage}/kl_loss", kl_loss)

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
        
        loss = kl_loss + itc_loss

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
class SHReConfig():
    embed_dim: int = 768
    depth: int = 6

class SHRe(nn.Module):
    def __init__(self,
                 cfg: SHReConfig,
                 ):
        super(SHRe, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=1e-6)

        self.shared = Block(
            dim=self.cfg.embed_dim,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            init_values=0.2,
            norm_layer=make_layer_norm,
        )

        self.fc_norm = make_layer_norm(self.cfg.embed_dim)
        self.head = nn.Linear(self.cfg.embed_dim, 1000)

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
        
        return self.encode_shared(x)

    def encode_text(self, text, padding_mask):
        x = self.text_model(
            input_ids=text,
            attention_mask=1-padding_mask,
        ).last_hidden_state
        
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

MODEL_REGISTRY['SHRe'] = {
    'cfg': SHReConfig,
    'module': SHRePreTrainingLightningModule
}
