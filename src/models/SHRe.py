import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
import numpy as np
import os
from utils import load_pretrained_d2v_model
import logging
import pytorch_lightning as L
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import timm
from . import MODEL_REGISTRY
from modules import Block, ClipLoss

logger = logging.getLogger(__name__)

class SHRePreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = SHRe(cfg=self.cfg.model)

        self.teacher = timm.create_model('resnet50.a1_in1k', pretrained=True)
        self.model._freeze(self.teacher)

        self.save_hyperparameters()

    def on_train_start(self):
        logger.info(f'World size: {self.trainer.world_size}')
        self.itc_loss = ClipLoss(
            cache_labels=True,
            rank=self.trainer.local_rank,
            world_size=self.trainer.world_size,
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

        self.model.logit_scale_interm.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo
        self.model.logit_scale_out.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo

        itc_loss1, _, _, _ = self.itc_loss(
            image_features=output_dict['x_interm_image'],
            text_features=output_dict['x_interm_text'],
            logit_scale=self.model.logit_scale_interm.exp(),
        )

        itc_loss2, logits_per_image, logits_per_text, target = self.itc_loss(
            image_features=output_dict['x_text'],
            text_features=output_dict['x_image'],
            logit_scale=self.model.logit_scale_out.exp(),
        )
        self.log_itc_acc(logits_per_text, logits_per_image, target, stage)
            
        itc_loss = (itc_loss1 + itc_loss2) / 2
        self.log(f"{stage}/itc_loss", itc_loss)
        
        loss = kl_loss + itc_loss

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def log_itc_acc(self, logits_per_text, logits_per_image, target, stage):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        self.log(f"{stage}/itc_text_acc", text_itc_acc)
        self.log(f"{stage}/itc_image_acc", img_itc_acc)
        self.log(f"{stage}/itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

    def configure_optimizers(self):
        wd_params, non_wd_params = self._get_param_groups()
        assert len(wd_params) + len(non_wd_params) == len(list(self.model.parameters()))
        optim_args = {
            "params": [
                {"params": wd_params, "weight_decay": self.cfg.optimizer.weight_decay},
                {"params": non_wd_params, "weight_decay": 0}
            ],
            "lr": self.cfg.optimizer.lr,
            "betas": tuple(self.cfg.optimizer.betas),
            "eps": self.cfg.optimizer.eps,
        }
        if 'deepspeed' in self.cfg.lightning_trainer:
            if self.cfg.lightning_trainer.deepspeed.offload_optimizer:
                opt_cls = DeepSpeedCPUAdam
                optim_args['model_params'] = optim_args.pop("params")
            else:
                opt_cls = FusedAdam
        else:
            opt_cls = torch.optim.AdamW
            
        optimizer = opt_cls(**optim_args)
        
        if self.cfg.optimizer.warmup:
            name = self.cfg.optimizer_schedule.type
            if name == 'cosine':
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                    num_training_steps=self.cfg.optimizer_schedule.max_steps,
                )
            else:
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": name}]
        else:
            return optimizer
        
    def _get_param_groups(self):
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or "extra_tokens" in name or 'embed_tokens' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

@dataclass
class SHReConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    embed_dim: int = 768

    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_init_scale: float = 0.2

class SHRe(nn.Module):
    def __init__(self,
                 cfg: SHReConfig,
                 ):
        super(SHRe, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)

        self.shared = Block(
            dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=True,
            init_values=self.cfg.layer_init_scale,
            norm_layer=make_layer_norm,
        )

        self.fc_norm = make_layer_norm(self.cfg.embed_dim)
        self.head = nn.Linear(self.cfg.embed_dim, 1000)

        self.apply(init_bert_params)

        self.logit_scale_interm = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_out = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.text_model = load_pretrained_d2v_model(state_dict_path=os.path.join(self.cfg.pretrained_path, self.cfg.pretrained.text))
        self.text_model.blocks = self.text_model.blocks[:self.cfg.depth]
        self.image_model = load_pretrained_d2v_model(state_dict_path=os.path.join(self.cfg.pretrained_path, self.cfg.pretrained.image))
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
        encoder_out = self.image_model.extract_features(
            source=image,
            remove_extra_tokens=False,
        )
        
        return self.encode_shared(encoder_out)

    def encode_text(self, text, padding_mask):
        encoder_out = self.text_model.extract_features(
            source=text,
            padding_mask=padding_mask,
            remove_extra_tokens=False,
        )
        
        return self.encode_shared(encoder_out)
    
    def encode_shared(self, encoder_out):
        out_dict = dict()
        x = encoder_out['x']
        mask = encoder_out['padding_mask'] if 'padding_mask' in encoder_out else None
        x_interm, x = self.shared(x=x, mask=mask)
        x_interm = x_interm[:, 0]
        x = x[:, 0]
        
        logits = self.fc_norm(x)
        logits = self.head(logits)

        out_dict["encoder_out"] = logits
        x = x / x.norm(dim=-1, keepdim=True)
        out_dict["x"] = x
        x_interm = x_interm / x_interm.norm(dim=-1, keepdim=True)
        out_dict["x_interm"] = x_interm
        return out_dict

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True
        module.train()

MODEL_REGISTRY['SHRe'] = {
    'cfg': SHReConfig,
    'module': SHRePreTrainingLightningModule
}
