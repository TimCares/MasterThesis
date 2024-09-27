import logging
from dataclasses import dataclass
from typing import Any, Dict
from omegaconf import MISSING
from . import MODEL_REGISTRY
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from timm.data import Mixup
from transformers.optimization import get_cosine_schedule_with_warmup
import json

logger = logging.getLogger(__name__)


class ImageClassificationLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.model = ImageClassificationModel(cfg=self.cfg.model)

        self.mixup_fn = None

        if cfg.mixup.mixup_alpha > 0 or cfg.mixup.cutmix_alpha > 0:
            self.mixup_fn = Mixup(**cfg.mixup)
        
        self.save_hyperparameters()

    def forward(self, image):
        return self.model(image=image)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx:int, stage:str='train'):
        target = batch['target']
        image = batch['image']

        if self.mixup_fn is not None and stage == 'train':
            image, target = self.mixup_fn(image, target)
        
        x = self(image) # call "forward"

        if stage == 'train' and self.mixup_fn is not None:
            loss = -target * F.log_softmax(x.float(), dim=-1)
        else:
            loss = F.cross_entropy(
                x.float(),
                target,
                reduction="none",
            )
        loss = loss.mean()

        if stage != 'train':
            with torch.no_grad():
                pred = x.argmax(-1)
                acc = (pred == target).sum() / target.size(0)
                self.log(f"{stage}/accuracy", acc, prog_bar=True)
        
        self.log(f"{stage}/loss", loss, prog_bar=True)

        return loss


    def configure_optimizers(self):
        ws = torch.cuda.device_count()
        logger.info(f"[Optimizer]: World size is {ws}")
        if 'lr' in self.cfg.optimizer:
            learning_rate = self.cfg.optimizer.lr
        else:
            assert 'base_lr' in self.cfg.optimizer
            learning_rate = self.cfg.optimizer.base_lr * (self.cfg.data.batch_size*ws) / 256
            logger.info(f"[Optimizer]: Base Learning rate is {self.cfg.optimizer.base_lr}")
        logger.info(f"[Optimizer]: Learning rate is {learning_rate}")
        param_groups = self._get_param_groups(lr=learning_rate)
        optim_args = {
            "params": param_groups,
            "lr": self.cfg.optimizer.lr,
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
        num_layers = len(self.model.model.model.blocks)
        layer_scales = list(self.model.cfg.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2))

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "fixed_positional_encoder.positions" in name or "extra_tokens" in name:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = self.cfg.optimizer.weight_decay
            
            if 'head' in name or 'head_norm' in name:
                group_name = f"head_{group_name}"
                scale = layer_scales[-1]
            elif 'blocks' in name:
                layer_no = int(name.split('blocks.')[1][0]) + 1
                group_name = f"layer_{layer_no}_{group_name}"
                scale = layer_scales[layer_no]
            else:
                group_name = f"pre_layer_{group_name}"
                scale = layer_scales[0]

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": scale*lr
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": scale*lr
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)

        logger.info(f"Param groups = {json.dumps(parameter_group_names, indent=2)}")
        return list(parameter_group_vars.values())

        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class ImageClassificationConfig():
    model_path: str = MISSING
    model_name: str = MISSING
    linear_classifier: bool = False
    num_classes: int = 1000

    layer_decay: float = 0.81

    prediction_mode: str = 'mean_pooling'


class ImageClassificationModel(nn.Module):
    def __init__(self, cfg: ImageClassificationConfig):
        super().__init__()
        self.cfg = cfg
        self.linear_classifier = cfg.linear_classifier

        model_cls:LightningModule = MODEL_REGISTRY[self.cfg.model_name]['module']
        self.model = model_cls.load_from_checkpoint(self.cfg.model_path).model
        if hasattr(self.model, 'prepare_image_finetuning'):
            self.model.prepare_image_finetuning()

        self.linear_classifier = cfg.linear_classifier

        if self.linear_classifier:
            self.model.requires_grad_(False)

        self.head_norm = nn.LayerNorm(768, eps=1e-6)
        nn.init.constant_(self.head_norm.bias, 0)
        nn.init.constant_(self.head_norm.weight, 1.0)

        self.head = nn.Linear(768, cfg.num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

    @property
    def model_call_fn(self):
        if hasattr(self.model, 'encode_image_only'):
            return self.model.encode_image_only
        else:
            return self.model

    def forward(
        self,
        image,
    ):
        
        if self.linear_classifier:
            with torch.no_grad():
                x = self.model_call_fn(image)['x']
        else:
            x = self.model_call_fn(image)['x']

        if self.cfg.prediction_mode == 'mean_pooling':
            x = x[:, 1:].mean(dim=1)
        elif self.cfg.prediction_mode == 'cls_token':
            x = x[:, 0]
        else:
            raise Exception(f"unknown prediction mode {self.cfg.prediction_mode.name}")

        return self.head(self.head_norm(x))

MODEL_REGISTRY['image_classification'] = {
    'cfg': ImageClassificationConfig,
    'module': ImageClassificationLightningModule
}
