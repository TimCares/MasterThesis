import logging
from typing import Any, Dict
from pytorch_lightning import LightningModule
import torch
import pytorch_lightning as L
from models import MODEL_REGISTRY
from modules import ClipLoss
import json
from transformers.optimization import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class RetrievalLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        pretrained_args = torch.load(self.cfg.pretrained.model_path)['hyper_parameters']['cfg']
        if 'dropout' in self.cfg:
            pretrained_args['model']['dropout'] = self.cfg.dropout
        
        model_cls:LightningModule = MODEL_REGISTRY[self.cfg.pretrained.model_name]['module']
        self.module = model_cls.load_from_checkpoint(self.cfg.pretrained.model_path, strict=False, cfg=pretrained_args)
        del self.module.teacher
        del self.module.logit_scale_target
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.module.model(**input_dict)
    
    def on_train_start(self):
        logger.info(f'World size: {self.trainer.world_size}')
        self.clip_loss = ClipLoss(
            cache_labels=True,
            rank=self.trainer.local_rank,
            world_size=self.trainer.world_size
        )

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx:int, stage:str='train'):
        output_dict = self(batch) # call "forward"

        self.module.model.logit_scales.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo

        itc_out = self.clip_loss(
            image_features=output_dict['x_image'],
            text_features=output_dict['x_text'],
            logit_scale=self.module.model.logit_scales[-1].exp(),
        )
        self.log_acc(itc_out['logits_per_image'], itc_out['logits_per_text'], itc_out['targets'], stage)
        loss = itc_out['loss']

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
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

        for name, param in self.module.model.named_parameters():
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
        super().log(batch_size=self.cfg.data.batch_size, sync_dist=True, *args, **kwargs)

MODEL_REGISTRY['retrieval_finetune'] = {
    'module': RetrievalLightningModule,
}
