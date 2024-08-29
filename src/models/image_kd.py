import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict, Any
import os
import logging
import pytorch_lightning as L
from dataclasses import dataclass
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from omegaconf import OmegaConf
from . import MODEL_REGISTRY
from utils import freeze_module, load_beit2_teacher
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from utils import load_pretrained_d2v_model, prepare_output

logger = logging.getLogger(__name__)

class ImageKDPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = ImageKDModel(cfg=self.cfg.model)

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

        self.save_hyperparameters()

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
            target = self.beitv2_forward_features(image, bool_masked_pos)
        
        output_dict = self({'image': image}) # call "forward"
        pred = output_dict['layer_results']
        pred = prepare_output(pred, modality=Modality.IMAGE)
        target = prepare_output(target, modality=Modality.IMAGE)
        
        loss = self.kd_loss(input=pred, target=target)

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def kd_loss(self, input:torch.Tensor, target:torch.Tensor) -> float:
        input = input.contiguous()
        input = input.view(-1, input.size(-1)).float() # (B, D, C) -> (B*D, C)
        target = target.contiguous()
        target = target.view(-1, target.size(-1)).float() # (B, D, C) -> (B*D, C)

        assert input.shape == target.shape # this must be the case

        return F.mse_loss(input, target, reduction="none").float().mean()
    
    def beitv2_forward_features(self, x, bool_masked_pos):
        x = self.teacher.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.teacher.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.teacher.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.teacher.pos_embed is not None:
            x = x + self.teacher.pos_embed
        x = self.teacher.pos_drop(x)

        rel_pos_bias = self.teacher.rel_pos_bias() if self.teacher.rel_pos_bias is not None else None
        layer_results = []
        for blk in self.teacher.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
            layer_results.append(x)

        return layer_results
    

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
            if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name:
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
class ImageKDConfig():
    depth: int = 6

class ImageKDModel(nn.Module):
    def __init__(self,
                 cfg: ImageKDConfig,
                 ):
        super(ImageKDModel, self).__init__()
        self.cfg = cfg
        
        self.model = load_pretrained_d2v_model(state_dict_path='/workspace/models/base_imagenet.pt')
        self.model.blocks = self.model.blocks[:self.cfg.depth]

    def forward(
        self,
        image:torch.Tensor,
    ):
        return self.model.extract_features(
            source=image,
            remove_extra_tokens=False,
        )


MODEL_REGISTRY['image_kd'] = {
    'cfg': ImageKDConfig,
    'module': ImageKDPreTrainingLightningModule
}
