import logging

from dataclasses import dataclass
from typing import Any, Dict
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from models import MODEL_REGISTRY
from data2vec_fairseq.data.modality import Modality

from transformers.optimization import get_cosine_schedule_with_warmup


logger = logging.getLogger(__name__)


class RetrievalLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.model = RetrievalModel(cfg=self.cfg.model)
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx:int, stage:str='train'):
        output_dict = self(batch) # call "forward"

        self.model.logit_scale_interm.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo
        self.model.logit_scale_out.data.clamp_(0, 4.6052) # as per FLAVA, also max value of VLMo

        itc_loss1 = self.itc_loss(
            text_features=output_dict['x_interm_text'],
            image_features=output_dict['x_interm_image'],
            logit_scale=self.model.logit_scale_interm.exp(),
            stage=stage)
        itc_loss2 = self.itc_loss(
            text_features=output_dict['x_text'],
            image_features=output_dict['x_image'],
            logit_scale=self.model.logit_scale_out.exp(),
            stage=stage)
        
        itc_loss = (itc_loss1 + itc_loss2) / 2
        self.log(f"{stage}/itc_loss", itc_loss)

        return itc_loss
    
    def itc_loss(
        self,
        text_features:torch.Tensor,
        image_features:torch.Tensor,
        logit_scale:torch.Tensor,
        stage:str='train') -> torch.Tensor:
        
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        target = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        self.log(f"{stage}/itc_text_acc", text_itc_acc)
        self.log(f"{stage}/itc_image_acc", img_itc_acc)
        self.log(f"{stage}/itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), target)
            + F.cross_entropy(logits_per_text.float(), target)
        ) / 2
        return itc_loss


    def configure_optimizers(self):
        wd_params, non_wd_params = self._get_param_groups()
        assert len(wd_params) + len(non_wd_params) == len(list(self.model.parameters()))
        optimizer = torch.optim.AdamW(
            params=[
                {"params": wd_params, "weight_decay": self.cfg.optimizer.weight_decay},
                {"params": non_wd_params, "weight_decay": 0}
            ],
            lr=self.cfg.optimizer.lr,
            betas=tuple(self.cfg.optimizer.betas),
            eps=self.cfg.optimizer.eps,
            # weight_decay=self.cfg.optimizer.weight_decay -> not needed becuase of param groups
        )
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
            num_training_steps=self.cfg.optimizer_schedule.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

@dataclass
class RetrievalConfig():
    model_path:str
    model_name:str


class RetrievalModel(nn.Module):
    def __init__(self, cfg: RetrievalConfig):
        super().__init__()
        self.cfg = cfg
        
        model_cls:LightningModule = MODEL_REGISTRY[cfg.model_name]['module']
        self.model = model_cls.load_from_checkpoint(self.cfg.model_path).model

        self.model.prepare_fine_tuning(keep_modality=Modality.IMAGE)


    def forward(
        self,
        image,
        text,
        padding_mask,
        id,
        modality,
    ):
        
        return self.model(image=image, text=text, padding_mask=padding_mask, id=id, modality=modality)