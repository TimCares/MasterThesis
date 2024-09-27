import logging
from dataclasses import dataclass
from omegaconf import MISSING
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Any, Dict
from sklearn.metrics import f1_score as f1_score
from sklearn.metrics import matthews_corrcoef as matthews_corrcoef
from scipy.stats import spearmanr as spearmanr_
import numpy as np
from . import MODEL_REGISTRY
from pytorch_lightning import LightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

logger = logging.getLogger(__name__)

def accuracy(target, pred):
    pred = np.array(pred)
    target = np.array(target)
    return round(((pred == target).sum() / target.shape[0])*100, 2)

def spearman(target, pred):
    return spearmanr_(pred, target)[0]

_METRIC_REGISTRY = {
    "f1": f1_score,
    "mcc": matthews_corrcoef,
    "spearman": spearman,
    "accuracy": accuracy,
}

class TextClassificationLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.model = TextClassificationModel(cfg=self.cfg.model)

        self.metric = _METRIC_REGISTRY[self.cfg.metric]

        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int, dataloader_idx=0):
        log_prefix = ""
        if self.cfg.data._name == 'mnli_glue':
            if dataloader_idx==0:
                log_prefix = "dev_matched_"
            else:
                log_prefix = "dev_mismatched_"
        return self._step(batch, batch_idx, stage='val', log_prefix=log_prefix)

    def _step(self, batch:Dict[str, Any], batch_idx:int, stage:str='train', log_prefix:str=""):
        target = batch.pop('target')
        input_dict = {'text': batch['text'], 'attention_mask': batch['attention_mask']}
        if 'token_type_ids' in batch:
            input_dict['token_type_ids'] = batch['token_type_ids']
        
        logits = self(input_dict) # call "forward"

        if not self.cfg.regression:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, target, reduction="mean")
            pred = logits.argmax(dim=-1).detach().cpu().tolist()
        else:
            logits = logits.view(-1).float()
            target = target.float()
            loss = F.mse_loss(logits, target, reduction="mean")
            pred = logits.detach().cpu().tolist()

        target = target.detach().cpu().tolist()

        score = self.metric(target, pred)
        self.log(f"{stage}/{log_prefix}{self.cfg.metric}", score, prog_bar=True)
        
        self.log(f"{stage}/{log_prefix}loss", loss, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        max_steps = self.cfg.optimizer.max_steps
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(max_steps * 0.1),
            num_training_steps=max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": 'poly_decay_w_warmup'}]

        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class TextClassificationConfig:
    model_path: str = MISSING
    model_name: str = MISSING
    num_classes: int = 2
    dropout: float = 0.0


class TextClassificationModel(nn.Module):
    def __init__(self, cfg: TextClassificationConfig):
        super().__init__()
        self.cfg = cfg

        model_cls:LightningModule = MODEL_REGISTRY[self.cfg.model_name]['module']
        self.model = model_cls.load_from_checkpoint(self.cfg.model_path, strict=False).model
        if hasattr(self.model, 'prepare_text_finetuning'):
            self.model.prepare_text_finetuning()

        embed_dim = 768
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.classification_head = nn.Linear(embed_dim, self.cfg.num_classes)

    @property
    def model_call_fn(self):
        if hasattr(self.model, 'encode_text_only'):
            return self.model.encode_text_only
        else:
            return self.model

    def forward(
        self,
        text,
        attention_mask,
        token_type_ids=None,
    ):
        x = self.model_call_fn(text=text, attention_mask=attention_mask, token_type_ids=token_type_ids)['pooler_output']
        return self.classification_head(self.dropout(x))

MODEL_REGISTRY['text_classification'] = {
    'cfg': TextClassificationConfig,
    'module': TextClassificationLightningModule
}
