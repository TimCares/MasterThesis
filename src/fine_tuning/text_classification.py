import logging

from dataclasses import dataclass

from omegaconf import MISSING, open_dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Any, Dict

from sklearn.metrics import f1_score as _f1_score
from scipy.stats import pearsonr, spearmanr

from fairseq.dataclass import FairseqDataclass
from fairseq.models.roberta.model import RobertaClassificationHead
from fairseq.criterions.sentence_prediction import (
    acc_and_f1 as __acc_and_f1, 
    pearson_and_spearman as __pearson_and_spearman, 
    matthews_corrcoef as __matthews_corrcoef
)

from data2vec_fairseq.data.modality import Modality
from multimodal_data2vec import KDMMData2Vec, KDData2VecPreTrainingLightningModule
from transformers.optimization import get_polynomial_decay_schedule_with_warmup

logger = logging.getLogger(__name__)

def accuracy(target, pred):
    return {'accuracy': round(((pred == target).sum() / target.size(0))*100, 2)}

def matthews_corrcoef(pred, target):
    return {'matthews_corrcoef': round(__matthews_corrcoef(pred.numpy(), target.numpy()), 2)}

def acc_and_f1(pred, target):
    result_dict = __acc_and_f1(pred.numpy(), target.numpy())
    return {k: round(v*100, 2) for k, v in result_dict.items()}

def f1_score(pred, target):
    return {'f1': round(_f1_score(target.numpy(), pred.numpy())*100, 2)}

def pearson_and_spearman(pred, target):
    result_dict = __pearson_and_spearman(pred.numpy(), target.numpy())
    return {k: round(v*100, 2) for k, v in result_dict.items()}  

_METRIC_REGISTRY = {
    "f1": f1_score,
    "mcc": matthews_corrcoef,
    "pearson": pearsonr,
    "spearman": spearmanr,
    "accuracy": accuracy,
    "acc_and_f1": acc_and_f1,
    "pearson_and_spearman": pearson_and_spearman,
}

class TextClassificationLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        with open_dict(self.cfg.model):
            # "last.ckpt" is a symlink to the checkpoint with the best zero-shot performance during pre-training (created by PL)
            self.cfg.model.model_path = os.path.join(self.cfg.model_path, self.cfg.model.model_version, "last.ckpt")
        
        self.model = TextClassificationModel(cfg=self.cfg.model)

        self.metrics = [_METRIC_REGISTRY[metric] for metric in self.cfg.metrics]
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx:int, stage:str='train'):
        target = batch.pop('target').detach().cpu().tolist()
        batch.pop('modes')
        
        logits = self(batch) # call "forward"

        if not self.cfg.regression:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, target, reduction="mean")
            pred = logits.argmax(dim=-1).detach().cpu().tolist()
        else:
            logits = logits.view(-1).float()
            target = target.float()
            loss = F.mse_loss(logits, target, reduction="mean")
            pred = logits.detach().cpu().tolist()

        for metric in self.metrics:
            result_dict = metric(pred, target)
            for k, v in result_dict.items():
                self.log(f"{stage}/{k}", v, prog_bar=True)
        
        self.log(f"{stage}/loss", loss, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        if self.cfg.optimizer.warmup:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                num_training_steps=self.cfg.optimizer_schedule.max_steps,
                lr_end=self.cfg.optimizer_schedule.lr_end,
                power=self.cfg.optimizer_schedule.power,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": 'poly_decay_w_warmup'}]
        else:
            return optimizer

@dataclass
class TextClassificationConfig:
    model_version: str = MISSING

    num_classes: int = 2
    pooler_dropout: float = 0.0
    pooler_activation_fn: str = "tanh"
    quant_noise_pq: int = 0
    quant_noise_pq_block_size: int = 8
    spectral_norm_classification_head: bool = False


class TextClassificationModel(nn.Module):
    def __init__(self, cfg: TextClassificationConfig):
        super().__init__()
        self.cfg = cfg

        pretrained_args = torch.load(cfg.model_path)['hyper_parameters']['cfg']

        self.model:KDMMData2Vec = KDData2VecPreTrainingLightningModule.load_from_checkpoint(self.cfg.model_path,
                                                                                            cfg=pretrained_args).model
        self.model.prepare_fine_tuning(keep_modes=[Modality.TEXT])

        embed_dim = pretrained_args.model.embed_dim
        self.classification_head = RobertaClassificationHead(
            input_dim=embed_dim,
            inner_dim=embed_dim,
            num_classes=self.cfg.num_classes,
            activation_fn=self.cfg.pooler_activation_fn,
            pooler_dropout=self.cfg.pooler_dropout,
            q_noise=self.cfg.quant_noise_pq,
            qn_block_size=self.cfg.quant_noise_pq_block_size,
            do_spectral_norm=self.cfg.spectral_norm_classification_head,
        )

    def forward(
        self,
        text,
        padding_mask,
    ):
        
        x = self.model.extract_features(
            text=text,
            modes=[Modality.TEXT],
            padding_mask=padding_mask,
            remove_extra_tokens=False, # we keep the bos token -> used by the classification head (but D2V removes it before, so check both)
        )["x"]

        return self.classification_head(x)
