import torch
import torch.nn as nn
from multimodal_data2vec import KDMMData2Vec, KDData2VecPreTrainingLightningModule
from typing import *
import pytorch_lightning as L
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup
from data2vec_fairseq.data.modality import Modality
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from fairseq.models.roberta.model import RobertaClassificationHead

class TextClassificationLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TextClassificationModel(cfg=self.cfg.model)

        
        self.save_hyperparameters()

    def forward(self, input_dict):
        if self.training and self.mixup_fn is not None \
            and 'target' in input_dict and input_dict['target'] is not None:
            input_dict['image'], input_dict['target'] = self.mixup_fn(input_dict['image'], input_dict['target'])
        
        return self.model(**input_dict)['x']

    def training_step(self, batch:Dict[str, Any], batch_idx):
        target:torch.Tensor = batch.pop('target')
        x = self(batch) # call "forward"

        loss = -target * F.log_softmax(x.float(), dim=-1)

        scores = F.softmax(x, dim=-1)
        top1_accuracy = accuracy_score(target, scores.argmax(dim=-1))
        top5_accuracy = top_k_accuracy_score(target, scores, k=5)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/top1-accuracy", top1_accuracy, prog_bar=True)
        self.log("train/top5-accuracy", top5_accuracy, prog_bar=True)
        return loss.mean()
    
    def validation_step(self, batch:Dict[str, Any], batch_idx):
        target:torch.Tensor = batch.pop('target')
        x = self(batch) # call "forward"

        loss = -target * F.log_softmax(x.float(), dim=-1)

        scores = F.softmax(x, dim=-1)
        top1_accuracy = accuracy_score(target, scores.argmax(dim=-1))
        top5_accuracy = top_k_accuracy_score(target, scores, k=5)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/top1-accuracy", top1_accuracy, prog_bar=True)
        self.log("val/top5-accuracy", top5_accuracy, prog_bar=True)
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
            num_training_steps=self.cfg.optimizer_schedule.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "cosine_w_warmup"}]


class TextClassificationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model:KDMMData2Vec = KDData2VecPreTrainingLightningModule.load_from_checkpoint(self.cfg.model_path).model

        embed_dim = self.cfg.embed_dim
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
        encoder_out = self.model(
            text=text,
            mode=Modality.TEXT,
            padding_mask=padding_mask,
            mask=False,
            features_only=True,
            remove_extra_tokens=False,
        )
        logits = self.classification_head(encoder_out["x"])
        return logits
