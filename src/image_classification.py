from timm.data import Mixup
import torch
import torch.nn as nn
from multimodal_data2vec import KDMMData2VecConfig, KDMMData2Vec, KDData2VecPreTrainingLightningModule
from typing import *
import pytorch_lightning as L
import torch.nn.functional as F
from transformers.optimization import get_cosine_schedule_with_warmup
from data2vec_fairseq.data.modality import Modality
from sklearn.metrics import accuracy_score, top_k_accuracy_score


class ImageClassificationLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ImageClassificationModel(cfg=self.cfg.model)

        self.mixup_fn = None

        if cfg.mixup > 0 or cfg.cutmix > 0:
            self.mixup_fn = Mixup(
                mixup_alpha=cfg.mixup,
                cutmix_alpha=cfg.cutmix,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=cfg.label_smoothing,
                num_classes=cfg.num_classes,
            )
        
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

class ImageClassificationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # self.model:KDMMData2Vec = merge_with_parent(dc=KDMMData2VecConfig(), cfg=self.cfg.model, remove_missing=False)
        self.model:KDMMData2Vec = KDData2VecPreTrainingLightningModule.load_from_checkpoint(self.cfg.model_path).model

        # self.model.load_state_dict(torch.load(cfg.model_path)['state_dict'])

        self.fc_norm = nn.LayerNorm(self.cfg.model.embed_dim)
        self.head = nn.Linear(self.cfg.model.embed_dim, cfg.num_classes)

        self.head.weight.data.mul_(1e-3)
        self.head.bias.data.mul_(1e-3)

    def forward(
        self,
        img,
    ):

        x = self.model(modes=[Modality.IMAGE],
                       image=img,
                       padding_mask=None,
                       mask=False,
                       features_only=True,
                       remove_extra_tokens=False,)
        x = x[:, 0] # x[:, 1:]
        x = self.fc_norm(x)
        x = self.head(x)

        return x
