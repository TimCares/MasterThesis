# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The code in this file is adapted from the BeiT implementation which can be found here:
# https://github.com/microsoft/unilm/tree/master/beit

import logging

from dataclasses import dataclass
from typing import Any, Optional, Dict

from omegaconf import MISSING, open_dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

from timm.data import Mixup, create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2 as transforms
import PIL

from data2vec_fairseq.data.modality import Modality
from multimodal_data2vec import KDMMData2Vec, KDData2VecPreTrainingLightningModule
from data2vec_fairseq.models.mae_image_classification import PredictionMode
from transformers.optimization import get_cosine_schedule_with_warmup

from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)


def build_transform(is_train, input_size, color_jitter, aa, reprob, remode, recount):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=color_jitter,
            auto_augment=aa,
            interpolation="bicubic",
            re_prob=reprob,
            re_mode=remode,
            re_count=recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(
            size, interpolation=PIL.Image.BICUBIC
        ),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class ImageClassificationLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.model = ImageClassificationModel(cfg=self.cfg)

        self.mixup_fn = None

        if cfg.mixup > 0 or cfg.cutmix > 0:
            self.mixup_fn = Mixup(**cfg.mixup)

        self.image_transforms = {
            "train": build_transform(is_train=True, **cfg.transforms),
            "val": build_transform(is_train=False, **cfg.transforms),
        }
        
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

        image = self.image_transforms[stage](image) # apply transforms batch-wise

        if self.mixup_fn is not None and stage == 'train':
            image, target = self.mixup_fn(image, target)
        
        x = self(image) # call "forward"

        loss = F.cross_entropy(
            x.float(),
            target,
            reduction="mean",
        )

        with torch.no_grad():
            pred = x.argmax(-1)
            acc = (pred == target).sum()
        
        self.log(f"{stage}/loss", loss, prog_bar=True)
        self.log(f"{stage}/accuracy", acc, prog_bar=True)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        if self.cfg.optimizer.warmup:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                num_training_steps=self.cfg.optimizer_schedule.max_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": 'cosine_w_warmup'}]
        else:
            return optimizer

@dataclass
class ImageClassificationConfig(FairseqDataclass):
    model_path: str = MISSING
    linear_classifier: bool = False
    num_classes: int = 1000

    drop_path_rate: float = 0.0
    layer_decay: float = 0.65

    norm_eps: Optional[float] = None

    # regularization overwrites
    encoder_dropout: float = 0
    post_mlp_drop: float = 0
    attention_dropout: float = 0
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0

    use_fc_norm: bool = True
    prediction_mode: PredictionMode = PredictionMode.MEAN_POOLING

    no_decay_blocks: bool = True


class ImageClassificationModel(nn.Module):
    def __init__(self, cfg: ImageClassificationConfig):
        super().__init__()
        self.cfg = cfg
        self.linear_classifier = cfg.linear_classifier

        pretrained_args = torch.load(cfg.model_path)['hyper_parameters']['cfg']

        with open_dict(pretrained_args.model):
            pretrained_args.model.drop_path_rate = cfg.drop_path_rate
            if cfg.norm_eps is not None:
                pretrained_args.model.norm_eps = cfg.norm_eps

        cfg.pretrained_model_args = pretrained_args

        model_blocks = pretrained_args.model["depth"]
        with open_dict(pretrained_args):
            dpr = np.linspace(0, cfg.drop_path_rate, model_blocks).tolist()
            
            pretrained_args.model["start_drop_path_rate"] = dpr[0]
            pretrained_args.model["end_drop_path_rate"] = dpr[-1]

            pretrained_args.model["encoder_dropout"] = cfg.encoder_dropout
            pretrained_args.model["post_mlp_drop"] = cfg.post_mlp_drop
            pretrained_args.model["attention_dropout"] = cfg.attention_dropout
            pretrained_args.model["activation_dropout"] = cfg.activation_dropout
            pretrained_args.model["dropout_input"] = cfg.dropout_input
            pretrained_args.model["layerdrop"] = cfg.layerdrop

        self.model:KDMMData2Vec = KDData2VecPreTrainingLightningModule.load_from_checkpoint(self.cfg.model_path,
                                                                                            cfg=pretrained_args).model
        self.model.prepare_fine_tuning(keep_modes=[Modality.IMAGE])

        self.linear_classifier = cfg.linear_classifier

        if self.linear_classifier:
            self.model.requires_grad_(False)

        self.fc_norm = None
        if self.cfg.use_fc_norm:
            self.fc_norm = nn.LayerNorm(768, eps=1e-6)
            nn.init.constant_(self.fc_norm.bias, 0)
            nn.init.constant_(self.fc_norm.weight, 1.0)

        self.head = nn.Linear(768, cfg.num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, 0)

        if self.model.norm is not None:
            for pn, p in self.model.norm.named_parameters():
                if len(p.shape) == 1 or pn.endswith(".bias"):
                    p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        if self.fc_norm is not None:
            for pn, p in self.fc_norm.named_parameters():
                if len(p.shape) == 1 or pn.endswith(".bias"):
                    p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        for pn, p in self.head.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias"):
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        mod_encs = list(self.model.modality_encoders.values())
        assert len(mod_encs) == 1, len(mod_encs)
        blocks = list(mod_encs[0].context_encoder.blocks) + list(self.model.blocks)

        num_layers = len(blocks) + 1
        layer_scales = list(
            cfg.layer_decay ** (num_layers - i) for i in range(num_layers + 1)
        )

        for n, p in self.model.named_parameters():
            optimizer_override_dict = {}

            if len(p.shape) == 1 or n.endswith(".bias"):
                optimizer_override_dict["weight_decay_scale"] = 0

            p.optim_overrides = {"optimizer": optimizer_override_dict}

        if cfg.layer_decay > 0:
            for i, b in enumerate(blocks):
                lid = i + 1
                if layer_scales[lid] == 1.0:
                    continue

                for n, p in b.named_parameters():
                    optim_override = getattr(p, "optim_overrides", {})
                    if "optimizer" not in optim_override:
                        optim_override["optimizer"] = {}

                    if cfg.no_decay_blocks:
                        optim_override["optimizer"]["lr_scale"] = layer_scales[lid]
                        p.optim_overrides = optim_override
                    else:
                        optim_override["optimizer"] = {
                            "lr_scale": layer_scales[lid]
                        }
                        p.optim_overrides = optim_override

    def forward(
        self,
        image,
    ):
        
        if self.linear_classifier:
            with torch.no_grad():
                x = self.model_forward(image)
        else:
            x = self.model_forward(image)

        if self.cfg.prediction_mode == PredictionMode.MEAN_POOLING:
            x = x.mean(dim=1)
        elif self.cfg.prediction_mode == PredictionMode.CLS_TOKEN:
            x = x[:, 0]
        elif self.cfg.prediction_mode == PredictionMode.LIN_SOFTMAX:
            dtype = x.dtype
            x = F.logsigmoid(x.float())
            x = torch.logsumexp(x + x, dim=1) - torch.logsumexp(x + 1e-6, dim=1)
            x = x.clamp(max=0)
            x = x - torch.log(-(torch.expm1(x)))
            x = torch.nan_to_num(x, nan=0, posinf=0, neginf=0)
            x = x.to(dtype=dtype)
        else:
            raise Exception(f"unknown prediction mode {self.cfg.prediction_mode.name}")

        if self.fc_norm is not None:
            x = self.fc_norm(x)

        return self.head(x)


    def model_forward(self, imgs):
        return self.model.extract_features(
            image=imgs,
            modes=[Modality.IMAGE],
            remove_extra_tokens=(
                self.cfg.prediction_mode != PredictionMode.CLS_TOKEN
            ),
        )["x"]
