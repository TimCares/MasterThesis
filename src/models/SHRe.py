import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, List
import numpy as np
import os
from utils import load_pretrained_d2v_model
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from transformers import ViTForImageClassification
from timm.layers import Mlp

logger = logging.getLogger(__name__)

class Mlp_(Mlp):
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x_interm = x
        x = self.fc2(x)
        x = self.drop2(x)
        return x_interm, x


class SHRePreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.itc = self.cfg.model.itc

        self.model = SHRe(cfg=self.cfg.model)

        self.teacher = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.model._freeze(self.teacher)

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

        with torch.no_grad():
            target = self.teacher({'pixel_values': batch['image']}).logits
        
        output_dict = self(batch) # call "forward"
        target = torch.nn.functional.log_softmax(target, dim=1)
        input_text = torch.nn.functional.log_softmax(output_dict['encoder_out_text'], dim=1)
        input_image = torch.nn.functional.log_softmax(output_dict['encoder_out_image'], dim=1)

        kl_loss1 = F.kl_div(input=input_text, target=target, log_target=True, reduction='batchmean')
        kl_loss2 = F.kl_div(input=input_image, target=target, log_target=True, reduction='batchmean')
        kl_loss = (kl_loss1 + kl_loss2) / 2
        
        if self.itc:
            itc_loss1 = self.itc_loss(text_features=output_dict['x_interm_text'], image_features=output_dict['x_interm_image'])
            itc_loss2 = self.itc_loss(text_features=output_dict['x_text'], image_features=output_dict['x_image'])
            itc_loss = (itc_loss1 + itc_loss2) / 2
            self.log(f"{stage}/itc_loss", itc_loss)
        else:
            itc_loss = torch.tensor(0.0).to(kl_loss.device)
        
        loss = kl_loss + itc_loss

        self.log(f"{stage}/kl_loss", kl_loss)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def itc_loss(self, text_features:torch.Tensor, image_features:torch.Tensor, stage:str='train') -> torch.Tensor:
        scale = self.model.logit_scale.exp()
        
        logits_per_image = image_features @ text_features.t()
        self._log_similarity(logits_per_image, stage)
        logits_per_image = logits_per_image*scale
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
    
    def _log_similarity(self, logits: torch.Tensor, stage:str='train') -> None:
        diagonal_mask = torch.eye(logits.size(0)).bool()
        mean_pos_sim = logits[diagonal_mask].mean()
        mean_neg_sim = logits[~diagonal_mask].mean()
        self.log(f"{stage}/itc_mean_pos_similarity", mean_pos_sim)
        self.log(f"{stage}/itc_mean_neg_similarity", mean_neg_sim)

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

    itc: bool = True

    embed_dim: int = 768

    depth: int = 6
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    norm_affine: bool = True

class SHRe(nn.Module):
    def __init__(self,
                 cfg: SHReConfig,
                 ):
        super(SHRe, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)
        if self.cfg.itc:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.shared = Mlp_(
            in_features=self.cfg.embed_dim,
            hidden_features=int(self.cfg.embed_dim*self.cfg.mlp_ratio),
            out_features=1000,
            norm_layer=make_layer_norm,
        )
        self.norm = make_layer_norm(self.cfg.embed_dim)

        self.apply(init_bert_params)

        self.text_model = load_pretrained_d2v_model(state_dict_path=os.path.join(self.cfg.pretrained_path, self.cfg.pretrained.text),
                                                    remove_dropout=True)
        self.text_model.blocks = self.text_model.blocks[:self.cfg.depth]
        self.image_model = load_pretrained_d2v_model(state_dict_path=os.path.join(self.cfg.pretrained_path, self.cfg.pretrained.image),
                                                    remove_dropout=True)
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
        x = self.image_model.extract_features(
            source=image,
            remove_extra_tokens=False,
        )['x']
        
        return self.encode_shared(x)

    def encode_text(self, text, padding_mask):
        x = self.text_model.extract_features(
            source=text,
            padding_mask=padding_mask,
            remove_extra_tokens=False,
        )['x']
        
        return self.encode_shared(x)
    
    def encode_shared(self, x):
        out_dict = dict()
        x_interm, x = self.shared(x[:, 0])
        x = self.norm(x)
        out_dict["encoder_out"] = x
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
