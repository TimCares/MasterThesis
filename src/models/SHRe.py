import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, List
import numpy as np
import os
from utils import load_pretrained_d2v_model, get_d2v_image_embed, get_d2v_text_embed
from omegaconf import OmegaConf
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from modules import MOMEAltBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import timm
from timm.layers import PatchEmbed, Mlp
from fairseq.data.dictionary import Dictionary

logger = logging.getLogger(__name__)


class SHRePreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.itc = self.cfg.model.itc

        self.model = SHRe(cfg=self.cfg.model)

        self.teacher = timm.create_model('resnet50.a1_in1k', pretrained=True)
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

        target = self.teacher(batch['image'])
        output_dict = self(batch) # call "forward"
        target = torch.nn.functional.log_softmax(target, dim=1)
        input_text = torch.nn.functional.log_softmax(output_dict['encoder_out_text'], dim=1)
        input_image = torch.nn.functional.log_softmax(output_dict['encoder_out_image'], dim=1)

        kl_loss = []
        for input in [input_text, input_image]:
            kl_loss_ = F.kl_div(input=input, target=target, log_target=True)
            kl_loss.append(kl_loss_)
        kl_loss = sum(kl_loss) / 2
        
        itc_loss = self.itc_loss(text_features=output_dict['x_text'], image_features=output_dict['x_image'])
        self.log(f"{stage}/itc_loss", itc_loss)
        
        loss = kl_loss + itc_loss

        self.log(f"{stage}/kl_loss", kl_loss)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def itc_loss(self, text_features:torch.Tensor, image_features:torch.Tensor, stage:str='train') -> torch.Tensor:
        # scale = self.model.logit_scale.exp()
        
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        self._log_similarity(logits_per_image, stage)

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
class SHReConfig():
    pretrained_path:str = '../models'

    itc: bool = True

    embed_dim: int = 768

    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    encoder_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    post_mlp_drop: float = 0.1
    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_norm_first: bool = False
    dropout_input: float = 0.0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    layerdrop: float = 0.0

    seed: int = 42

class SHRe(nn.Module):
    def __init__(self,
                 cfg: SHReConfig,
                 ):
        super(SHRe, self).__init__()
        self.cfg = cfg
        self.supported_modalities = [Modality.IMAGE, Modality.TEXT]

        self.patch_embed = PatchEmbed(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=self.cfg.embed_dim,
        )
        self.img_pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, self.cfg.embed_dim), requires_grad=False)
        self.img_norm = nn.LayerNorm(self.cfg.embed_dim, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.embed_dim))

        d = Dictionary.load(os.path.join(self.cfg.pretrained_path, 'dict.txt'))
        self.token_embed = nn.Embedding(len(d), self.cfg.embed_dim, d.pad())
        self.text_pos_embed = nn.Embedding(512, self.cfg.embed_dim, d.pad())
        self.text_norm = nn.LayerNorm(self.cfg.embed_dim, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)

        self.shared = Mlp(
            in_features=self.cfg.embed_dim,
            hidden_features=self.cfg.embed_dim*self.cfg.mlp_ratio,
            out_features=1000,
            norm_layer=partial(nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine),
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        blocks:List[MOMEAltBlock] = []
        for i in range(self.cfg.depth):
            blocks.append(self.make_block(drop_path=dpr[i], multimodal=False))

        self.blocks:nn.ModuleList = nn.ModuleList(blocks)

        self.layerdrop = self.cfg.layerdrop

        self.apply(init_bert_params)

        self._init_from_pretrained()
        
    def make_block(self, drop_path, dim=None, heads=None, multimodal=False, with_fuzed=False, shared_attn=True):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return MOMEAltBlock(
            self.cfg.embed_dim if dim is None else dim,
            self.cfg.num_heads if heads is None else heads,
            self.cfg.mlp_ratio,
            qkv_bias=True,
            drop=self.cfg.encoder_dropout,
            attn_drop=self.cfg.attention_dropout,
            mlp_drop=self.cfg.activation_dropout,
            post_mlp_drop=self.cfg.post_mlp_drop,
            drop_path=drop_path,
            norm_layer=make_layer_norm,
            layer_norm_first=self.cfg.layer_norm_first,
            multimodal=multimodal,
            with_fuzed=with_fuzed,
            shared_attn=shared_attn,
        )

    def _init_from_pretrained(self) -> None:
        for modality in self.supported_modalities:
            state_dict_name = self.cfg.pretrained[modality.name.lower()]
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)

            for i in range(self.cfg.depth):
                self.blocks[i].init_from_pretrained(
                    pretained_block=d2v_model.blocks[i],
                    modality=modality,
                    init_attention=True,
                )

        text_embed_sd = get_d2v_text_embed()
        img_embed_sd = get_d2v_image_embed()
        embed_sd = text_embed_sd | img_embed_sd
        _, unexpected_keys = self.load_state_dict(embed_sd, strict=False)
        assert len(unexpected_keys) == 0
        for module in [self.patch_embed, self.img_pos_embed, self.cls_token, self.img_norm,
                       self.token_embed, self.text_pos_embed, self.text_norm]:
            self._freeze(module)

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
        x = self.patch_embed(image)
        x = x + self.cls_token.expand(x.shape[0], -1, -1)
        x = x + self.img_pos_embed
        x = self.img_norm(x)

        if self.dropout_input is not None:
            x = self.dropout_input(x)
        
        for i in range(self.cfg.depth):
            x, _ = self.blocks[i](
                x,
                modality=Modality.IMAGE,
            )
        
        return self.encode_shared(x)

    def encode_text(self, text, padding_mask):
        x = self.token_embed(text)
        x = x + self.text_pos_embed(text)
        x = self.text_norm(x)

        if self.dropout_input is not None:
            x = self.dropout_input(x)
        
        for i in range(self.cfg.depth):
            x, _ = self.blocks[i](
                x,
                modality=Modality.TEXT,
                padding_mask=padding_mask,
            )
        
        return self.encode_shared(x)
    
    def encode_shared(self, x):
        out_dict = dict()
        x = self.shared(x[:, 0])
        out_dict["encoder_out"] = x
        x = x / x.norm(dim=-1, keepdim=True)
        out_dict["x"] = x
        return out_dict

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True
        module.train()
