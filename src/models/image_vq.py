import torch
from torch import nn
from functools import partial
from typing import Dict, Any
import os
import json
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2.modeling_finetune import Block
from .vq import NormEMAVectorQuantizer
from omegaconf import OmegaConf
from . import MODEL_REGISTRY
from utils import freeze_module, load_beit2_teacher

logger = logging.getLogger(__name__)

class ImageVQLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = ImageVQ(cfg=self.cfg.model)

        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):
        
        image = batch['image']
        input_dict = {
            'image': image,
        }

        output_dict = self(input_dict)
        rec = output_dict['x']
        target = output_dict['target']

        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        vq_loss = output_dict['vq_loss']

        loss = vq_loss + rec_loss

        self.log(f"{stage}/vq_loss", vq_loss, prog_bar=True)
        self.log(f"{stage}/rec_loss", rec_loss, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return {'loss': loss, 'embed_ind': output_dict['embed_ind']}

    def configure_optimizers(self):
        ws = torch.cuda.device_count()
        logger.info(f"[Optimizer]: World size is {ws}")
        if 'lr' in self.cfg.optimizer:
            learning_rate = self.cfg.optimizer.lr
        else:
            assert 'base_lr' in self.cfg.optimizer
            learning_rate = self.cfg.optimizer.base_lr * (self.cfg.data.batch_size*ws) / 256
            logger.info(f"[Optimizer]: Base Learning rate is {self.cfg.optimizer.base_lr}")
        logger.info(f"[Optimizer]: Learning rate is {learning_rate}")
        params = self._get_param_groups(lr=learning_rate)
        optim_args = {
            "params": params,
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
        no_wd_keys = {'quantize.embedding.weight', 'decoder.cls_token',
                      'decoder.pos_embed', 'encoder.cls_token', 'encoder.pos_embed'}
        
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or any(no_wd_key in name for no_wd_key in no_wd_keys):
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
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class BEiTv2Config():
    model_path:str = "/workspace/models"
    model_name:str = "beitv2_base_patch16_224_pt1k.pth"
    drop_path_rate: float = 0.1
    use_shared_rel_pos_bias:bool = True
    use_abs_pos_emb: bool = False
    vocab_size: int = 8192
    init_values: float =  0.1

@dataclass
class ImageVQConfig():
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)
    decoder_depth: int = 1
    n_codebook_embed: int = 1024
    vq_dim: int = 16
    vq_decay: float = 0.99
    mask_ratio: float = 0.90

class ImageVQ(nn.Module):
    def __init__(self,
                 cfg: ImageVQConfig,
                 ):
        super(ImageVQ, self).__init__()
        self.cfg = cfg
        
        beit2_kwargs = OmegaConf.to_container(self.cfg.beitv2, resolve=True)
        sd_path = beit2_kwargs.pop("model_path")
        sd_name = beit2_kwargs.pop("model_name")
        beit_path = os.path.join(sd_path, sd_name)

        self.beitv2:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        freeze_module(self.beitv2)
        embed_dim = self.beitv2.embed_dim

        self.decoder = nn.ModuleList([
            Block(
                dim=self.beitv2.embed_dim,
                num_heads=self.beitv2.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
                window_size=self.beitv2.patch_embed.patch_shape,
            )
            for _ in range(self.cfg.decoder_depth)]
        )

        self.quantize = NormEMAVectorQuantizer(
            n_embed=self.cfg.n_codebook_embed, embedding_dim=self.cfg.vq_dim, beta=1.0,
            kmeans_init=True, decay=self.cfg.vq_decay,
        )

        self.embed_to_vq_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, self.cfg.vq_dim),
        )
        self.embed_to_vq_proj.apply(self.beitv2._init_weights)
        self.vq_to_embed_proj = nn.Linear(self.cfg.vq_dim, embed_dim)
        self.vq_to_embed_proj.apply(self.beitv2._init_weights)

    def forward(
        self,
        image:torch.Tensor,
    ):
        x_beit = self.forward_beitv2(image)
        to_quantizer_features = self.embed_to_vq_proj(x_beit[:, 0])

        quantize_result_dict = self.quantize(to_quantizer_features)
        quantize = quantize_result_dict['z_q']
        quantize = self.vq_to_embed_proj(quantize)

        mask = self.random_masking(x_beit[:, 1:], self.cfg.mask_ratio)
        
        x = self.visual_embed(image, bool_masked_pos=mask, quantize=quantize)

        rel_pos_bias = self.beitv2.rel_pos_bias()
        for blk in self.decoder:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        out_dict = {
            'x': x[:, 0],
            'vq_loss': quantize_result_dict['loss'],
            'target': x_beit[:, 0],
            'embed_ind': quantize_result_dict['encoding_indices'],
        }
        return out_dict
    
    def quantize_image(
        self,
        image:torch.Tensor,
    ):
        cls_token = self.forward_beitv2(image)

        to_quantizer_features = self.embed_to_vq_proj(cls_token)

        quantize_result_dict = self.quantize(to_quantizer_features, return_scores=True)

        out_dict = {
            'embed_ind': quantize_result_dict['encoding_indices'],
            'encoding_scores': quantize_result_dict['encoding_scores'],
        }
        return out_dict
    
    @torch.no_grad()
    def forward_beitv2(self, image:torch.Tensor):
        bool_masked_pos = torch.zeros((image.shape[0], self.beitv2.patch_embed.num_patches),
                                      dtype=torch.bool).to(image.device)
        
        x = self.beitv2.forward_features(
            x=image,
            bool_masked_pos=bool_masked_pos,
        )
        
        return x

    def visual_embed(self, x:torch.Tensor, bool_masked_pos:torch.Tensor, quantize:torch.Tensor):
        with torch.no_grad():
            x = self.beitv2.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        mask_token = self.beitv2.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((quantize.unsqueeze(1), x), dim=1)

        return x

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, _ = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

MODEL_REGISTRY['image_vq'] = {
    'cfg': ImageVQConfig,
    'module': ImageVQLightningModule
}
