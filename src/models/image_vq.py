import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as distributed
from functools import partial
from typing import Dict, Any
import os
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2.norm_ema_quantizer import norm_ema_inplace, ema_inplace, l2norm, EmbeddingEMA
from beit2.modeling_finetune import Block
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
        if 'target' in batch:
            batch.pop('target') # unused, layer activations are the targets

        if 'image_teacher' in batch:
            batch.pop('image_teacher')

        output_dict = self(batch) # call "forward"

        target = output_dict['x_beitv2']
        rec = output_dict['x']
        
        target = target / target.norm(dim=-1, keepdim=True)
        rec = rec / rec.norm(dim=-1, keepdim=True)
        rec_loss = (1 - (target * rec).sum(-1)).mean()

        loss = output_dict['vq_loss'] + rec_loss

        self.log(f"{stage}/vq_loss", output_dict['vq_loss'])
        self.log(f"{stage}/rec_loss", rec_loss)
        self.log(f"{stage}/loss", loss)
        
        return loss

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
        if 'deepspeed' in self.cfg.lightning_trainer:
            if self.cfg.lightning_trainer.deepspeed.offload_optimizer:
                from deepspeed.ops.adam import DeepSpeedCPUAdam
                opt_cls = DeepSpeedCPUAdam
                optim_args['model_params'] = optim_args.pop("params")
            else:
                from deepspeed.ops.adam import FusedAdam
                opt_cls = FusedAdam
        else:
            opt_cls = torch.optim.AdamW
            
        optimizer = opt_cls(**optim_args)
        
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
        no_wd_keys = {'quantize.embedding.weight', 'decoder.cls_token',
                      'decoder.pos_embed', 'encoder.cls_token', 'encoder.pos_embed'}
        
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if any(no_wd_key in name for no_wd_key in no_wd_keys):
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class BEiTv2Config():
    model_path:str
    model_name:str = "beitv2_base_patch16_224_pt1k.pth"
    drop_path_rate: float = 0.1
    use_shared_rel_pos_bias:bool = True
    use_abs_pos_emb: bool = False
    vocab_size: int = 8192
    init_values: float =  0.1

@dataclass
class ImageVQConfig(): 
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)
    decoder_depth: int = 3
    pass_through_layer_idx: int = 2
    n_classes: int = 1000
    vq_dim: int = 32
    vq_decay: float = 0.95 # 0.99

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

        self.decoder = nn.ModuleList([
            Block(
                dim=self.cfg.vq_dim,
                num_heads=self.beitv2.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True, qk_scale=None,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.0,
                window_size=self.beitv2.patch_embed.patch_shape,
            )
            for _ in range(self.cfg.decoder_depth)]
        )

        dim = self.beitv2.embed_dim
        self.embed_to_vq_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, self.cfg.vq_dim),
        )
        self.vq_to_embed_proj = nn.Sequential(
            nn.Linear(self.cfg.vq_dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

        self.apply(self.beitv2._init_weights)

        self.beitv2:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        freeze_module(self.beitv2)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=self.cfg.n_classes, embedding_dim=self.cfg.vq_dim, beta=1.0, kmeans_init=True, decay=self.cfg.vq_decay,
        )

    def forward(
        self,
        image:torch.Tensor,
    ):
        with torch.no_grad():
            beitv2_out = self.forward_beitv2(image)

        x = beitv2_out['x']
        x_cache = beitv2_out['x_cache']
                
        to_quantizer_features = self.embed_to_vq_proj(x[:, 0]) # cls token

        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        x = torch.cat((quantize.unsqueeze(1), x_cache), dim=1)

        for blk in self.decoder:
            x = blk(x, rel_pos_bias=beitv2_out['rel_pos_bias'])
        
        x = self.vq_to_embed_proj(x)

        out_dict = {
            'x': x,
            'x_beitv2': beitv2_out['x'],
            'embed_ind': embed_ind,
            'vq_loss': loss,
        }
        return out_dict
    
    def forward_beitv2(self, image:torch.Tensor):
        x = self.visual_embed(image)
        
        rel_pos_bias = self.beitv2.rel_pos_bias() if self.beitv2.rel_pos_bias is not None else None
        for l, blk in enumerate(self.beitv2.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if l == self.cfg.pass_through_layer_idx:
                x_cache = x[:, 1:] # blk.norm1(x)[:, 1:]

        return {
            'x': x,
            'x_cache': x_cache,
            'rel_pos_bias': rel_pos_bias,
        }

    def visual_embed(self, x:torch.Tensor):
        bool_masked_pos = torch.zeros((x.shape[0], self.beitv2.patch_embed.num_patches), dtype=torch.bool).to(x.device)

        x = self.beitv2.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.beitv2.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.beitv2.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.beitv2.pos_embed is not None:
            x = x + self.beitv2.pos_embed
        x = self.beitv2.pos_drop(x)
        
        return x
    
class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        
        # learnable = True if orthogonal_reg_weight > 0 else False
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            self.all_reduce_fn = nn.Identity()
    
    def reset_cluster_size(self, device):
        if self.statistic_code_usage:
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            self.cluster_size = self.cluster_size.to(device)

    def forward(self, z):
        z = l2norm(z)
        z_flattened = z
        
        self.embedding.init_embed_(z_flattened)
        
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'
        
        encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(encoding_indices)#.view(z.shape)
        
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        
        if not self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(0)
                self.all_reduce_fn(cluster_size)
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        
        if self.training and self.embedding.update:
            #EMA cluster size

            bins = encodings.sum(0)
            self.all_reduce_fn(bins)

            # self.embedding.cluster_size_ema_update(bins)
            ema_inplace(self.cluster_size, bins, self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = z_flattened.t() @ encodings
            self.all_reduce_fn(embed_sum)
                        
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = l2norm(embed_normalized)
            
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        # compute loss for embedding
        loss = self.beta * F.mse_loss(z_q.detach(), z) 
        
        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, encoding_indices
    

MODEL_REGISTRY['image_vq'] = {
    'cfg': ImageVQConfig,
    'module': ImageVQLightningModule
}
