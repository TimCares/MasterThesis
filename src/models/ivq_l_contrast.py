import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as distributed
import numpy as np
from typing import Dict, Any, Optional
import os
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2.norm_ema_quantizer import norm_ema_inplace, ema_inplace, l2norm, EmbeddingEMA
from transformers.models.bert.modeling_bert import BertModel
from omegaconf import OmegaConf
from modules.layers import Attention
from timm.models.vision_transformer import LayerScale
from timm.layers import Mlp, DropPath
from modules import ClipLoss
from . import MODEL_REGISTRY
from utils import freeze_module, load_beit2_teacher


logger = logging.getLogger(__name__)

class ImageVQLContrastLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = ImageVQLContrast(cfg=self.cfg.model)

        self.register_buffer("embedding_usage", torch.zeros(self.cfg.model.n_codebook_embed))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.save_hyperparameters()

    def on_train_start(self):
        logger.info(f'World size: {self.trainer.world_size}')
        logger.info(f'Local rank: {self.trainer.local_rank}')
        self.clip_loss = ClipLoss(
            cache_labels=True,
            rank=self.trainer.local_rank,
            world_size=self.trainer.world_size
        )

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):
        input_dict = {key:batch[key] for key in ['image', 'text', 'padding_mask']}

        output_dict = self(input_dict) # call "forward"

        x_image = output_dict['x_image']
        x_text = output_dict['x_text']
        x_image = x_image / x_image.norm(dim=-1, keepdim=True)
        x_text = x_text / x_text.norm(dim=-1, keepdim=True)

        itc_out = self.clip_loss(
            image_features=x_image,
            text_features=x_text,
            logit_scale=self.logit_scale.exp(),
        )

        self.log_itc_acc(itc_out['logits_per_image'], itc_out['logits_per_text'], itc_out['targets'], stage)
        itc_loss = itc_out['loss']
        self.log(f"{stage}/itc_loss", itc_loss)

        vq_loss = output_dict['vq_loss']
        self.log(f"{stage}/vq_loss", vq_loss)

        loss = itc_loss + vq_loss
        self.log(f"{stage}/loss", loss)

        unique_indices, counts = output_dict['embed_ind'].unique(return_counts=True)
        self.embedding_usage[unique_indices] += counts

        self.log_codebook_usage(output_dict, stage=stage)
        
        return loss
    
    def log_itc_acc(self, logits_per_image, logits_per_text, target, stage, key_prefix=""):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        if key_prefix != "":
            key_prefix = key_prefix + "_"
        self.log(f"{stage}/{key_prefix}itc_text_acc", text_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_image_acc", img_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)
    
    def log_codebook_usage(self, output_dict:Dict[str, Any], stage:str='train'):
        embeds_utilized = output_dict['embed_ind'].unique().numel()
        utilization_percentage = (embeds_utilized / output_dict['x'].shape[0]) * 100
        self.log(f"{stage}/codebook_utilization", utilization_percentage)

    def on_validation_start(self):
        mean_usage = torch.mean(self.embedding_usage.float())
        std_dev_usage = torch.std(self.embedding_usage.float())
        cv_usage = std_dev_usage / mean_usage
        min_usage = torch.min(self.embedding_usage.float())
        no_usage = torch.sum(self.embedding_usage == 0).float() / self.embedding_usage.numel()

        self.log("mean_codebook_usage", mean_usage)
        self.log("std_dev_codebook_usage", std_dev_usage)
        self.log("cv_codebook_usage", cv_usage)
        self.log("min_codebook_usage", min_usage)
        self.log("no_usage_pct", no_usage)

        self.embedding_usage.zero_()

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
        no_wd_keys = {'quantize.embedding.weight', 'text_embeddings', 'beitv2.cls_token'}
        
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if any(no_wd_key in name for no_wd_key in no_wd_keys):
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params
    
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith('bert.'):
                del sd[k]
        return sd
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)

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
class ImageVQLContrastConfig(): 
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)
    n_codebook_embed: int = 8192
    vq_dim: int = 32
    vq_decay: float = 0.95 # 0.99

class ImageVQLContrast(nn.Module):
    def __init__(self,
                 cfg: ImageVQLContrastConfig,
                 ):
        super(ImageVQLContrast, self).__init__()
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
        self.bert_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )

        self.embed_to_vq_proj.apply(self.beitv2._init_weights)
        self.embed_to_vq_proj.apply(self.beitv2._init_weights)
        self.bert_proj.apply(self.beitv2._init_weights)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=self.cfg.n_codebook_embed, embedding_dim=self.cfg.vq_dim, beta=1.0, kmeans_init=True, decay=self.cfg.vq_decay,
        )

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        freeze_module(self.bert)

    def forward(
        self,
        image:torch.Tensor,
        text:torch.Tensor,
        padding_mask:torch.Tensor=None,
    ):
        result_dict = self.quantize_image(image)

        cls_token = self.vq_to_embed_proj(result_dict['x'])

        bert_cls_token = self.bert(input_ids=text, attention_mask=1-padding_mask).last_hidden_state[:, 0]
        bert_cls_token = self.bert_proj(bert_cls_token)

        out_dict = {
            'x_image': cls_token,
            'x_text': bert_cls_token,
            'embed_ind': result_dict['embed_ind'],
            'vq_loss': result_dict['vq_loss'],
        }
        return out_dict
    
    def quantize_image(self, image:torch.Tensor):
        with torch.no_grad():
            bool_masked_pos = torch.zeros((image.shape[0], self.beitv2.patch_embed.num_patches), dtype=torch.bool).to(image.device)
            x = self.beitv2.forward_features(x=image, bool_masked_pos=bool_masked_pos)[:, 0] # cls token

        to_quantizer_features = self.embed_to_vq_proj(x)
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        out_dict = {
            'x': quantize,
            'embed_ind': embed_ind,
            'vq_loss': loss,
        }
        return out_dict
    
    def prepare_inference(self):
        del self.bert

    
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
        self.all_reduce_fn = distributed.all_reduce
    
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
    

MODEL_REGISTRY['iqv-l-contrast'] = {
    'cfg': ImageVQLContrastConfig,
    'module': ImageVQLContrastLightningModule
}
