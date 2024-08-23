import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as distributed
from functools import partial
from typing import Dict, Any, Optional
import os
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2.norm_ema_quantizer import norm_ema_inplace, ema_inplace, l2norm, EmbeddingEMA
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings, BertLMPredictionHead, BertModel
from timm.models.vision_transformer import Block
from omegaconf import OmegaConf
from modules.layers import Attention
from timm.models.vision_transformer import LayerScale
from timm.layers import Mlp, DropPath
from . import MODEL_REGISTRY
from utils import freeze_module, load_beit2_teacher


logger = logging.getLogger(__name__)

class ImageVQLLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = ImageVQL(cfg=self.cfg.model)

        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='train')
    
    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):
        input_dict = {key:batch[key] for key in ['image', 'text', 'padding_mask']}

        output_dict = self(input_dict) # call "forward"

        vq_loss = output_dict['vq_loss']

        input = output_dict['x'].view(-1, 30522)
        target = batch['targets'].view(-1)
        
        mlm_loss = F.cross_entropy(
            input=input,
            target=target,
            ignore_index=-100,
        )

        loss = vq_loss + mlm_loss

        self.log(f"{stage}/vq_loss", vq_loss, prog_bar=True)
        self.log(f"{stage}/mlm_loss", mlm_loss, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)

        input_ = input.argmax(dim=-1)[target != -100]
        target_ = target[target != -100]
        mlm_acc = (input_ == target_).float().mean()
        self.log(f"{stage}/mlm_acc", mlm_acc)
        
        return {'loss': loss, 'embed_ind': output_dict['embed_ind']}

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
    
    def _get_param_groups(self):
        no_wd_keys = {'quantize.embedding.weight', 'text_embeddings', 'beitv2.cls_token'}
        
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
    model_path:str = "/workspace/models"
    model_name:str = "beitv2_base_patch16_224_pt1k.pth"
    drop_path_rate: float = 0.1
    use_shared_rel_pos_bias:bool = True
    use_abs_pos_emb: bool = False
    vocab_size: int = 8192
    init_values: float =  0.1

@dataclass
class ImageVQLConfig(): 
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)
    decoder_depth: int = 3
    n_codebook_embed: int = 1000
    vq_dim: int = 32
    vq_decay: float = 0.96 # 0.99

class ImageVQL(nn.Module):
    def __init__(self,
                 cfg: ImageVQLConfig,
                 ):
        super(ImageVQL, self).__init__()
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
        self.embed_to_vq_proj.apply(self.beitv2._init_weights)

        self.vq_to_embed_proj = nn.Linear(dim, dim)
        self.vq_to_embed_proj.apply(self.beitv2._init_weights)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=self.cfg.n_codebook_embed, embedding_dim=self.cfg.vq_dim, beta=1.0, kmeans_init=True, decay=self.cfg.vq_decay,
        )

        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=dim,
            max_position_embeddings=64,
            hidden_dropout_prob=0.0,
            position_embedding_type="absolute",
            pad_token_id=0,
            layer_norm_eps=1e-6,
        )

        self.text_embeddings = BertModel.from_pretrained('bert-base-uncased').embeddings
        freeze_module(self.text_embeddings)

        self.decoder = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=self.beitv2.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                proj_drop=0.0,
                attn_drop=0.0,
                init_values=None,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(self.cfg.decoder_depth)]
        )
        self.decoder.apply(self.beitv2._init_weights)

        self.lm_head = BertLMPredictionHead(bert_config)
        self.lm_head.apply(self.beitv2._init_weights)

    def forward(
        self,
        image:torch.Tensor,
        text:torch.Tensor,
        padding_mask:torch.Tensor=None,
    ):
        result_dict = self.quantize_image(image)

        with torch.no_grad():
            x = self.text_embeddings(input_ids=text)
        
        x[:, 0] = result_dict['x']

        for blk in self.decoder:
            x = blk(x, mask=padding_mask)

        x = self.lm_head(x)

        out_dict = {
            'x': x,
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
        quantize = self.vq_to_embed_proj(quantize)

        out_dict = {
            'x': quantize,
            'embed_ind': embed_ind,
            'vq_loss': loss,
        }
        return out_dict
    
    def prepare_inference(self):
        del self.text_embeddings
        del self.decoder
        del self.lm_head
    

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
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
    

MODEL_REGISTRY['iqv-l'] = {
    'cfg': ImageVQLConfig,
    'module': ImageVQLLightningModule
}
