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
        
        image1 = batch['image1']
        image2 = batch['image2']
        input_dict = {
            'image1': image1,
            'image2': image2,
        }

        output_dict = self(input_dict)
        
        embedding_scores1 = output_dict['encoding_scores'][:image1.shape[0]]
        embedding_scores2 = output_dict['encoding_scores'][image1.shape[0]:]

        kl_loss = F.kl_div(
            input=F.log_softmax(embedding_scores1, dim=-1),
            target=F.log_softmax(embedding_scores2, dim=-1),
            log_target=True,
            reduction='batchmean'
        )

        me_max_loss = 0
        for scores in [embedding_scores1, embedding_scores2]:
            scores = F.softmax(scores, dim=-1).mean(dim=0)
            me_max_ = - torch.sum(torch.log(scores**(-scores)))
            me_max_loss += me_max_
        me_max_loss /= 2

        vq_loss = output_dict['vq_loss']

        loss = vq_loss + kl_loss + me_max_loss * self.cfg.model.me_max_weight

        self.log(f"{stage}/vq_loss", vq_loss, prog_bar=True)
        self.log(f"{stage}/kl_loss", kl_loss, prog_bar=True)
        self.log(f"{stage}/me_max_loss", me_max_loss, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
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
    n_codebook_embed: int = 1024
    vq_dim: int = 32
    vq_decay: float = 0.95 # 0.99
    temperature: float = 0.1
    me_max_weight: float = 2.0

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

    def forward(
        self,
        image1:torch.Tensor,
        image2:torch.Tensor,
    ):
        result = []
        for batch in [image1, image2]:
            cls_token = self.forward_beitv2(batch)
            to_quantizer_features = self.embed_to_vq_proj(cls_token)
            result.append(to_quantizer_features)
        features = torch.cat(result, dim=0)

        quantize_result_dict = self.quantize(features, return_scores=True)

        d = quantize_result_dict['encoding_scores']
        encoding_scores = d*self.cfg.temperature

        out_dict = {
            'x': quantize_result_dict['z_q'],
            'embed_ind': quantize_result_dict['encoding_indices'],
            'vq_loss': quantize_result_dict['loss'],
            'encoding_scores': encoding_scores,
        }
        return out_dict
    
    def quantize_image(
        self,
        image:torch.Tensor,
    ):
        cls_token = self.forward_beitv2(image)

        to_quantizer_features = self.embed_to_vq_proj(cls_token)

        quantize_result_dict = self.quantize(to_quantizer_features, return_scores=True)

        d = quantize_result_dict['encoding_scores']
        encoding_scores = d*self.cfg.temperature

        out_dict = {
            'embed_ind': quantize_result_dict['encoding_indices'],
            'encoding_scores': encoding_scores,
        }
        return out_dict
    
    @torch.no_grad()
    def forward_beitv2(self, image:torch.Tensor):
        bool_masked_pos = torch.zeros((image.shape[0], self.beitv2.patch_embed.num_patches),
                                      dtype=torch.bool).to(image.device)
        
        target = self.beitv2.forward_features(
            x=image,
            bool_masked_pos=bool_masked_pos,
        )[:, 0]
        
        return target


MODEL_REGISTRY['image_vq'] = {
    'cfg': ImageVQConfig,
    'module': ImageVQLightningModule
}
