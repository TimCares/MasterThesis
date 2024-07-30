import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
import os
from utils import load_pretrained_d2v_model
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from omegaconf import OmegaConf
from . import MODEL_REGISTRY
from modules import Block, CosineCMLILoss
from utils import freeze_module, load_beit2_teacher

logger = logging.getLogger(__name__)

class Sx3HRePreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = Sx3HRe(cfg=self.cfg.model)

        beit2_kwargs = OmegaConf.to_container(self.cfg.beit2, resolve=True)
        sd_path = beit2_kwargs.pop("pretrained_path")

        self.teacher = load_beit2_teacher(
            sd_path=sd_path,
            **beit2_kwargs,
        )
        freeze_module(self.teacher)

        self.c_cmli_loss = CosineCMLILoss(align_margin=self.cfg.align_margin)

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

        image_teacher = batch.pop('image_teacher')

        # no mask
        bool_masked_pos = torch.zeros((image_teacher.shape[0], self.teacher.patch_embed.num_patches), 
                                      dtype=torch.bool).to(image_teacher.device)

        with torch.no_grad():
            target = self.teacher.forward_features(
                x=image_teacher,
                bool_masked_pos=bool_masked_pos,
            )[:, 0]
        
        output_dict = self(batch) # call "forward"
        input_text = output_dict['encoder_out_text']
        input_image = output_dict['encoder_out_image']

        kd_loss1 = F.mse_loss(input=input_text, target=target)
        self.log(f"{stage}/kd_text_loss", kd_loss1)
        kd_loss2 = F.mse_loss(input=input_image, target=target)
        self.log(f"{stage}/kd_image_loss", kd_loss2)
        kd_loss = (kd_loss1 + kd_loss2) / 2
        self.log(f"{stage}/kd_loss", kd_loss)

        perm = self.get_perm(input_text.shape[0]).to(input_text.device)

        image_features, text_features, padding_mask, target = self.make_features(
            image_features=output_dict['x_raw_image'],
            text_features=output_dict['x_raw_text'],
            padding_mask=output_dict['padding_mask_text'],
            perm=perm,
        )
        
        align_loss = self.c_cmli_loss(
            image_features=image_features,
            text_features=text_features,
            padding_mask=padding_mask,
            target=target,
        )['loss']

        self.log(f"{stage}/align_loss", align_loss, prog_bar=True)

        loss = kd_loss + align_loss * self.cfg.align_loss_weight

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def get_perm(self, B):
        true_order = torch.arange(B)
        perm = torch.randperm(B)

        while (true_order==perm).any().item():
            perm = torch.randperm(B)
        return perm

    def make_features(self, image_features, text_features, padding_mask, perm):
        image_features = image_features.repeat(2, 1, 1)
        B = text_features.shape[0]
        text_features = torch.concat([text_features, text_features[perm]], dim=0)
        padding_mask = torch.concat([padding_mask, padding_mask[perm]], dim=0)

        target = torch.full((B*2, ), 1).to(text_features.device)
        target[B:] = -1
        return image_features, text_features, padding_mask, target
    
    def log_itc_acc(self, logits_per_image, logits_per_text, target, stage, key_prefix=""):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        if key_prefix != "":
            key_prefix = key_prefix + "_"
        self.log(f"{stage}/{key_prefix}itc_text_acc", text_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_image_acc", img_itc_acc)
        self.log(f"{stage}/{key_prefix}itc_acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

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
        wd_params, non_wd_params = [], []
        for name, param in self.model.named_parameters():
            if len(param.shape) == 1 or name.endswith(".bias") or "extra_tokens" in name or 'embed_tokens' in name:
                non_wd_params.append(param)
            else:
                wd_params.append(param)
        return wd_params, non_wd_params
    
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith('teacher.'):
                del sd[k]
        return sd
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

@dataclass
class Sx3HReConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    embed_dim: int = 768
    cmli_dim: int = 256

    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    norm_affine: bool = True
    layer_init_scale: float = 0.2

class Sx3HRe(nn.Module):
    def __init__(self,
                 cfg: Sx3HReConfig,
                 ):
        super(Sx3HRe, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine)

        self.cmli_down_proj = nn.Linear(self.cfg.embed_dim, self.cfg.cmli_dim)

        self.norm = make_layer_norm(self.cfg.embed_dim)

        self.shared = Block(
            dim=self.cfg.embed_dim,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=True,
            init_values=self.cfg.layer_init_scale,
            norm_layer=make_layer_norm,
        )

        self.apply(init_bert_params)

        self.text_model = load_pretrained_d2v_model(state_dict_path=os.path.join(self.cfg.pretrained_path, self.cfg.pretrained.text))
        self.text_model.blocks = self.text_model.blocks[:self.cfg.depth]
        self.image_model = load_pretrained_d2v_model(state_dict_path=os.path.join(self.cfg.pretrained_path, self.cfg.pretrained.image))
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
        encoder_out = self.image_model.extract_features(
            source=image,
            remove_extra_tokens=False,
        )
        
        return self.encode_shared(encoder_out)

    def encode_text(self, text, padding_mask):
        encoder_out = self.text_model.extract_features(
            source=text,
            padding_mask=padding_mask,
            remove_extra_tokens=False,
        )
        out_dict = self.encode_shared(encoder_out)
        out_dict['padding_mask'] = padding_mask
        return out_dict
    
    def encode_shared(self, encoder_out):
        out_dict = dict()
        x = encoder_out['x']
        mask = encoder_out['padding_mask'] if 'padding_mask' in encoder_out else None
        x_interm, x = self.shared(x=x, mask=mask)
        out_dict['x_raw'] = self.cmli_down_proj(x)
        x_interm = x_interm[:, 0]
        x = x[:, 0]

        out_dict["encoder_out"] = self.norm(x)
        x = x / x.norm(dim=-1, keepdim=True)
        out_dict["x"] = x
        x_interm = x_interm / x_interm.norm(dim=-1, keepdim=True)
        out_dict["x_interm"] = x_interm
        return out_dict

MODEL_REGISTRY['Sx3HRe'] = {
    'cfg': Sx3HReConfig,
    'module': Sx3HRePreTrainingLightningModule
}