import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any
from collections import namedtuple
from omegaconf import OmegaConf
import os
import logging
import pytorch_lightning as L
import json
from timm.models.vision_transformer import LayerScale
from dataclasses import dataclass, field
from transformers.optimization import get_cosine_schedule_with_warmup
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from transformers import BertModel
from . import MODEL_REGISTRY
from modules import Block, ClipLoss
from modules.layers import Pooler, MoMEBlock
from utils import freeze_module, load_beit2_teacher, MaskingGenerator
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2.run_beitv2_pretraining import get_visual_tokenizer

logger = logging.getLogger(__name__)

class SSMKECPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = SSMKEC(cfg=self.cfg.model)

        self.lm_head = nn.Linear(self.model.cfg.embed_dim, 30522)
        self.lm_head.apply(init_bert_params)
        self.im_head = nn.Linear(self.model.cfg.embed_dim, 8192)
        self.im_head.apply(init_bert_params)

        self.masked_position_generator = MaskingGenerator(
            input_size=(14, 14),
            num_masking_patches=75,
            max_num_patches=None,
            min_num_patches=16,
        )
        VQKDArgs = namedtuple('VQKDArgs', ['tokenizer_model', 'tokenizer_weight', 'codebook_size', 'codebook_dim'])
        vqkd_args = VQKDArgs(
            **OmegaConf.to_container(self.cfg.vqkd, resolve=True)
        )
        self.vqkd = get_visual_tokenizer(vqkd_args)
        del self.vqkd.decoder
        del self.vqkd.decoder_task_layer
        freeze_module(self.vqkd)

        self.save_hyperparameters()

    def on_train_start(self):
        logger.info(f'World size: {self.trainer.world_size}')
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
        mlm_output = self.mlm_forward(batch)
        mim_output = self.mim_forward(batch)

        for k, v in mlm_output.items():
            self.log(f"{stage}/{k}", v, prog_bar=True)
        
        loss = mlm_output['mlm_loss'] + mim_output['mim_loss']

        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def mlm_forward(self, batch):
        x = self.model.encode_shared(
            image=batch['image'],
            text=batch['masked_text'],
            padding_mask=batch['padding_mask'],
        )[:, :batch['masked_text'].size(1)] # 64 is the max text length
        x = self.lm_head(x)

        input = x.view(-1, 30522)
        target = batch['targets'].view(-1)
        
        mlm_loss = F.cross_entropy(
            input=input,
            target=target,
            ignore_index=-100,
        )

        input_ = input.argmax(dim=-1)[target != -100]
        target_ = target[target != -100]
        mlm_acc = (input_ == target_).float().mean()
        
        output_dict = {
            "mlm_loss": mlm_loss,
            "mlm_acc": mlm_acc,
        }
        return output_dict
    
    def mim_forward(self, batch):
        bool_masked_pos = self.masked_position_generator().flatten(1).to(torch.bool)
        x = self.model.encode_shared(
            image=batch['image'],
            text=batch['text'],
            padding_mask=batch['padding_mask'],
            bool_masked_pos=bool_masked_pos,
        )[:, batch['text'].size(1):] # 64 is the max text length

        input = self.im_head(x[bool_masked_pos]).view(-1, 8192)

        with torch.no_grad():
            input_ids = self.vqkd.get_codebook_indices(batch['image'])
            target = input_ids[bool_masked_pos].view(-1)
        
        mim_loss = F.cross_entropy(
            input=input,
            target=target,
        )

        mim_acc = (input.argmax(dim=-1) == target).float().mean()
        
        output_dict = {
            "mim_loss": mim_loss,
            "mim_acc": mim_acc,
        }
        return output_dict
    
    def log_acc(self, logits_per_image, logits_per_text, target, stage, key_prefix=""):
        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()
        if key_prefix != "":
            key_prefix = key_prefix + "_"
        self.log(f"{stage}/{key_prefix}text_acc", text_itc_acc)
        self.log(f"{stage}/{key_prefix}image_acc", img_itc_acc)
        self.log(f"{stage}/{key_prefix}acc", (img_itc_acc + text_itc_acc) / 2, prog_bar=True)

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
        param_groups = self._get_param_groups(lr=learning_rate)
        optim_args = {
            "params": param_groups,
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
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or name.endswith(".bias") or "fixed_positional_encoder.positions" in name or "extra_tokens" in name or 'embeddings' in name:
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
    
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith('teacher.'):
                del sd[k]
        return sd
        
    def log(self, *args, **kwargs):
        super().log(batch_size=self.cfg.data.dataloader.batch_size, sync_dist=True, *args, **kwargs)

@dataclass
class BEiTv2Config():
    model_path:str = "/workspace/models"
    model_name:str = "beitv2_base_patch16_224_pt1k.pth"
    drop_path_rate: float = 0.05
    use_shared_rel_pos_bias:bool = True
    use_abs_pos_emb: bool = False
    vocab_size: int = 8192
    init_values: float =  0.1

@dataclass
class SSMKECConfig():
    beitv2: BEiTv2Config = field(default_factory=BEiTv2Config)

    embed_dim: int = 768
    depth: int = 6
    shared_depth: int = 2

class SSMKEC(nn.Module):
    def __init__(self,
                 cfg: SSMKECConfig,
                 ):
        super(SSMKEC, self).__init__()
        self.cfg = cfg
        make_layer_norm = partial(nn.LayerNorm, eps=1e-6)

        self.token_type_embeddings = nn.Embedding(2, self.cfg.embed_dim)
        self.tte_scale = LayerScale(self.cfg.embed_dim, init_values=1e-5)

        self.shared_blocks = nn.ModuleList([
            MoMEBlock(
                dim=self.cfg.embed_dim,
                num_heads=12,
                mlp_ratio=4.0,
                qkv_bias=True,
                layer_scale_init_values=0.2,
                norm_layer=make_layer_norm,
                max_text_len=64,
            ) for _ in range(self.cfg.shared_depth)
        ])

        self.apply(init_bert_params)

        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_model.encoder.layer = self.text_model.encoder.layer[:self.cfg.depth]
        self.text_model.pooler = None # remove pooler
        freeze_module(self.text_model)
        
        beit2_kwargs = OmegaConf.to_container(self.cfg.beitv2, resolve=True)
        sd_path = beit2_kwargs.pop("model_path")
        sd_name = beit2_kwargs.pop("model_name")
        beit_path = os.path.join(sd_path, sd_name)

        self.image_model:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        self.image_model.blocks = self.image_model.blocks[:self.cfg.depth]
        freeze_module(self.image_model)
    
    def encode_image_only(self, image, bool_masked_pos=None):
        x = self.image_model.forward_features(
            x=image,
            bool_masked_pos=bool_masked_pos,
        )

        img_tte = self.token_type_embeddings(
            torch.ones_like(x[:, :, 0], dtype=torch.long)
        )
        img_tte = self.tte_scale(img_tte)
        x_tte = x + img_tte

        return {
            'x': x,
            'x_tte': x_tte,
        }

    def encode_text_only(self, text, padding_mask):
        x = self.text_model(
            input_ids=text,
            attention_mask=1-padding_mask,
        ).last_hidden_state

        text_tte = self.token_type_embeddings(
            torch.zeros_like(padding_mask)
        )
        text_tte = self.tte_scale(text_tte)
        x_tte = x + text_tte

        return {
            'x': x,
            'x_tte': x_tte,
        }
    
    def encode_shared(self, image=None, text=None, padding_mask=None, bool_masked_pos=None):
        modality_type = None
        if text is None:
            x = self.encode_image_only(image, bool_masked_pos=bool_masked_pos)['x_tte']
            modality_type = "image"
        if image is None:
            x = self.encode_text_only(text, padding_mask)['x_tte']
            modality_type = "text"

        if modality_type is None:
            x_image = self.encode_image_only(image, bool_masked_pos=bool_masked_pos)['x_tte']
            x_text = self.encode_text_only(text, padding_mask)['x_tte']
            modality_type = 'vl'

            x = torch.cat([x_text, x_image], dim=1)
            image_mask = torch.zeros(image.size(0), image.size(1)).to(padding_mask.device)
            padding_mask = torch.cat([padding_mask, image_mask], dim=1)

        for block in self.shared_blocks:
            x = block(x=x, mask=padding_mask, modality_type=modality_type)
        
        return x
        

MODEL_REGISTRY['SSMKEC'] = {
    'cfg': SSMKECConfig,
    'module': SSMKECPreTrainingLightningModule
}
