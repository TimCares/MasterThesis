import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, List
import numpy as np
import os
from utils import load_pretrained_d2v_model
from omegaconf import OmegaConf
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
from modules import MOMEAltBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2 import modeling_pretrain

logger = logging.getLogger(__name__)


class AMMData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.itc = self.cfg.model.itc

        self.model = AMMData2Vec(cfg=self.cfg.model)
        
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

        output_dict = self(batch) # call "forward"

        kd_loss = F.mse_loss(output_dict['encoder_out_text'][:, 0], output_dict['encoder_out_image'][:, 0], reduction="mean")
        
        if self.itc:
            kd_loss = kd_loss / 2
            itc_loss = self.itc_loss(text_features=output_dict['x_text'], image_features=output_dict['x_image'])
            self.log(f"{stage}/itc_loss", itc_loss)
        else:
            itc_loss = torch.tensor(0.0).to(kd_loss.device)
        
        loss = kd_loss + itc_loss

        self.log(f"{stage}/kd_loss", kd_loss)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def itc_loss(self, text_features:torch.Tensor, image_features:torch.Tensor, stage:str='train') -> torch.Tensor:
        scale = self.model.logit_scale.exp()
        
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        self._log_similarity(logits_per_image, logits_per_text, stage)

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
    
    def _log_similarity(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, stage:str='train') -> None:
        diagonal_mask = torch.eye(logits_per_image.size(0)).bool()
        for logits, name in [(logits_per_image, "i2t"), (logits_per_text, "t2i")]:
            mean_pos_sim = logits[diagonal_mask].mean()
            mean_neg_sim = logits[~diagonal_mask].mean()
            self.log(f"{stage}/{name}_mean_pos_similarity", mean_pos_sim)
            self.log(f"{stage}/{name}_mean_neg_similarity", mean_neg_sim)

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
class BEiTArgs():
    pretrained_path: str
    drop_path_rate:float = 0.1
    use_shared_rel_pos_bias:bool =  True
    use_abs_pos_emb:bool = False
    vocab_size:int = 8192
    init_values:float = 0.1

@dataclass
class AMMData2VecConfig():
    beit2_args:BEiTArgs = field(default_factory=BEiTArgs)
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

class AMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: AMMData2VecConfig,
                 ):
        super(AMMData2Vec, self).__init__()
        self.cfg = cfg
        self.teacher = self.load_beit2_teacher()
        self._freeze(self.teacher)

        if self.cfg.itc:
            make_proj = partial(nn.Linear, self.cfg.embed_dim, self.cfg.embed_dim, bias=False)
            self.text_proj = make_proj()
            self.image_proj = make_proj()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
        state_dict_path = os.path.join(self.cfg.pretrained_path, 'nlp_base.pt')
        d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)

        for i in range(self.cfg.depth):
            self.blocks[i].init_from_pretrained(
                pretained_block=d2v_model.blocks[i],
                modality=Modality.TEXT,
                init_attention=True,
            )
        
        self.text_feature_extractor = d2v_model.modality_encoders['text']
        self._freeze(self.text_feature_extractor)

    def load_beit2_teacher(self):
        sd = torch.load(self.cfg.beit2_args.pretrained_path)['model']
        for key in list(sd.keys()):
            if "cls_pt_layers" in key:
                del sd[key]
        kwargs = OmegaConf.to_container(self.cfg.beit2_args, resolve=True)
        kwargs.pop("pretrained_path")

        beit2 = VisionTransformerForMaskedImageModeling(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
        )

        result = beit2.load_state_dict(sd)
        logger.info(f"Loaded BEiT2 teacher state dict with result: {result}")
        del beit2.lm_head
        return beit2

    def forward(
        self,
        image:torch.Tensor,
        text:torch.Tensor,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None,
    ):
        img_out = self.encode_image(image)
        text_out = self.encode_text(text, padding_mask)
        out = dict()
        out.update({k+'_image': v for k, v in img_out.items()})
        out.update({k+'_text': v for k, v in text_out.items()})
        return out
    
    def encode_image(self, image):
        # no mask
        bool_masked_pos = torch.zeros((image.shape[0], self.teacher.patch_embed.num_patches), dtype=torch.bool).to(image.device)

        with torch.no_grad():
            x = self.teacher.forward_features(
                x=image,
                bool_masked_pos=bool_masked_pos,
            )
        encoder_out = x
        out = {
            "encoder_out": encoder_out,
        }
        if self.cfg.itc:
            x = self.image_proj(x[:, 0])
            x = x / x.norm(dim=-1, keepdim=True)
            out["x"] = x
        return out

    def encode_text(self, text, padding_mask):
        extractor_out = self.text_feature_extractor(
            features=text,
            padding_mask=padding_mask,
            mask=False,
            remove_masked=False,
        )

        x = extractor_out["x"]
        padding_mask = extractor_out["padding_mask"]

        if self.dropout_input is not None:
            x = self.dropout_input(x)
        
        for i in range(self.cfg.depth):
            x, _ = self.blocks[i](
                x,
                modality=Modality.TEXT,
                padding_mask=padding_mask,
            )
        encoder_out = x
        out = {
            "encoder_out": encoder_out,
        }
        if self.cfg.itc:
            x = self.text_proj(x[:, 0])
            x = x / x.norm(dim=-1, keepdim=True)
            out["x"] = x

        return out
    
    def freeze_attention_blocks(self):
        for block in self.blocks:
            self._freeze(block.attn)

    def unfreeze_attention_blocks(self):
        for block in self.blocks:
            self._unfreeze(block.attn)

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False
        module.eval()

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True
        module.train()
