import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from collections import namedtuple
from typing import Dict, Any, Optional, List, Union
import numpy as np
import os
from utils import load_pretrained_d2v_model
import logging
import lightning as L
import math
from omegaconf import II
from dataclasses import dataclass, field
from datasets import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from kd_precompute import instance_norm_and_average

from data2vec_fairseq.models.modalities.modules import AltBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params

logger = logging.getLogger(__name__)

MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])

class KDData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        assert self.cfg.average_top_k_layers_student <= self.cfg.model.depth,\
            f"Can't aggregate more layers {self.cfg.average_top_k_layers_student} than available {self.cfg.model.depth}"
        if not cfg.dry_run:
            self.model = KDMMData2Vec(cfg=cfg.model)
        else:
            self.model = None

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        target_dict = batch.pop('target')
        output_dict = self(batch) # call "forward"

        y_hat = instance_norm_and_average(output_dict['layer_results'])

        y_hat = y_hat[output_dict['padding_mask']]
        target = target[target_dict['padding_mask']]
        
        assert y_hat.shape == target.shape() # for simple pretraining (only variant 1!) this must be the case
        return self.kd_loss(y_hat=y_hat, y=target)
    
    def kd_loss(self, y_hat, y):
        y_hat = y_hat.view(-1, y_hat.size(-1)).float()
        y = y.view(-1, y_hat.size(-1))

        loss = F.mse_loss(y_hat, y, reduction="none")

        if self.cfg.loss_scale is not None:
            scale = self.cfg.loss_scale
        else:
            scale = 1 / math.sqrt(y_hat.size(-1))
        reg_loss = loss * scale

        return reg_loss


    def configure_optimizers(self, cfg):
        optimizer = torch.optim.AdamW(self.model.parameters(), **cfg.params)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.schedule.warmup_steps,
            num_training_steps=cfg.schedule.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "cosine_w_warmup"}]


@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'
@dataclass
class KDMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = PretrainedStateDictsConfig()

    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    encoders_embed_dim: int = II("embed_dim") # only as a first test
    embed_dim: int = 768

    depth: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4
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

    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    modality_encoder_proj: bool = False

    seed: int = 42

class KDMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: KDMMData2VecConfig,
                 ):
        super().__init__()
        self.cfg = cfg
        self.modality_encoders:nn.ModuleDict[Modality, nn.Module] = self._get_modality_encoders()
        self.modality_encoders.eval()
        self.modality_encoders.is_pretrained = True

        self.projections = nn.ModuleDict({
            mode.name: 
            (nn.Linear(self.cfg.encoders_embed_dim, self.cfg.embed_dim) 
             if self.cfg.modality_encoder_proj 
             else nn.Identity())
             for mode in [Modality.AUDIO, Modality.IMAGE, Modality.TEXT]
        })

        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                dim=self.cfg.embed_dim if dim is None else dim,
                num_heads=self.cfg.num_heads if heads is None else heads,
                mlp_ratio=self.cfg.mlp_ratio,
                qkv_bias=True,
                drop=self.cfg.encoder_dropout,
                attn_drop=self.cfg.attention_dropout,
                mlp_drop=self.cfg.activation_dropout,
                post_mlp_drop=self.cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=self.cfg.layer_norm_first,
                ffn_targets=not self.cfg.end_of_block_targets,
            )
        
        self.alibi_biases = {}

        self.loss_scale = self.cfg.loss_scale

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(self.cfg.depth)])

        self.norm = None
        if self.cfg.layer_norm_first:
            self.norm = make_layer_norm(self.cfg.embed_dim)

        self.layerdrop = self.cfg.layerdrop
        self.mask_seed = self.cfg.mask_seed

        self.apply(self._init_except_pretrained)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
        

    def _get_modality_encoders(self) -> None:
        modality_encoders = {}
        for mode, state_dict_name in self.cfg.pretrained.items():
            mode_enum:Modality = Modality[mode.upper()] # Modality[mode.upper()]: e.g. 'text' => Modality.Text
            logger.info(f'Loading modality encoder for: {mode_enum}')
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)
            mode_feature_extractor = d2v_model.modality_encoders[mode_enum.name]
            modality_encoders[mode_enum.name] = mode_feature_extractor

        return nn.ModuleDict(modality_encoders)

    def _init_except_pretrained(self, module:nn.Module):
        if hasattr(module, "is_pretrained") and module.is_pretrained:
            return
        else:
            module.apply(init_bert_params)

    def forward(
        self,
        modes:List[Modality],
        audio:torch.Tensor=None,
        image:torch.Tensor=None,
        text:torch.Tensor=None,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None, # Union[torch.Tensor, Dict[str, torch.Tensor]] for multi input later
        mask:bool=True,
        features_only:bool=False,
        force_remove_masked:bool=False,
        remove_extra_tokens:bool=True,
        precomputed_mask=None,
    ):
        assert len(modes)==1, f"This model accepts exactly modality indicator at a time, received: {modes}"
        n_sources = sum(mode is not None for mode in [audio, image, text])
        assert n_sources==1,\
            f"This model accepts exactly one modality data source at a time, got {n_sources}."
        mode = modes[0].name # is now a string, ModuleDict does not support enums as keys
        source = audio or image or text

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.mask_seed, update=self.num_updates, ids=id)
        
        feature_extractor = self.modality_encoders[mode]
        with torch.no_grad():
            extractor_out = feature_extractor(
                source,
                padding_mask,
                mask,
                remove_masked=not features_only or force_remove_masked,
                clone_batch=1,
                mask_seeds=mask_seeds,
                precomputed_mask=precomputed_mask,
            )

        extractor_out = self.projections[mode](extractor_out)

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if remove_extra_tokens:
            x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
            if masked_padding_mask is not None:
                masked_padding_mask = masked_padding_mask[
                    :, feature_extractor.modality_cfg.num_extra_tokens :
                ]

        return {
            "x": x,
            "padding_mask": masked_padding_mask,
            "layer_results": layer_results,
            "mask": encoder_mask,
        }
    
    def extract_features(
        self, audio=None, image=None, text=None, modes:List[Modality]=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            audio=audio,
            image=image,
            text=text,
            modes=modes,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res
    
    def encode_modality(self, mode:Union[Modality, List[Modality]], source:torch.Tensor, padding_mask=None, normalize:bool=True):
        if isinstance(mode, List):
            assert len(mode)==1, 'Only one modality allowed when calling "encode_modality".'
            mode = mode[0]
        # at this point Modality has to be of type "Modality".
        audio, image, text = None
        if mode == Modality.AUDIO:
            audio = source
        elif mode == Modality.IMAGE:
            image = source
        elif mode == Modality.TEXT:
            text = source
        else:
            raise ValueError(f"Did not find mand for modality {mode}, allowed modes are: [{Modality.AUDIO}, {Modality.IMAGE}, {Modality.TEXT}]")

        output = self.extract_features(
            audio=audio,
            image=image,
            text=text,
            modes=[mode],
            padding_mask=padding_mask,
            remove_extra_tokens=False,
        )['x']
        output = self.projections[mode](output[:, 0, :])

        if normalize:
            output = F.normalize(output, dim=-1)
        
        return output # shape: (batch_size, 1, embed_dim)

    def encode_audio(self, audio, padding_mask, normalize:bool=True):
        return self.encode_modality(source=audio,
                                     mode=Modality.AUDIO,
                                     padding_mask=padding_mask,
                                     normalize=normalize)
    
    def encode_image(self, image, normalize:bool=True):
        return self.encode_modality(source=image,
                                     mode=Modality.IMAGE,
                                     normalize=normalize)

    def encode_text(self, text, padding_mask, normalize:bool=True):
        return self.encode_modality(source=text,
                                     mode=Modality.TEXT,
                                     padding_mask=padding_mask,
                                     normalize=normalize)


class InitializedD2V(KDMMData2Vec):
    def __init__(self,
                 cfg: KDMMData2VecConfig):
        super().__init__(cfg)


    def _get_modality_encoders(self) -> None:
        modality_encoders = {}
        for mode, state_dict_name in self.cfg.pretrained.items():
            mode_enum:Modality = Modality[mode.upper()] # Modality[mode.upper()]: e.g. 'text' => Modality.Text
            logger.info(f'Loading modality encoder for: {mode_enum}')
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)
            mode_feature_extractor = d2v_model.modality_encoders[mode_enum.name]
            modality_encoders[mode_enum.name] = mode_feature_extractor

        return nn.ModuleDict(modality_encoders)
    
    def _init_blocks() -> None:
        pass