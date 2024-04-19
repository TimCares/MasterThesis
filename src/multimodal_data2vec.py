import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, Optional, List, Union
import numpy as np
import os
from utils import load_pretrained_d2v_model
import logging
import pytorch_lightning as L
import math
from omegaconf import II
from dataclasses import dataclass, field
from datasets_ import Modality
from transformers.optimization import get_cosine_schedule_with_warmup
from kd_precompute import special_token_and_average, average_twice

from data2vec_fairseq.models.modalities.modules import AltBlock
from data2vec_fairseq.models.modalities.base import MaskSeed
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from mome_alt_attention import MOMEAltBlock

logger = logging.getLogger(__name__)

class KDData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.model = KDMMData2Vec(cfg=self.cfg.model)

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        target = batch.pop('target')
        output_dict = self(batch) # call "forward"

        if self.cfg.average_twice:
            y_hat = average_twice(output_dict['layer_results'], output_dict['padding_mask'])
        else:
            y_hat = special_token_and_average(output_dict['layer_results'])
        
        assert y_hat.shape == target.shape # for simple pretraining this must be the case
        return self.kd_loss(y_hat=y_hat, y=target)
    
    def kd_loss(self, y_hat, y):
        y_hat = y_hat.view(-1, y_hat.size(-1)).float()
        y = y.view(-1, y_hat.size(-1)).float()

        loss = F.mse_loss(y_hat, y, reduction="none").float()

        if self.cfg.model.loss_scale is not None:
            scale = self.cfg.model.loss_scale
        else:
            scale = 1 / math.sqrt(y_hat.size(-1))
        reg_loss = loss * scale
        
        return reg_loss.sum()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.optimizer)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
            num_training_steps=self.cfg.optimizer_schedule.max_steps,
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
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    init_block_from: Optional[str] = None
    init_strategy: Optional[str] = None # 'ffill' or 'leave_one_out'
    freeze_attention: bool = False

    init_attention_from: Optional[str] = None

    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    encoders_embed_dim: int = II("embed_dim")
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

    end_of_block_targets: bool = False

    modality_encoder_proj: bool = False

    seed: int = 42

class KDMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: KDMMData2VecConfig,
                 ):
        super(KDMMData2Vec, self).__init__()
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
        
        # self.alibi_biases = {}

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        self.blocks = nn.ModuleList([self.make_block(dpr[i]) for i in range(self.cfg.depth)])

        self.norm = None
        if self.cfg.layer_norm_first:
            self.norm = make_layer_norm(self.cfg.embed_dim)

        self.layerdrop = self.cfg.layerdrop
        self.mask_seed = self.cfg.seed

        self.apply(self._init_except_pretrained)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
        
    def make_block(self, drop_path, dim=None, heads=None):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

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

    def _get_modality_encoders(self) -> None:
        modality_encoders = {}
        for mode in self.cfg.pretrained.keys():
            state_dict_name = self.cfg.pretrained[mode]
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
        
        if audio is not None:
            source = audio
        elif image is not None:
            source = image
        elif text is not None:
            source = text
        else:
            raise ValueError("Audio, image or text must be provided, found all to be None.")

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
        audio = None
        image = None
        text = None
        if mode == Modality.AUDIO:
            audio = source
        elif mode == Modality.IMAGE:
            image = source
        elif mode == Modality.TEXT:
            text = source
        else:
            raise ValueError(f"Did not find mode for modality \"{mode}\", allowed modes are: [{Modality.AUDIO}, {Modality.IMAGE}, {Modality.TEXT}]")

        output = self.extract_features(
            audio=audio,
            image=image,
            text=text,
            modes=[mode],
            padding_mask=padding_mask,
            remove_extra_tokens=False,
        )['x']
        output = self.projections[mode.name](output[:, 0, :])

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

    def _get_pretrained_block_indices(self, depth, n_blocks_pretrained) -> List[int]:
        blocks_pretrained = [i for i in range(n_blocks_pretrained)]
        blocks = []
        pretrained_blocks_count = len(blocks_pretrained)

        if depth*2 > pretrained_blocks_count:
            pretrained_remaining = pretrained_blocks_count
            model_remaining = depth
            current_block_idx = 0
            while model_remaining*2 > pretrained_remaining and model_remaining > 0:
                blocks.append(blocks_pretrained[current_block_idx])
                pretrained_remaining -= 1
                model_remaining -= 1
                current_block_idx += 1
            
            if model_remaining > 0:
                for i in range(0, depth-current_block_idx):
                    take_block_idx = i*2+current_block_idx
                    blocks.append(blocks_pretrained[take_block_idx])
        else:
            for i in range(depth):
                blocks.append(blocks_pretrained[i*2])
        
        assert len(blocks) == depth

        return blocks
    
    def freeze_attention_blocks(self):
        if self.cfg.freeze_attention:
            for block in self.blocks:
                for param in block.attn.parameters():
                    param.requires_grad = False
                block.attn.eval()

    def unfreeze_attention_blocks(self):
        if self.cfg.freeze_attention:
            for block in self.blocks:
                for param in block.attn.parameters():
                    param.requires_grad = True
                block.attn.train()


class InitializedD2V(KDMMData2Vec):
    def __init__(self,
                 cfg: KDMMData2VecConfig):
        super().__init__(cfg)
        self._init_blocks()
        if self.cfg.freeze_attention:
            logger.info("Freezing attention weights.")
            self.freeze_attention_blocks()
    
    def _init_blocks(self) -> None:
        logger.info(f"Initializing transformer blocks from: {self.cfg.init_block_from}")
        assert self.cfg.init_block_from in self.cfg.pretrained.keys(), f"Could not find pretrained state dict for: {self.cfg.init_block_from} "
        "(Used to initialize the transformer blocks)"

        state_dict_name = self.cfg.pretrained[self.cfg.init_block_from]
        state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
        d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)

        if self.cfg.init_strategy == 'ffill':
            indices = [i for i in range(self.cfg.depth)]
        elif self.cfg.init_strategy == 'leave_one_out':
            indices = self._get_pretrained_block_indices(depth=self.cfg.depth, n_blocks_pretrained=len(d2v_model.blocks))
        else:
            raise ValueError(f"Unknown initialization strategy: {self.cfg.init_strategy}")
        
        self.blocks = nn.ModuleList([d2v_model.blocks[i] for i in indices])
        self.blocks.is_pretrained = True

class MoMEInitializedD2V(KDMMData2Vec):
    def __init__(self,
                 cfg: KDMMData2VecConfig):
        super().__init__(cfg)
        self._init_blocks()
        if self.cfg.freeze_attention:
            logger.info("Freezing attention weights.")
            self.freeze_attention_blocks()

    def _init_blocks(self) -> None:
        logger.info(f"Initializing Mlps in transformer blocks from all modalities using MoME.")
        assert self.cfg.init_block_from in self.cfg.pretrained.keys(), f"Could not find pretrained state dict for: {self.cfg.init_block_from} "
        "(Used to initialize the transformer blocks)"

        state_dict_name = self.cfg.pretrained[self.cfg.init_block_from]
        state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
        d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)

        if self.cfg.init_strategy == 'ffill':
            indices = [i for i in range(self.cfg.depth)]
        elif self.cfg.init_strategy == 'leave_one_out':
            indices = self._get_pretrained_block_indices(depth=self.cfg.depth, n_blocks_pretrained=len(d2v_model.blocks))
        else:
            raise ValueError(f"Unknown initialization strategy: {self.cfg.init_strategy}")
        
        for mode in [Modality.AUDIO, Modality.IMAGE, Modality.TEXT]:
            if self.cfg.init_attention_from is not None and self.cfg.init_attention_from == mode.name.lower():
                logger.info(f"Initializing Attation in transformer blocks from modality: {mode.name}")
                for i in range(self.cfg.depth):
                    self.blocks[i].init_from_pretrained(pretained_block=d2v_model.blocks[indices[i]], # blocks defined in KDMMData2Vec constructor
                                                        mode=mode.name.lower(),
                                                        init_attention=True)
            else:
                for i in range(self.cfg.depth):
                    self.blocks[i].init_from_pretrained(pretained_block=d2v_model.blocks[indices[i]], # blocks defined in KDMMData2Vec constructor
                                                        mode=mode.name.lower(),
                                                        init_attention=False)
        
        self.blocks.is_pretrained = True


    def make_block(self, drop_path, dim=None, heads=None):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return MOMEAltBlock(
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


class DummyModel(nn.Module):
    def __init__(self,
                 cfg: KDMMData2VecConfig,
                 ):
        super(DummyModel, self).__init__()
        self.cfg = cfg
        self.embed_dim = 20
        # so model.parameters() is not empty
        self.lin = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self,
                modes:List[Modality],
                audio:torch.Tensor=None,
                image:torch.Tensor=None,
                text:torch.Tensor=None,
                id:torch.Tensor=None,
                padding_mask:torch.Tensor=None,
                mask:bool=True,
                features_only:bool=False,
                force_remove_masked:bool=False,
                remove_extra_tokens:bool=True,
                precomputed_mask=None,
        ):

        if audio is not None:
            source = audio
        elif image is not None:
            source = image
        elif text is not None:
            source = text
        else:
            raise ValueError("Audio, image or text must be provided, found all to be None.")
        
        out = self.lin(torch.randn_like(source))
        return {'layer_results': [out for _ in range(self.cfg.depth)],}
    
    def encode_modality(self, mode:Union[Modality, List[Modality]], source:torch.Tensor, padding_mask=None, normalize:bool=True):
        if isinstance(mode, List):
            assert len(mode)==1, 'Only one modality allowed when calling "encode_modality".'
        
        return torch.rand((source.shape[0], 1, self.embed_dim))
    
    def encode_text(self, text, padding_mask, normalize:bool=True):
        return torch.rand((text.shape[0], 1, self.embed_dim))
    
    def encode_image(self, image, normalize:bool=True):
        return torch.rand((image.shape[0], 1, self.embed_dim))
    
    def encode_audio(self, audio, padding_mask, normalize:bool=True):
        return torch.rand((audio.shape[0], 1, self.embed_dim))

class TestLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = DummyModel(cfg=self.cfg.model)

    def forward(self, input_dict):
        return self.model(**input_dict)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        target = batch.pop('target')
        output_dict = self(batch) # call "forward"

        if self.cfg.average_twice:
            y_hat = average_twice(output_dict['layer_results'], output_dict['padding_mask'])
        else:
            y_hat = special_token_and_average(output_dict['layer_results'])
        
        assert y_hat.shape == target.shape # for simple pretraining this must be the case
        return self.kd_loss(y_hat=y_hat, y=target)
    
    def kd_loss(self, y_hat, y):
        y_hat = y_hat.view(-1, y_hat.size(-1)).float()
        y = y.view(-1, y_hat.size(-1)).float()

        loss = F.mse_loss(y_hat, y, reduction="none").float()

        if self.cfg.model.loss_scale is not None:
            scale = self.cfg.model.loss_scale
        else:
            scale = 1 / math.sqrt(y_hat.size(-1))
        reg_loss = loss * scale
        
        return reg_loss.sum()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), **self.cfg.optimizer)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
            num_training_steps=self.cfg.optimizer_schedule.max_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "cosine_w_warmup"}]