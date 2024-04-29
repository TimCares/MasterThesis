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
from omegaconf import II
from dataclasses import dataclass, field
from enum import Enum, auto
from data2vec_fairseq.data.modality import Modality
from transformers.optimization import get_cosine_schedule_with_warmup

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from modules import LayerResultBlock

logger = logging.getLogger(__name__)

def cosine_similarity_loss(input, target):
    input = F.normalize(input, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1 - F.cosine_similarity(input, target, dim=-1)).mean()

def prepare_output(out:List[torch.Tensor]) -> List[torch.Tensor]:
    out = [
        F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
        for tl in out  # BTC -> BCT -> BTC
    ]

    y = out[0].float()
    for tl in out[1:]:
        y.add_(tl.float())
    y = y.div_(len(out))
    return y

class KDSharedData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        state_dict_name = self.cfg.model.pretrained['image']
        state_dict_path = os.path.join(self.cfg.model.pretrained_path, state_dict_name)
        self.teacher = load_pretrained_d2v_model(state_dict_path=state_dict_path)
        del self.teacher.modality_encoders
        
        self.model = KDSharedMMData2Vec(cfg=self.cfg.model)

        if self.cfg.model.loss_fn == 'cosine_similarity':
            self.loss_fn = cosine_similarity_loss
        else:
            self.loss_fn = partial(F.mse_loss, reduction='mean')
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict, return_encoder_output=True)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        if 'target' in batch:
            batch.pop('target')
        output_dict = self(batch) # call "forward"

        with torch.no_grad():
            target = self.teacher.extract_features(
                    source=batch['image'],
                    mode=None, # determined automatically in model
                    padding_mask=None,
                    mask=False, # we are creating targets from a teacher model for the student model, so no mask
                    remove_extra_tokens=False,
                    precomputed_encoder_output=output_dict['encoder_output'],
                )
        
        target = target['layer_results']
        target = prepare_output(target[-self.cfg.model.depth:])
        pred = output_dict['layer_results']
        pred = prepare_output(pred)

        pred = pred.view(-1, pred.size(-1)).float() # BLT -> (B*L)T
        target = target.view(-1, target.size(-1)).float() # BLT -> (B*L)T
        assert pred.shape == target.shape # this must be the case

        loss = self.loss_fn(input=pred, target=target).float()
        self.log("train/loss", loss, prog_bar=True)
        if batch['modes'][0] == Modality.IMAGE:
            self.log("train/loss_img", loss, prog_bar=True)
        elif batch['modes'][0] == Modality.AUDIO:
            self.log("train/loss_audio", loss, prog_bar=True)
        else:
            self.log("train/loss_text", loss, prog_bar=True)
        return loss
                
    
    def kd_loss(self, y_hat, y):
        y_hat = y_hat.view(-1, y_hat.size(-1)).float() # (B, D, C) -> (B*D, C)
        y = y.contiguous()
        y = y.view(-1, y.size(-1)).float() # (B, D, C) -> (B*D, C)

        return F.mse_loss(y_hat, y, reduction="mean").float()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        if self.cfg.optimizer.warmup:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.optimizer_schedule.warmup_steps,
                num_training_steps=self.cfg.optimizer_schedule.max_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "cosine_w_warmup"}]
        else:
            return optimizer


@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

class PredictionAggregationMode(Enum):
    CLS_TOKEN = auto()
    MEAN_POOLING = auto()
@dataclass
class KDMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)
    prediction_mode: PredictionAggregationMode = PredictionAggregationMode.MEAN_POOLING

    supported_modalities: List[Modality] = field(default_factory=lambda: [Modality.AUDIO, Modality.IMAGE, Modality.TEXT])

    cls_loss_weight: Optional[float] = None
    loss_fn: str = 'cosine_similarity' # 'cosine_similarity' or 'mse'

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

class KDSharedMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: KDMMData2VecConfig,
                 ):
        super(KDSharedMMData2Vec, self).__init__()
        self.cfg = cfg
        self.supported_modalities = cfg.supported_modalities

        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        self.blocks = nn.ModuleList([self.make_block(dpr[i]) for i in range(self.cfg.depth)])

        self.norm = None
        if self.cfg.layer_norm_first:
            self.norm = make_layer_norm(self.cfg.embed_dim)

        self.layerdrop = self.cfg.layerdrop
        self.mask_seed = self.cfg.seed

        # self.apply(self._init_except_pretrained)
        self.apply(init_bert_params)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        # add modality encoders later, so that they are not part of the model's parameters when model is initialized
        self.modality_encoders:nn.ModuleDict[Modality, nn.Module] = self._get_modality_encoders()
        self._freeze(self.modality_encoders)
        self.modality_encoders.eval()
        
    def make_block(self, drop_path, dim=None, heads=None):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return LayerResultBlock(
            dim=self.cfg.embed_dim if dim is None else dim,
            num_heads=self.cfg.num_heads if heads is None else heads,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=True,
            proj_drop=self.cfg.encoder_dropout,
            attn_drop=self.cfg.attention_dropout,
            drop_path=drop_path,
            norm_layer=make_layer_norm,
            layer_norm_first=self.cfg.layer_norm_first,
        )

    def _get_modality_encoders(self) -> None:
        modality_encoders = {}
        for mode in self.supported_modalities: # mode is instance of Modality
            state_dict_name = self.cfg.pretrained[mode.name.lower()]
            logger.info(f'Loading modality encoder for: {mode}')
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)
            mode_feature_extractor = d2v_model.modality_encoders[mode.name]
            modality_encoders[mode.name.lower()] = mode_feature_extractor
            
            for name, module in mode_feature_extractor.named_children():
                total_params = sum(p.numel() for p in module.parameters())
                logger.info(f"{name} has {total_params} parameters")

        return nn.ModuleDict(modality_encoders)

    def _init_except_pretrained(self, module:nn.Module):
        if all(param.requires_grad for param in module.parameters(recurse=False)):
            init_bert_params(module)

    def forward(
        self,
        modes:List[Modality],
        audio:torch.Tensor=None,
        image:torch.Tensor=None,
        text:torch.Tensor=None,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None, 
        mask:bool=False,
        features_only:bool=True,
        force_remove_masked:bool=False,
        remove_extra_tokens:bool=False,
        precomputed_mask=None,
        return_encoder_output:bool=False,
    ):
        assert len(modes)==1, f"This model accepts exactly modality indicator at a time, received: {modes}"
        n_sources = sum(mode is not None for mode in [audio, image, text])
        assert n_sources==1,\
            f"This model accepts exactly one modality data source at a time, got {n_sources}."
        
        assert modes[0] in self.supported_modalities, f"Unsupported modality: {modes[0]}, supported modalities are: {self.supported_modalities}"
        mode = modes[0].name.lower() # is now a string, ModuleDict does not support enums as keys
        
        if audio is not None:
            source = audio
        elif image is not None:
            source = image
        elif text is not None:
            source = text
        else:
            raise ValueError("Audio, image or text must be provided, found all to be None.")
        
        feature_extractor = self.modality_encoders[mode]
        with torch.no_grad():
            extractor_out = feature_extractor(
                source,
                padding_mask,
                mask,
                remove_masked=not features_only or force_remove_masked,
                clone_batch=1,
                mask_seeds=None,
                precomputed_mask=precomputed_mask,
            )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        # masked_alibi_bias = extractor_out.get("alibi_bias", None)
        # alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for blk in self.blocks:
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                # ab = masked_alibi_bias
                # if ab is not None and alibi_scale is not None:
                #     scale = (
                #         alibi_scale[i]
                #         if alibi_scale.size(0) > 1
                #         else alibi_scale.squeeze(0)
                #     )
                #     ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    #padding_mask=masked_padding_mask,
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

        out = {
            "x": x,
            "padding_mask": masked_padding_mask,
            "layer_results": layer_results,
            "mask": encoder_mask,
        }
        if return_encoder_output:
            out["encoder_output"] = extractor_out
        return out
    
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
    
    @torch.no_grad()
    def encode_modality(self, modes:Union[Modality, List[Modality]], source:torch.Tensor, padding_mask=None, normalize:bool=True):
        if isinstance(modes, List):
            assert len(modes)==1, 'Only one modality allowed when calling "encode_modality".'
            mode = modes[0]
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
            remove_extra_tokens=False, # important!
        )['x']

        if mode == Modality.AUDIO:
            output = output.mean(dim=1)
        else:
            output = output[:, 0, :]

        if normalize:
            output = F.normalize(output, dim=-1)
        
        return output # shape: (batch_size, embed_dim)

    def encode_audio(self, audio, padding_mask, normalize:bool=True):
        return self.encode_modality(source=audio,
                                    modes=Modality.AUDIO,
                                    padding_mask=padding_mask,
                                    normalize=normalize)
    
    def encode_image(self, image, normalize:bool=True):
        return self.encode_modality(source=image,
                                    modes=Modality.IMAGE,
                                    normalize=normalize)

    def encode_text(self, text, padding_mask, normalize:bool=True):
        return self.encode_modality(source=text,
                                    modes=Modality.TEXT,
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
                self._freeze(block.attn)
                block.attn.eval()

    def unfreeze_attention_blocks(self):
        if self.cfg.freeze_attention:
            for block in self.blocks:
                self._unfreeze(block.attn)
                block.attn.train()

    def _freeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze(self, module:nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True

    def remove_modalities_except(self, keep_modes:List[Modality]) -> None:
        """
        Removes all modalities from the model except the ones specified.
        Useful when fine-tuning the model on downstream task
        involving only a subset of the supported modalities.
        """
        for modality in self.supported_modalities:
            if modality not in keep_modes:
                del self.modality_encoders[modality.name.lower()]
