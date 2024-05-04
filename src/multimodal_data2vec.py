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
from data2vec_fairseq.data.modality import Modality
from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel
from data2vec_fairseq.models.modalities.base import MaskInfo, ModalitySpecificEncoder
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import contextlib
import math
from omegaconf import open_dict

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from data2vec_fairseq.models.modalities.modules import AltBlock

logger = logging.getLogger(__name__)

def prepare_output(out:List[torch.Tensor], modality:Modality) -> List[torch.Tensor]:
    out = [
        F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
        for tl in out  # BTC -> BCT -> BTC
    ]

    y = out[0].float()
    for tl in out[1:]:
        y.add_(tl.float())
    y = y.div_(len(out))

    if modality == Modality.IMAGE:
        y = F.layer_norm(y, y.shape[-1:])
    return y

class KDData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        state_dict_name = self.cfg.model.pretrained['image']
        state_dict_path = os.path.join(self.cfg.model.pretrained_path, state_dict_name)
        self.teacher = load_pretrained_d2v_model(state_dict_path=state_dict_path, keep_decoder=False)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        teacher_mode = self.teacher.cfg.supported_modality.name
        teacher_extra_tokens = self.teacher.modality_encoders[teacher_mode].modality_cfg.num_extra_tokens
        del self.teacher.modality_encoders # we share it between teacher and student
        
        self.model = KDMMData2Vec(cfg=self.cfg.model)
        
        assert self.model.modality_encoders['image'].modality_cfg.num_extra_tokens == teacher_extra_tokens, \
            f"Extra tokens mismatch: {self.model.modality_encoders['image'].modality_cfg.num_extra_tokens} != {teacher_extra_tokens} " \
                "between student and teacher model for modality 'image'" # TODO: add support for other modalities
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict, features_only=False, return_encoder_output=True)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        if 'target' in batch:
            batch.pop('target')
        output_dict = self(batch) # call "forward"

        precomputed_encoder_output = output_dict['encoder_output']
        if self.cfg.model.mask_student_input:
            # we do knowledge distillation the same way d2v is trained => teacher gets unmasked input
            precomputed_encoder_output['x'] = precomputed_encoder_output['x_unmasked']
            precomputed_encoder_output['padding_mask'] = precomputed_encoder_output['original_padding_mask']
            del precomputed_encoder_output['x_unmasked']
            del precomputed_encoder_output['original_padding_mask']

        with torch.no_grad():
            target = self.teacher.extract_features(
                    source=batch['image'],
                    mode=None, # determined automatically in model
                    padding_mask=None, # the padding mask is provided in the precomputed_encoder_output and used by the teacher model
                    mask=False, # we are creating targets from a teacher model for the student model, so no mask
                    remove_extra_tokens=self.cfg.model.mask_student_input, # "decoder_input" in d2v (decoder in student model) removes extra tokens
                    # ... so we need to remove them from the teacher output as well if we mask the student input. If not, then we do not need to remove them
                    # because we also regress the extra tokens, if they are present.
                    precomputed_encoder_output=precomputed_encoder_output,
                )
        
        target = target['layer_results']
        target = prepare_output(target, Modality.IMAGE)

        if self.cfg.model.mask_student_input:
            pred = output_dict['x']

            if self.cfg.model.clone_batch > 1:
                target = target.repeat_interleave(self.cfg.model.clone_batch, 0)

            if self.cfg.model.regress_masked_only:
                masked_b = output_dict['mask'].mask.bool()
                assert pred.size(1) == masked_b.size(1), f"Size mismatch: {pred.size(1)} != {masked_b.size(1)}"
                assert target.size(1) == masked_b.size(1), f"Size mismatch: {target.size(1)} != {masked_b.size(1)}"
                pred = pred[masked_b]
                target = target[masked_b]
        else:
            pred = output_dict['layer_results']
            pred = prepare_output(pred, Modality.IMAGE)
        

        loss = self.kd_loss(input=pred, target=target)
        self.log("train/loss", loss, prog_bar=True)
        if batch['modes'][0] == Modality.IMAGE:
            self.log("train/loss_img", loss, prog_bar=True)
        elif batch['modes'][0] == Modality.AUDIO:
            self.log("train/loss_audio", loss, prog_bar=True)
        else:
            self.log("train/loss_text", loss, prog_bar=True)
        return loss
                
    
    def kd_loss(self, input:torch.Tensor, target:torch.Tensor) -> float:
        input = input.contiguous()
        input = input.view(-1, input.size(-1)).float() # (B, D, C) -> (B*D, C)
        target = target.contiguous()
        target = target.view(-1, target.size(-1)).float() # (B, D, C) -> (B*D, C)

        assert input.shape == target.shape # this must be the case

        loss = F.mse_loss(input, target, reduction="none").float()

        if self.cfg.model.loss_scale is not None:
            scale = self.cfg.model.loss_scale
        else:
            scale = 1 / math.sqrt(input.size(-1))
        
        reg_loss = loss * scale
        
        return reg_loss.mean().float()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
        if self.cfg.optimizer.warmup:
            if self.cfg.optimizer_schedule.type == 'cosine':
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
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "cosine_w_warmup"}]
        else:
            return optimizer


@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

@dataclass
class BlockInitConfig():
    init_from: Optional[Modality] = Modality.IMAGE # if None, then no blocks are initialized
    init_type: str = 'attention' # 'attention' or 'block'
    block_indices: Optional[List[int]] = None # if None, then all blocks are initialized
    freeze_blocks: Optional[List[int]] = None # if None, then all blocks are frozen, if empty list, then no blocks are frozen

@dataclass
class KDMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    supported_modalities: List[Modality] = field(default_factory=lambda: [Modality.AUDIO, Modality.IMAGE, Modality.TEXT])

    block_init_cfg: BlockInitConfig = field(default_factory=BlockInitConfig)

    mask_student_input: bool = False
    regress_masked_only: bool = False

    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    embed_dim: int = 768

    clone_batch: int = 1

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

    seed: int = 42

class KDMMData2Vec(nn.Module):
    def __init__(self,
                 cfg: KDMMData2VecConfig,
                 ):
        super(KDMMData2Vec, self).__init__()
        self.cfg = cfg
        self.supported_modalities = cfg.supported_modalities
        self.fine_tuning = False

        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        if self.cfg.block_init_cfg.init_from is None:
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

        # init pretrained later, so that they are not part of the model's parameters when model is initialized
        self._init_from_pretrained()
        assert hasattr(self, 'blocks'), "Blocks must be initialized before initializing the model."
        
    def make_block(self, drop_path, dim=None, heads=None):
        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        return AltBlock(
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
            ffn_targets=True,
        )

    def _init_from_pretrained(self) -> None:
        modality_encoders = {}
        for mode in self.supported_modalities: # mode is instance of Modality
            state_dict_name = self.cfg.pretrained[mode.name.lower()]
            logger.info(f'Loading modality encoder for: {mode}')
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            # if we are masking the input to the student model, then we need to keep the decoder
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path, keep_decoder=self.cfg.mask_student_input)
            mode_feature_extractor = d2v_model.modality_encoders[mode.name]
            modality_encoders[mode.name.lower()] = mode_feature_extractor
            
            for name, module in mode_feature_extractor.named_children():
                total_params = sum(p.numel() for p in module.parameters())
                logger.info(f"{name} has {total_params} parameters")

            if self.cfg.block_init_cfg.init_from is not None and mode == self.cfg.block_init_cfg.init_from:
                self._init_blocks(d2v_model=d2v_model)

        self.modality_encoders:nn.ModuleDict[str, nn.Module] = nn.ModuleDict(modality_encoders)
        self._freeze(self.modality_encoders)
        self.modality_encoders.eval()

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
        remove_extra_tokens:bool=False,
        precomputed_mask=None,
        return_encoder_output:bool=False,
        features_only:bool=False,
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
        
        mask_condition = self.cfg.mask_student_input and not features_only

        feature_extractor = self.modality_encoders[mode]
        with torch.no_grad() if not self.fine_tuning else contextlib.ExitStack():
            extractor_out = feature_extractor(
                source,
                padding_mask,
                mask=mask_condition,
                remove_masked=mask_condition,
                clone_batch=self.cfg.clone_batch if mask_condition else 1,
                mask_seeds=None,
                precomputed_mask=precomputed_mask,
                process_unmasked=mask_condition, # if True, then "x_unmasked" is in "extractor_out" dict
            )

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
                layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        # we do not use the decoder for features extraction, when we use vanilla KD
        if features_only or not self.cfg.mask_student_input:
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
        
        del layer_results # not needed
    
        with torch.no_grad():
            if feature_extractor.decoder is not None:
                x = self.forward_decoder( # expands input back to original size -> adds masked time steps back
                    x,
                    feature_extractor,
                    feature_extractor.decoder,
                    encoder_mask,
                )

        out = {
            "x": x,
            "padding_mask": masked_padding_mask,
            "mask": encoder_mask,
        }
        if return_encoder_output:
            out["encoder_output"] = extractor_out

        return out


    def forward_decoder(
        self,
        x,
        feature_extractor,
        decoder,
        mask_info,
    ):
        x = feature_extractor.decoder_input(x, mask_info)
        x = decoder(*x)

        return x
    
    def extract_features(
        self, audio=None, image=None, text=None, modes:List[Modality]=None, padding_mask=None, remove_extra_tokens=True
    ):
        res = self.forward(
            audio=audio,
            image=image,
            text=text,
            modes=modes,
            padding_mask=padding_mask,
            remove_extra_tokens=remove_extra_tokens,
            features_only=True,
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

    def _init_blocks(self, d2v_model:Data2VecMultiModel) -> None:
        init_cfg:BlockInitConfig = self.cfg.block_init_cfg
        
        if init_cfg.block_indices is None:
            take_block_indices = [i for i in range(self.cfg.depth)]
        else:
            take_block_indices = init_cfg.block_indices

        logger.info(f"Initializing blocks from pretrained mode: {init_cfg.init_from}")
        logger.info(f'Taking pretrained block indices: {take_block_indices}')

        if init_cfg.init_type == 'attention':
            dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)
            self.blocks = [self.make_block(dpr[i]) for i in range(self.cfg.depth)]
        else:
            self.blocks = []

        for idx in take_block_indices:
            if init_cfg.init_type == 'attention':
                self.blocks[idx].attn = d2v_model.blocks[idx].attn
            else:
                self.blocks.append(d2v_model.blocks[idx])

        n_blocks_missing = self.cfg.depth - len(take_block_indices)
        if n_blocks_missing > 0:
            assert init_cfg.init_type == 'block', "Only 'block' initialization supports adding new blocks"
            logger.info(f"Adding {n_blocks_missing} new blocks")
            dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, n_blocks_missing)
            for i in range(n_blocks_missing):
                self.blocks.append(self.make_block(dpr[i]))

        self.blocks = nn.ModuleList(self.blocks)

        if init_cfg.freeze_blocks is None:
            if init_cfg.init_type == 'attention':
                self.freeze_attention_blocks()
            else:
                self._freeze(self.blocks)
        elif len(init_cfg.freeze_blocks) == 0:
            pass # do not freeze any blocks
        else:
            if init_cfg.init_type == 'attention':
                for idx in init_cfg.freeze_blocks:
                    self._freeze(self.blocks[idx].attn)
            else:
                for idx in init_cfg.freeze_blocks:
                    self._freeze(self.blocks[idx])
        
    
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


    def prepare_fine_tuning(self, keep_modes:List[Modality]) -> None:
        self.cfg.clone_batch = 1
        self.fine_tuning = True
        self._remove_modalities_except(keep_modes=keep_modes)
        self._unfreeze(self.modality_encoders)

    def _remove_modalities_except(self, keep_modes:List[Modality]) -> None:
        """
        Removes all modalities from the model except the ones specified.
        Useful when fine-tuning the model on downstream task
        involving only a subset of the supported modalities.
        """
        for modality in self.supported_modalities:
            if modality not in keep_modes:
                del self.modality_encoders[modality.name.lower()] # includes removing the decoder
            else:
                if hasattr(self.modality_encoders[modality.name.lower()], 'decoder'):
                    del self.modality_encoders[modality.name.lower()].decoder # not needed in any case
