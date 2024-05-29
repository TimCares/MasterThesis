import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, Optional, List
import numpy as np
import os
import itertools
from utils import load_pretrained_d2v_model
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from data2vec_fairseq.models.modalities.base import ModalitySpecificEncoder
from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import contextlib
from modules import MOMEAltBlock

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from utils import prepare_output, get_max_saliency_patches, prepare_salient_patches

logger = logging.getLogger(__name__)


class AMMData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.automatic_optimization = False
        self.acc_grad_batches = self.cfg.acc_grad_batches

        self.prev_text_features = []
        self.prev_image_features = []

        self.model = AMMData2Vec(cfg=self.cfg.model)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        embed_dim = self.cfg.model.embed_dim
        self.itc_head = nn.Linear(embed_dim, embed_dim, bias=False)

        modality_str = self.cfg.teacher.name.lower()
        state_dict_name = self.cfg.model.pretrained[modality_str]
        state_dict_path = os.path.join(self.cfg.model.pretrained_path, state_dict_name)
        self.teacher = load_pretrained_d2v_model(state_dict_path=state_dict_path)

        teacher_modality = self.teacher.cfg.supported_modality.name
        teacher_extra_tokens = self.teacher.modality_encoders[teacher_modality].modality_cfg.num_extra_tokens
        del self.teacher.modality_encoders # we share it between teacher and student
    
        student_extra_tokens = self.model.modality_encoders[modality_str].modality_cfg.num_extra_tokens
        assert student_extra_tokens == teacher_extra_tokens, \
            f"Extra tokens mismatch: {student_extra_tokens} != {teacher_extra_tokens} " \
                f"between student and teacher model for modality '{modality_str}'"

        self.model._freeze(self.teacher)
        
        self.save_hyperparameters()

    def forward(self, input_dict):
        return self.model(**input_dict,
                          features_only=False,
                          return_encoder_output=True)
        # "feature_extractor_only" has higher precedence than "features_only" and "return_encoder_output"

    def training_step(self, batch:Dict[str, Any], batch_idx):
        opt = self.optimizers()

        step_loss, text_features, image_features = self._training_step(
            batch,
            batch_idx,
        )

        loss = step_loss / self.acc_grad_batches
        self.prev_text_features.append(text_features)
        self.prev_image_features.append(image_features)

        if (batch_idx + 1) % self.acc_grad_batches == 0:
            concat_text_features = torch.cat(self.prev_text_features, dim=1)
            concat_image_features = torch.cat(self.prev_text_features, dim=1)
            itc_loss = self.itc_loss(text_features=concat_text_features, image_features=concat_image_features)
            loss += itc_loss
            self.manual_backward(loss) 

            opt.step()
            # clear
            opt.zero_grad()
            self.prev_text_features = []
            self.prev_image_features = []

            self.log("train/itc_loss", loss, prog_bar=True)
            self.log("train/loss", loss, prog_bar=True)
        else:
            self.manual_backward(loss)
        

    def training_step(self, batch:Dict[str, Any], batch_idx):
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):   
        if 'target' in batch:
            batch.pop('target') # unused, layer activations are the targets

        text = batch['text']
        padding_mask = batch['padding_mask']
        image = batch['image']

        output_dict_text = self({'x': text, 'padding_mask': padding_mask}) # call "forward"
        output_dict_image = self({'x': image}) # call "forward"

        modality:Modality = batch['modality']
        assert modality == Modality.VL

        # in output_dict, because "return_encoder_output=True" in "forward"
        precomputed_encoder_output = output_dict_image['encoder_output']

        with torch.no_grad():
            target = self.teacher.extract_features(
                source=image,
                mode=None, # determined automatically in model
                padding_mask=None, # the padding mask is provided in the precomputed_encoder_output and used by the teacher model
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False, # special tokens are also regressed
                precomputed_encoder_output=precomputed_encoder_output,
            )
        
        target = target['layer_results']
        kd_target = prepare_output(target, modality)

        kd_losses = []
        for output_dict in [output_dict_text, output_dict_image]:
            kd_pred = prepare_output(output_dict['layer_results'], modality)
            _kd_loss = self.kd_loss(input=kd_pred, target=kd_target)
            kd_losses.append(_kd_loss)
        
        kd_loss = sum(kd_losses) / 2
        
        text_features = output_dict_text['layer_results'][-1].mean(dim=1)
        image_features = output_dict_image['layer_results'][-1].mean(dim=1)

        itc_loss = self.itc_loss(text_features=text_features, image_features=image_features)
        loss = kd_loss + itc_loss

        self.log(f"{stage}/kd_loss", kd_loss, prog_bar=True)
        self.log(f"{stage}/itc_loss", itc_loss, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss, text_features, image_features
    
    def itc_loss(self, text_features:torch.Tensor, image_features:torch.Tensor) -> torch.Tensor:
        text_features = self.itc_head(text_features)
        image_features = self.itc_head(image_features)

        scale = self.logit_scale.exp()
        with torch.no_grad():
            text_features = F.normalize(text_features, dim=-1)
            image_features = F.normalize(image_features, dim=-1)
        
        logits_per_image = scale * image_features @ text_features.t()
        logits_per_text = scale * text_features @ image_features.t()
        target = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), target)
            + F.cross_entropy(logits_per_text.float(), target)
        ) / 2
        return itc_loss
    
    
    def kd_loss(self, input:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        input = input.contiguous()
        input = input.view(-1, input.size(-1)).float() # (B, D, C) -> (B*D, C)
        target = target.contiguous()
        target = target.view(-1, target.size(-1)).float() # (B, D, C) -> (B*D, C)

        assert input.shape == target.shape # this must be the case

        return F.mse_loss(input, target, reduction="none").float().mean()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), 
                                      lr=self.cfg.optimizer.lr,
                                      betas=tuple(self.cfg.optimizer.betas),
                                      eps=self.cfg.optimizer.eps,
                                      weight_decay=self.cfg.optimizer.weight_decay)
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


@dataclass
class PretrainedStateDictsConfig():
    audio:str = 'base_libri.pt'
    image:str = 'base_imagenet.pt'
    text:str = 'nlp_base.pt'

@dataclass
class BlockInitConfig():
    init_from: Optional[Modality] = None # if None, then no blocks are initialized
    init_type: Optional[str] = None # 'attention' or 'block', only relevant if "init_from" not None
    block_indices: Optional[List[int]] = None # if None, then all blocks are initialized
    freeze_blocks: Optional[List[int]] = None # if None, then all blocks are frozen, if empty list, then no blocks are frozen

@dataclass
class AMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

    supported_modalities: List[Modality] = field(default_factory=lambda: [Modality.AUDIO, Modality.IMAGE, Modality.TEXT])

    block_init_cfg: BlockInitConfig = field(default_factory=BlockInitConfig)

    n_fuzed_layers: int = 2

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
        self.supported_modalities = cfg.supported_modalities
        self.fine_tuning = False

        make_layer_norm = partial(
            nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        blocks = []
        if self.cfg.block_init_cfg.init_from is None:
            for i in range(self.cfg.depth):
                if self.cfg.depth - i <= self.cfg.n_fuzed_layers:
                    blocks.append(self.make_block(drop_path=dpr[i], multimodal=True, with_fuzed=True))
                else:
                    blocks.append(self.make_block(drop_path=dpr[i], multimodal=True))

            self.blocks = nn.ModuleList(blocks)

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
        
    def make_block(self, drop_path, dim=None, heads=None, multimodal=False, with_fuzed=False):
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
        )

    def _init_from_pretrained(self) -> None:
        modality_encoders = {}
        for modality in list(Modality): # modality is instance of Modality, "list(Modality)" is a list of all enum/modality values
            if modality not in self.supported_modalities and modality != self.cfg.block_init_cfg.init_from:
                continue

            state_dict_name = self.cfg.pretrained[modality.name.lower()]
            logger.info(f'Loading modality encoder for: {modality}')
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)

            if modality in self.supported_modalities:
                modality_feature_extractor = d2v_model.modality_encoders[modality.name]
                modality_encoders[modality.name.lower()] = modality_feature_extractor
                
                for name, module in modality_feature_extractor.named_children():
                    total_params = sum(p.numel() for p in module.parameters())
                    logger.info(f"{name} has {total_params} parameters")

            if modality == self.cfg.block_init_cfg.init_from:
                self._init_blocks(d2v_model=d2v_model)

        self.modality_encoders:nn.ModuleDict[str, nn.Module] = nn.ModuleDict(modality_encoders)
        self._freeze(self.modality_encoders)

    def _init_except_pretrained(self, module:nn.Module):
        if all(param.requires_grad for param in module.parameters(recurse=False)):
            init_bert_params(module)

    def forward(
        self,
        x:torch.Tensor,
        modality:Modality,
        id:torch.Tensor=None,
        padding_mask:torch.Tensor=None,
        remove_extra_tokens:bool=False,
        return_encoder_output:bool=False,
        features_only:bool=False,
        precomputed_encoder_output:Optional[Dict[str, torch.Tensor]]=None,
    ):

        if precomputed_encoder_output is None:
            feature_extractor:ModalitySpecificEncoder = self.modality_encoders[modality.name.lower()]
            with torch.no_grad() if not self.fine_tuning else contextlib.ExitStack():
                extractor_out = feature_extractor(
                    features=x,
                    padding_mask=padding_mask,
                    mask=False,
                    remove_masked=False,
                )
        else:
            extractor_out = precomputed_encoder_output

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
                    modality=modality,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                
                if not features_only:
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
            # teacher gets all tokens. Only if we do MaskedKD, then the teacher only gets the unmasked tokens
            out["encoder_output"] = extractor_out
        return out
    

    def extract_features(
        self, x:torch.Tensor, modality:Modality, padding_mask=None, remove_extra_tokens=True
    ):
        res = self.forward(
            x=x,
            modality=modality,
            padding_mask=padding_mask,
            remove_extra_tokens=remove_extra_tokens,
            features_only=True,
        )
        return res
    
    @torch.no_grad()
    def encode_modality(self, x:torch.Tensor, modality:Modality, padding_mask=None, normalize:bool=True):
        output = self.extract_features(
            x=x,
            modality=modality,
            padding_mask=padding_mask,
            remove_extra_tokens=False, # important!
        )['x']

        output = output.mean(dim=1)

        if normalize:
            output = F.normalize(output, dim=-1)
        
        return output # shape: (batch_size, embed_dim)

    def encode_audio(self, audio, padding_mask, normalize:bool=True):
        return self.encode_modality(x=audio,
                                    modality=Modality.AUDIO,
                                    padding_mask=padding_mask,
                                    normalize=normalize)
    
    def encode_image(self, image, normalize:bool=True):
        return self.encode_modality(x=image,
                                    modality=Modality.IMAGE,
                                    normalize=normalize)

    def encode_text(self, text, padding_mask, normalize:bool=True):
        return self.encode_modality(x=text,
                                    modality=Modality.TEXT,
                                    padding_mask=padding_mask,
                                    normalize=normalize)

    def _init_blocks(self, d2v_model:Data2VecMultiModel) -> None:
        init_cfg:BlockInitConfig = self.cfg.block_init_cfg
        
        if init_cfg.block_indices is None:
            take_block_indices = [i for i in range(self.cfg.depth)]
        else:
            take_block_indices = init_cfg.block_indices

        logger.info(f"Initializing blocks from pretrained modality: {init_cfg.init_from}")
        logger.info(f"Init type: {init_cfg.init_type}")
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

        self.blocks = nn.ModuleList(self.blocks)

        if init_cfg.freeze_blocks is None:
            if init_cfg.init_type == 'attention':
                logger.info("Freezing all attention blocks")
                self.freeze_attention_blocks()
            else:
                logger.info("Freezing all blocks")
                self._freeze(self.blocks)
        elif len(init_cfg.freeze_blocks) == 0:
            pass # do not freeze any blocks
        else:
            if init_cfg.init_type == 'attention':
                logger.info(f"Freezing attention block indices: {init_cfg.freeze_blocks}")
                for idx in init_cfg.freeze_blocks:
                    self._freeze(self.blocks[idx].attn)
            else:
                logger.info(f"Freezing block indices: {init_cfg.freeze_blocks}")
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


    def prepare_fine_tuning(self, keep_modality:Modality) -> None:
        self.fine_tuning = True
        self.cfg.mask = False
        self._remove_modalities_except(keep_modality=keep_modality)
        self._unfreeze(self)

    def _remove_modalities_except(self, keep_modality:Modality) -> None:
        """
        Removes all modalities from the model except the ones specified.
        Useful when fine-tuning the model on downstream task
        involving only a subset of the supported modalities.
        """
        # comparison done on name basis, as on "enum" basis yields problems after serialization
        for modality in self.supported_modalities:
            modality_str = modality.name.lower()
            if modality_str != keep_modality:
                del self.modality_encoders[modality_str] # includes removing the decoder
