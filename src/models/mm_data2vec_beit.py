import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from typing import Dict, Any, Optional, List
import numpy as np
import os
from utils import load_pretrained_d2v_model, prepare_output
from omegaconf import OmegaConf
import logging
import pytorch_lightning as L
from dataclasses import dataclass, field
from data2vec_fairseq.data.modality import Modality
from data2vec_fairseq.models.modalities.base import ModalitySpecificEncoder
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup
import contextlib
from modules import MOMEAltBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling

logger = logging.getLogger(__name__)
    

class AMMData2VecPreTrainingLightningModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.last_n_layer_targets = self.cfg.model.n_fuzed_layers

        self.model = AMMData2Vec(cfg=self.cfg.model)

        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.teacher = self.load_beit2_teacher()

        self.model._freeze(self.teacher)
        
        self.save_hyperparameters()

    def forward(self, input_dict, return_encoder_output=False):
        return self.model(**input_dict,
                          features_only=False,
                          return_encoder_output=return_encoder_output)

    def training_step(self, batch:Dict[str, Any], batch_idx):
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch:Dict[str, Any], batch_idx:int):
        return self._step(batch, batch_idx, stage='val')

    def _step(self, batch:Dict[str, Any], batch_idx, stage:str='train'):   
        if 'target' in batch:
            batch.pop('target') # unused, layer activations are the targets

        # assert torch.unique(batch['id']).shape[0] == batch['id'].shape[0], "IDs must be unique for ITC loss."
        # batch['id'] later important for ITC loss!

        text = batch['text']
        padding_mask = batch['padding_mask']
        image = batch['image']

        output_dict_text = self({
            'x': text,
            'modality': Modality.TEXT,
            'padding_mask': padding_mask,
        }) # call "forward"

        output_dict_image = self({
            'x': image,
            'modality': Modality.IMAGE
        }) # call "forward"

        # no mask
        bool_masked_pos = torch.zeros((image.shape[0], self.teacher.patch_embed.num_patches), dtype=torch.bool).to(image.device)

        with torch.no_grad():
            target = self.teacher.forward_features(
                x=image,
                bool_masked_pos=bool_masked_pos,
            )

        kd_losses = []
        for output_dict in [output_dict_text, output_dict_image]:
            _kd_loss = F.mse_loss(output_dict['x'][:, 0], target['x'][:, 0], reduction="mean")
            kd_losses.append(_kd_loss)
        
        kd_loss = sum(kd_losses)
        
        # no layer norm here, as last operation of each block is layer norm (therefore also of last block)
        # we do not do ...[layer_results][-1] as we do not want the raw ffn outputs, but the layer normed ones
        #text_features = output_dict_text['x'][:, 0]
        #image_features = output_dict_image['x'][:, 0]

        #itc_loss = self.itc_loss(text_features=text_features, image_features=image_features)
        loss = kd_loss# + itc_loss

        self.log(f"{stage}/kd_text_loss", kd_losses[0])
        self.log(f"{stage}/kd_image_loss", kd_losses[1])
        self.log(f"{stage}/kd_loss", kd_loss)
        #self.log(f"{stage}/itc_loss", itc_loss)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        
        return loss
    
    def itc_loss(self, text_features:torch.Tensor, image_features:torch.Tensor, stage:str='train') -> torch.Tensor:
        text_features = self.model.itc_text_head(text_features)
        image_features = self.model.itc_img_head(image_features)

        scale = self.logit_scale.exp()

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
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
    
    def _log_similarity(self, logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, stage:str='train') -> None:
        diagonal_mask = torch.eye(logits_per_image.size(0)).bool()
        for logits, name in [(logits_per_image, "i2t"), (logits_per_text, "t2i")]:
            mean_pos_sim = logits[diagonal_mask].mean()
            mean_neg_sim = logits[~diagonal_mask].mean()
            self.log(f"{stage}/{name}_mean_pos_similarity", mean_pos_sim)
            self.log(f"{stage}/{name}_mean_neg_similarity", mean_neg_sim)


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
class AMMData2VecConfig():
    pretrained_path:str = '../models'
    pretrained: PretrainedStateDictsConfig = field(default_factory=PretrainedStateDictsConfig)

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
        self.supported_modalities = [Modality.IMAGE, Modality.TEXT]
        self.fine_tuning = False

        # initialized through self.apply(init_bert_params)
        # self.itc_img_head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=False)
        # self.itc_text_head = nn.Linear(self.cfg.embed_dim, self.cfg.embed_dim, bias=False)
        # self.token_type_embedding = nn.Embedding(len(self.supported_modalities), self.cfg.embed_dim)

        # make_layer_norm = partial(
        #     nn.LayerNorm, eps=self.cfg.norm_eps, elementwise_affine=self.cfg.norm_affine
        # )

        self.dropout_input = nn.Dropout(self.cfg.dropout_input)

        dpr = np.linspace(self.cfg.start_drop_path_rate, self.cfg.end_drop_path_rate, self.cfg.depth)

        blocks:List[MOMEAltBlock] = []
        fuzed_block_indices = list(range(self.cfg.depth - self.cfg.n_fuzed_layers, self.cfg.depth))
        logger.info(f"Fuzed block indices: {fuzed_block_indices}")
        for i in range(self.cfg.depth):
            if self.cfg.depth - i <= self.cfg.n_fuzed_layers:
                blocks.append(self.make_block(drop_path=dpr[i], multimodal=True, with_fuzed=True))
            else:
                blocks.append(self.make_block(drop_path=dpr[i], multimodal=True))

        self.blocks:nn.ModuleList[str, MOMEAltBlock] = nn.ModuleList(blocks)

        self.layerdrop = self.cfg.layerdrop
        self.mask_seed = self.cfg.seed

        # self.apply(self._init_except_pretrained)
        self.apply(init_bert_params)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        # init pretrained later, so that they are not part of the model's parameters when model is initialized
        assert hasattr(self, 'blocks'), "Blocks must be initialized before initializing the model."
        self._init_from_pretrained()
        
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
        start_fuzed = self.cfg.depth-self.cfg.n_fuzed_layers
        for modality in self.supported_modalities:
            state_dict_name = self.cfg.pretrained[modality.name.lower()]
            state_dict_path = os.path.join(self.cfg.pretrained_path, state_dict_name)
            d2v_model = load_pretrained_d2v_model(state_dict_path=state_dict_path)

            # for i in range(start_fuzed):
            #     self.blocks[i].init_from_pretrained(
            #         pretained_block=d2v_model.blocks[i],
            #         modality=modality,
            #         init_attention=modality == Modality.IMAGE,
            #     )
            # if modality == Modality.IMAGE:
            #     for i in range(start_fuzed, self.cfg.depth):
            #         assert self.blocks[i].with_fuzed
            #         self.blocks[i].init_attention_from_pretrained(
            #             pretained_block=d2v_model.blocks[i],
            #             modality=modality,
            #         )
            
            modality_feature_extractor = d2v_model.modality_encoders[modality.name]
            modality_encoders[modality.name.lower()] = modality_feature_extractor

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
    ):

        feature_extractor:ModalitySpecificEncoder = self.modality_encoders[modality.name.lower()]
        with torch.no_grad() if not self.fine_tuning else contextlib.ExitStack():
            extractor_out = feature_extractor(
                features=x,
                padding_mask=padding_mask,
                mask=False,
                remove_masked=False,
            )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        # token_ids = torch.full(x.shape[:-1], modality.value-2).to(x.device)
        # x = x + self.token_type_embedding(token_ids)

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

        output = output[:, 0]
        # if modality == Modality.IMAGE:
        #     output = self.itc_img_head(output)
        # elif modality == Modality.TEXT:
        #     output = self.itc_text_head(output)
        # else:
        #     raise ValueError(f"Modality {modality} not supported")

        if normalize:
            output = output / output.norm(dim=-1, keepdim=True)
        
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


    def prepare_fine_tuning(self, keep_modality:Modality, remove_itc_head:bool=True) -> None:
        self.fine_tuning = True
        if remove_itc_head:
            del self.itc_head
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
                del self.modality_encoders[modality_str]

                for block in self.blocks:
                    block.remove_modality(modality)
