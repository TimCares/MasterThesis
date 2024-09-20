from omegaconf import OmegaConf
import os
import torch
import math
import random
import numpy as np
import logging
from functools import partial
from collections import namedtuple
import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import torch.nn as nn
import torch.nn.functional as F
from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel
from data2vec_fairseq.models.data2vec2 import Data2VecMultiConfig
from data2vec_fairseq.models.modalities.base import ModalitySpecificEncoder
from data2vec_fairseq.data.modality import Modality
from fairseq.data import Dictionary
from fairseq.dataclass.utils import merge_with_parent
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling

logger = logging.getLogger(__name__)

def load_model(pretrained_model_cfg:DictConfig,
               model_state_dict:OrderedDict[str, torch.Tensor]) -> Data2VecMultiModel:
    
    pretrained_model_cfg = merge_with_parent(Data2VecMultiConfig(), pretrained_model_cfg, remove_missing=True)

    logger.info(f"Modality used: {pretrained_model_cfg.supported_modality}")
    if pretrained_model_cfg.supported_modality.name.lower() == 'text':
        Task = namedtuple('Task', 'source_dictionary')

        dictionary = Dictionary.load(os.path.join('..', 'data', "dict.txt"))
        dictionary.add_symbol("<mask>")
        dummy_task = Task(source_dictionary=dictionary)
    else:
        dummy_task = False

    model = Data2VecMultiModel.build_model(pretrained_model_cfg, task=dummy_task)

    result = model.load_state_dict(model_state_dict)
    logger.info(f'Loaded state dict, result: {result}')
    return model


def load_pretrained_d2v_model(state_dict_path:str, keep_decoder:bool=False, remove_dropout:bool=False) -> Data2VecMultiModel:
    model_meta_data = torch.load(state_dict_path)
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])
    if remove_dropout:
        for k in pretrained_model_cfg.keys():
            if 'drop' in k:
                pretrained_model_cfg[k] = 0.0
    model = load_model(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])

    # removes decoder, and all encoders with modality != supported modality
    model.remove_pretraining_modules(modality=pretrained_model_cfg.supported_modality, keep_decoder=keep_decoder)

    return model


def pad_text_sequence(tokens:List[int],
                      num_max_bpe_tokens:int,
                      pad_idx:int,
                      bos_idx:int,
                      eos_idx:int) -> Tuple[List[int], List[int]]:
    
    if len(tokens) > num_max_bpe_tokens - 2:
        tokens = tokens[:num_max_bpe_tokens - 2]
    tokens = ([bos_idx] if tokens[0]!=bos_idx else []) + tokens + ([eos_idx] if tokens[-1]!=eos_idx else [])
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (num_max_bpe_tokens - num_tokens)
    language_tokens = tokens + [pad_idx] * (num_max_bpe_tokens - num_tokens)

    return language_tokens, padding_mask


def prepare_output(out:List[torch.Tensor], modality:Optional[Modality]=None, norm:bool=True) -> torch.Tensor:
    if norm:
        out = [
            F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
            for tl in out  # BTC -> BCT -> BTC
        ]

    y = out[0].float()
    for tl in out[1:]:
        y.add_(tl.float())
    y = y.div_(len(out))

    if modality is not None and modality == Modality.IMAGE:
        y = F.layer_norm(y, y.shape[-1:])
    return y


def get_max_saliency_patches(
        frac_keep_tokens:float,
        attn_results:List[torch.Tensor],
        extractor_out:Dict[str, torch.Tensor],
        feature_extractor:ModalitySpecificEncoder,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
    alibi_scale = extractor_out.get("alibi_scale", None)
    x_unmasked = extractor_out['x_pre_context'] # will definitely be unmasked, as "d2v_masking" is False
    masked_padding_mask = extractor_out["padding_mask"]
    masked_alibi_bias = extractor_out.get("alibi_bias", None)
    
    # TODO: think about option for audio -> no special token at the beginning
    attn_score = attn_results[0].float()
    for tl in attn_results[1:]:
        attn_score.add_(tl.float())
    attn_score = attn_score.div_(len(attn_results))

    ### partly adapted from: https://arxiv.org/pdf/2302.10494 (MaskedKD) ###
    num_keep = int(frac_keep_tokens*attn_score.size(-1)) # or size(-2) because shape is (B, H, T, T)
    keep_timesteps = torch.topk(attn_score.mean(dim=1)[:, 0, 1:], num_keep).indices # attn_score.mean(dim=1): B x H x T x T -> B x T x T

    # pre context already contains special token at the beginning
    cls_save = x_unmasked[:, 0, :].unsqueeze(dim=1) # B x 1 x D
    x_unmasked = x_unmasked[:, 1:, :]

    embed_dim = x_unmasked.size(-1)
    index = keep_timesteps.unsqueeze(-1).repeat(1, 1, embed_dim)
    x_unmasked_tokens_only = torch.gather(x_unmasked, dim=1, index=index)
    x_unmasked_tokens_only = torch.cat((cls_save , x_unmasked_tokens_only), dim=1)
    # B x num_keep+1 x D -> one (+1) stems from additional special token

    ### end of adaptation ###

    if masked_padding_mask is not None:
        padding_masked_unmasked_tokens = torch.gather(masked_padding_mask[:, 1:], dim=1, index=keep_timesteps)
        # the following should never raise and exception, as long as "num_keep" is not larger than the number of
        # non-padding tokens in the input, this is because padded tokens have an attention score of 0, which is the minimum
        assert (~padding_masked_unmasked_tokens).all(), "All non-masked MaskedKD tokens should be padding tokens."

    # now compute modality encoder output for the teacher model, we mask the tokens before the
    # context encoder of the modality encoder -> same approach as d2v masking
    # we reuse the features before the context encoder, saved from the forward pass of the modality encoder at the beginning
    x_unmasked_tokens_only = feature_extractor.context_encoder(
        x_unmasked_tokens_only, # our "x" here
        masked_padding_mask, # masked padding mask can be reused from previous modality encoder forward pass
        masked_alibi_bias, # alibi can be reused from previous modality encoder forward pass
        alibi_scale[: feature_extractor.modality_cfg.prenet_depth]
        if alibi_scale is not None
        else None,
    )
    return x_unmasked_tokens_only, keep_timesteps

def prepare_salient_patches(
        layer_results:List[torch.Tensor],
        keep_timesteps:torch.Tensor,
        modality:Modality,
        norm_first:bool,
        ) -> torch.Tensor:
    # if final_attn_layer_saliency_score is True, then "layer_results" only contains one element, the last layer
    # ... but it can be treated the same way as if it contains all layers

    if norm_first: # norms over all time steps, including those that are not used for the teacher
        layer_results = [F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2) for tl in layer_results]
    
    layer_results = torch.stack(layer_results)
    cls_save = layer_results[:, :, 0, :].unsqueeze(dim=2) # L x B x 1 x D
    layer_results = layer_results[:, :, 1:, :]

    embed_dim = layer_results.size(-1)
    # add dim for embedding dimension (-1) and for layer dimension (0)
    n_layers = layer_results.size(0)
    index = keep_timesteps.unsqueeze(-1).unsqueeze(0).repeat(n_layers, 1, 1, embed_dim)
    layer_results = torch.gather(layer_results, dim=2, index=index)
    layer_results = torch.cat((cls_save , layer_results), dim=2)
    # "layer_results" now only consists of same tokens as "x_unmasked_tokens_only"

    # averaged and normed layer results only on unmasked tokens
    # -> teacher only gets the unmasked tokens, and the teacher output is normed,
    # so we need to norm only the unmasked tokens for the student output as well
    layer_results = [layer_results[i] for i in range(len(layer_results))] # expand to list
    layer_results = prepare_output(out=layer_results, modality=modality, norm=not norm_first)
    # B x num_keep+1 x D -> one (+1) stems from additional special token
    return layer_results

def freeze_module(module:nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

def unfreeze_module(module:nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def load_beit2_teacher(sd_path:str, **kwargs) -> VisionTransformerForMaskedImageModeling:
    sd = torch.load(sd_path)['model']
    for key in list(sd.keys()):
        if "cls_pt_layers" in key:
            del sd[key]

    beit2 = VisionTransformerForMaskedImageModeling(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )

    result = beit2.load_state_dict(sd)
    logger.info(f"Loaded BEiT2 teacher state dict with result: {result}")
    del beit2.lm_head
    return beit2

# from: https://github.com/microsoft/unilm/blob/master/beit2/masking_generator.py
class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta
        
        # maintain a fix number {self.num_masking_patches}
        if mask_count > self.num_masking_patches:
            delta = mask_count - self.num_masking_patches
            mask_x, mask_y = mask.nonzero()
            to_vis = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_vis], mask_y[to_vis]] = 0

        elif mask_count < self.num_masking_patches:
            delta = self.num_masking_patches - mask_count
            mask_x, mask_y = (mask == 0).nonzero()
            to_mask = np.random.choice(mask_x.shape[0], delta, replace=False)
            mask[mask_x[to_mask], mask_y[to_mask]] = 1

        assert mask.sum() == self.num_masking_patches, f"mask: {mask}, mask count {mask.sum()}"

        return mask