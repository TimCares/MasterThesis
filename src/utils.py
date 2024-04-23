from omegaconf import OmegaConf
import os
import torch
import logging
from collections import namedtuple
import os
import logging
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict
from dataclasses import is_dataclass
from typing import List, Tuple

from data2vec_fairseq.models.data2vec2 import Data2VecMultiModel
from data2vec_fairseq.models.data2vec2 import Data2VecMultiConfig
from fairseq.data import Dictionary

logger = logging.getLogger(__name__)

# source: https://github.com/facebookresearch/fairseq/blob/bedb259bf34a9fc22073c13a1cee23192fa70ef3/fairseq/dataclass/utils.py#L487
def merge_with_parent(dc, cfg: DictConfig, remove_missing=False):
    if remove_missing:

        def remove_missing_rec(src_keys, target_cfg):
            if is_dataclass(target_cfg):
                target_keys = set(target_cfg.__dataclass_fields__.keys())
            else:
                target_keys = set(target_cfg.keys())

            for k in list(src_keys.keys()):
                if k not in target_keys:
                    del src_keys[k]
                elif OmegaConf.is_config(src_keys[k]):
                    tgt = getattr(target_cfg, k)
                    if tgt is not None and (is_dataclass(tgt) or hasattr(tgt, "keys")):
                        remove_missing_rec(src_keys[k], tgt)

        with open_dict(cfg):
            remove_missing_rec(cfg, dc)

    merged_cfg = OmegaConf.merge(dc, cfg)
    merged_cfg.__dict__["_parent"] = cfg.__dict__["_parent"]
    OmegaConf.set_struct(merged_cfg, True)
    return merged_cfg


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


def load_pretrained_d2v_model(state_dict_path:str):
    model_meta_data = torch.load(state_dict_path)
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])
    model = load_model(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])

    # removes decoder, and all encoders with modality != supported modality
    model.remove_pretraining_modules(modality=pretrained_model_cfg.supported_modality)

    return model


def pad_text_sequence(tokens:List[int],
                      num_max_bpe_tokens:int,
                      pad_idx:int,
                      bos_idx:int,
                      eos_idx:int=None) -> Tuple[List[int], List[int]]:
    """
    Pads a list of language tokens to num_max_bpe_tokens and inserts bos token at front.
    Also inserts eos token if provided.
    """
    
    substract = 2 if eos_idx is not None else 1
    
    if len(tokens) > num_max_bpe_tokens - substract:
        tokens = tokens[:num_max_bpe_tokens - substract]
    tokens = [bos_idx] + tokens + ([eos_idx] if eos_idx is not None else [])
    num_tokens = len(tokens)
    padding_mask = [0] * num_tokens + [1] * (num_max_bpe_tokens - num_tokens)
    language_tokens =  tokens + [pad_idx] * (num_max_bpe_tokens - num_tokens)

    return language_tokens, padding_mask