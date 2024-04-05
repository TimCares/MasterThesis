import sys
sys.path.append('../fairseq/')
sys.path.append('../')
import torch
from typing import Dict, Any
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict
from examples.data2vec.models.data2vec2 import Data2VecMultiModel
from examples.data2vec.models.data2vec2 import Data2VecMultiConfig
from examples.data2vec.data.modality import Modality
from fairseq.dataclass.utils import merge_with_parent
from collections import namedtuple
from fairseq.data import Dictionary
import os
import logging
from utils import hydra_utils

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

def extract_targets(pretrained_model_cfg:DictConfig,
                    model_state_dict:OrderedDict[str, torch.Tensor]):
    
    pretrained_model_cfg = merge_with_parent(Data2VecMultiConfig(), pretrained_model_cfg, remove_missing=True)

    logger.info(f"Modality used: {pretrained_model_cfg.supported_modality}")
    if pretrained_model_cfg.supported_modality == Modality.TEXT:
        Task = namedtuple('Task', 'source_dictionary')

        dictionary = Dictionary.load(os.path.join('..', '..', 'data', "dict.txt"))
        dictionary.add_symbol("<mask>")
        dummy_task = Task(source_dictionary=dictionary)
    else:
        dummy_task = False

    model = Data2VecMultiModel.build_model(pretrained_model_cfg, task=dummy_task)

    result = model.load_state_dict(model_state_dict)
    logger.info(f'Loaded state dict, {result}')
    

    
    

if __name__ == "__main__":
    cfg = hydra_utils.build_config()
    model_meta_data = torch.load(os.path.join('..', '..', 'models', cfg.model_state_dict))
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])

    extract_targets(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])
