from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
import torch
import logging
from collections import namedtuple
import os
import logging
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict

import sys
sys.path.append('fairseq/')
from examples.data2vec.models.data2vec2 import Data2VecMultiModel
from examples.data2vec.models.data2vec2 import Data2VecMultiConfig
from examples.data2vec.data.modality import Modality
from fairseq.dataclass.utils import merge_with_parent
from fairseq.data import Dictionary

logger = logging.getLogger(__name__)

def build_config(conf_path:str=None):
    cli_conf = OmegaConf.from_cli()
    if conf_path is None:
        if "config" not in cli_conf:
            raise ValueError(
                "Please pass 'config' to specify configuration yaml file"
            )
        yaml_conf = OmegaConf.load(cli_conf.config)
        cli_conf.pop("config")
    else:
        yaml_conf = OmegaConf.load(conf_path)
    conf = instantiate(yaml_conf)
    config = OmegaConf.merge(conf, cli_conf)
    return config


def load_model(pretrained_model_cfg:DictConfig,
               model_state_dict:OrderedDict[str, torch.Tensor]) -> Data2VecMultiModel:
    
    pretrained_model_cfg = merge_with_parent(Data2VecMultiConfig(), pretrained_model_cfg, remove_missing=True)

    logger.info(f"Modality used: {pretrained_model_cfg.supported_modality}")
    if pretrained_model_cfg.supported_modality == Modality.TEXT:
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
