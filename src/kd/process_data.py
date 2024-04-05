import sys
sys.path.append('../fairseq/')
sys.path.append('../')
sys.path.append('../../')
from typing import Dict, Any
from collections import namedtuple
import os
import logging
import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from collections import OrderedDict
import torch
from examples.data2vec.models.data2vec2 import Data2VecMultiModel
from examples.data2vec.models.data2vec2 import Data2VecMultiConfig
from examples.data2vec.data.modality import Modality
from fairseq.dataclass.utils import merge_with_parent
from fairseq.data import Dictionary
from datamodules import REGISTRY as DATAMODULE_REGISTRY
from datamodules import BaseDataModule

logger = logging.getLogger(__name__)

def load_model(pretrained_model_cfg:DictConfig,
               model_state_dict:OrderedDict[str, torch.Tensor]) -> Data2VecMultiModel:
    
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
    logger.info(f'Loaded state dict, result: {result}')
    return model


@hydra.main(config_path=os.path.join("..", "..", "configs", "kd"), config_name="base")
def extract_targets(cfg: DictConfig) -> None:
    model_meta_data = torch.load(os.path.join('..', '..', 'models', cfg.model_state_dict))
    pretrained_model_cfg = OmegaConf.create(model_meta_data['cfg']['model'])
    model = load_model(pretrained_model_cfg=pretrained_model_cfg, model_state_dict=model_meta_data['model'])
    
    datamodule_kwargs = OmegaConf.to_container(cfg.datamodule)
    model_name = datamodule_kwargs.pop('_name', None)

    if model_name is None:
        raise ValueError('Field "_name" of cfg.datamodule either missing or is None!')
    
    datamodule_cls = DATAMODULE_REGISTRY[model_name]
    
    datamodule:BaseDataModule = datamodule_cls(**datamodule_kwargs)

    datamodule.prepare_data()

    datamodule.setup(stage='fit')

    loader = datamodule.train_dataloader()

    print(loader)



if __name__ == "__main__":
    extract_targets()
