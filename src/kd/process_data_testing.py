import sys
sys.path.append('../fairseq/')
sys.path.append('../')
sys.path.append('../../')
from typing import Dict, Any
import json
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
from rich.progress import track

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

    # removes decoder and all encoders with modality != supported modality
    model.remove_pretraining_modules(modality=pretrained_model_cfg.supported_modality)
    model.eval()
    
    datamodule_kwargs = OmegaConf.to_container(cfg.datamodule)
    datamodule_name = datamodule_kwargs.pop('_name', None)

    if datamodule_name is None:
        raise ValueError('Field "_name" of cfg.datamodule either missing or is None!')
    
    datamodule_cls = DATAMODULE_REGISTRY[datamodule_name]
    
    datamodule:BaseDataModule = datamodule_cls(**datamodule_kwargs)

    logger.info('Setting up dataset and dataloader...')

    datamodule.prepare_data()

    datamodule.setup(stage='fit')

    train_dataloader = datamodule.train_dataloader()

    dir_name = f'kd_{datamodule_name}'

    kd_targets_path = os.path.join(cfg.datamodule.data_path, dir_name)

    os.makedirs(kd_targets_path, exist_ok=True)

    index_items = []
    offset=0
    with torch.no_grad():
        for idx, batch in track(enumerate(train_dataloader), description="Running predictions...", total=len(train_dataloader)):
            pred=torch.arange(start=offset, end=offset+batch['audio'].shape[0]-1)
            offset+=batch['audio'].shape[0]
            # pred = model.extract_features(
            #     source=batch['source'],
            #     mode=None, # determined automatically in model
            #     padding_mask=batch['padding_mask'],
            #     mask=False, # we are creating targets from a teacher model for the student model, so no mask
            #     remove_extra_tokens=False,
            # )
            filename = f'{idx}_{pred[0]}-{pred[-1]}.pt'
            index_items.append({
                "path": os.path.join(dir_name, filename),
                "batch_idx": idx, 
                "indices": pred.tolist(),
            })

            torch.save(pred, os.path.join(kd_targets_path, filename))

    index = {
        'datamodule': OmegaConf.to_container(cfg.datamodule),
        'model_state_dict': cfg.model_state_dict,
        'index': index_items,
    }

    with open(os.path.join(kd_targets_path, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f)

    logger.info(f'Knowledge-Distillation (KD) targets successfully created under: {kd_targets_path}')


if __name__ == "__main__":
    extract_targets()
