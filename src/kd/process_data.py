import sys
sys.path.append('../fairseq/')
sys.path.append('../')
sys.path.append('../../')
import json
import os
import logging
import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
from datamodules import REGISTRY as DATAMODULE_REGISTRY
from datamodules import BaseDataModule
from datasets import Modality
from rich.progress import track
from src.utils import load_pretrained_d2v_model

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("..", "..", "configs", "kd"), config_name="base")
def extract_targets(cfg: DictConfig) -> None:
    model = load_pretrained_d2v_model(state_dict_path=os.path.join('..', '..', 'models', cfg.model_state_dict))

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

    with torch.no_grad():
        for idx, batch in track(enumerate(train_dataloader), description="Running predictions...", total=len(train_dataloader)):
            key = batch['modes'][0].name.lower() if key is None else key

            padding_mask = batch['padding_mask'] if 'padding_mask' in batch else None 
            pred = model.extract_features(
                source=batch[key],
                mode=None, # determined automatically in model
                padding_mask=padding_mask,
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False,
            )
            filename = f'{idx}_{batch["id"][0]}-{batch["id"][-1]}.pt'
            index_items.append({
                "path": os.path.join(dir_name, filename),
                "batch_idx": idx, 
                "indices": batch["id"].tolist(),
            })
            item = {
                'target': pred,
                key: batch[key],
                'modes': batch['modes'],
            }
            if padding_mask is not None:
                item['padding_mask'] = padding_mask
            torch.save(item, os.path.join(kd_targets_path, filename))

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
