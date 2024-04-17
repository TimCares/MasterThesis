from typing import List
import json
import os
import logging
import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn.functional as F
from datamodules import DATAMODULE_REGISTRY
from datamodules import BaseDataModule
from rich.progress import track
from utils import load_pretrained_d2v_model

logger = logging.getLogger(__name__)

def special_token_and_average(target_layer_results:List[torch.Tensor]) -> torch.Tensor:
    target_layer_results = [
        F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
        for tl in target_layer_results  # BTC -> BCT -> BTC
    ]

    # clone() -> only the first time step is actually retained, so we not just change the view: https://pytorch.org/docs/stable/notes/serialization.html#saving-loading-tensors
    y = target_layer_results[0][:, 0, :].clone().float()
    for tl in target_layer_results[1:]:
        y.add_(tl[:, 0, :].clone().float())
    y = y.div_(len(target_layer_results))
    return y.squeeze(1) # BTC -> BC

def average_twice(target_layer_results:List[torch.Tensor]) -> torch.Tensor:
    target_layer_results = [
        F.instance_norm(tl.transpose(1, 2).float()).transpose(1, 2)
        for tl in target_layer_results  # BTC -> BCT -> BTC
    ]

    y = target_layer_results[0].float()
    for tl in target_layer_results[1:]:
        y.add_(tl.float())
    y = y.div_(len(target_layer_results))
    y = y.mean(dim=1) # BTC -> BC
    return y


@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "kd"), config_name="base")
def extract_targets(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Running with device: {device}')

    datamodule_kwargs = OmegaConf.to_container(cfg.datamodule)
    datamodule_name = datamodule_kwargs.pop('_name', None)
    model_state_dict_name = datamodule_kwargs.pop('model_state_dict', None)

    model = load_pretrained_d2v_model(state_dict_path=os.path.join('..', 'models', model_state_dict_name))
    model = model.to(device)
    model.eval()
    
    if datamodule_name is None:
        raise ValueError('Field "_name" of cfg.datamodule either missing or is None!')
    
    datamodule_cls = DATAMODULE_REGISTRY[datamodule_name]
    
    datamodule:BaseDataModule = datamodule_cls(**datamodule_kwargs)

    logger.info('Setting up dataset and dataloader...')

    datamodule.prepare_data()

    datamodule.setup(stage='fit')

    train_dataloader = datamodule.train_dataloader()

    if cfg.average_twice:
        dir_name = f'kd_{datamodule_name}_at'
    else:
        dir_name = f'kd_{datamodule_name}'

    kd_targets_path = os.path.join(cfg.out_path, dir_name)

    os.makedirs(kd_targets_path, exist_ok=True)

    batch_idx_offset = cfg.offset
    id_offset = batch_idx_offset*cfg.datamodule.batch_size

    index_items = []
    key = None
    with torch.no_grad():
        for idx, batch in track(enumerate(train_dataloader), description="Running predictions...", total=len(train_dataloader)):
            key = batch['modes'][0].name.lower() if key is None else key

            padding_mask = batch['padding_mask'] if 'padding_mask' in batch else None 
            pred = model.extract_features(
                source=batch[key].to(device),
                mode=None, # determined automatically in model
                padding_mask=padding_mask.to(device),
                mask=False, # we are creating targets from a teacher model for the student model, so no mask
                remove_extra_tokens=False,
            )
            idx = idx+batch_idx_offset
            filename = f'{idx}_{id_offset+batch["id"][0]}-{id_offset+batch["id"][-1]}.pt'
            index_items.append({
                "path": os.path.join(dir_name, filename),
                "batch_idx": idx,
            })

            pred.pop('x', None) # output of final layer not interesting, also, it is contained in 'layer_results' at [-1]
            pred.pop('mask', None) # is non here, as we do not mask the kd targets
            # pred in now dict with keys "padding_mask" and "layer_results"
            if cfg.average_twice:
                pred['layer_results'] = average_twice(pred['layer_results']).cpu()
                pred['padding_mask'] = torch.full((pred['padding_mask'].sum(dim=1).max().item(),), False, dtype=torch.bool)
            else:
                pred['layer_results'] = special_token_and_average(pred['layer_results']).cpu()
                pred['padding_mask'] = pred['padding_mask'][0]

            pred['padding_mask'] = pred['padding_mask'].cpu()

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
        'model_state_dict': model_state_dict_name,
        'index': index_items,
    }

    with open(os.path.join(kd_targets_path, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f)

    logger.info(f'Knowledge-Distillation (KD) targets successfully created under: {kd_targets_path}')


if __name__ == "__main__":
    extract_targets()
