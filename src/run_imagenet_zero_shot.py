import logging
from typing import *
import torch
import os
from pytorch_lightning import LightningModule
import sys
sys.path.append('beit2')
from models import MODEL_REGISTRY
from datamodules import DATAMODULE_REGISTRY
from collections import namedtuple
from omegaconf import DictConfig
import hydra
import json
from callbacks.zero_shot import run_multimodal_zero_shot, run_filip_zero_shot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@hydra.main(version_base=None, config_path=os.path.join("..", "configs"), config_name='imagenet_zero_shot')
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    imagenet = DATAMODULE_REGISTRY['imagenet'](**cfg.data.imagenet)
    
    path = os.path.join(cfg.pretrained_path, cfg.model_version)
    model_cls:LightningModule = MODEL_REGISTRY[cfg.model_name]['module']
    model = model_cls.load_from_checkpoint(path).model
    model = model.to(device)
    model.requires_grad_(False)
    model.eval()

    PLModuleDummyWrapper = namedtuple('PLModuleDummyWrapper', 'model')
    pl_module = PLModuleDummyWrapper(model=model)

    imagenet.prepare_data()
    imagenet.setup('train')
    benchmark_func = run_filip_zero_shot if cfg.filip_zero_shot else run_multimodal_zero_shot
    zero_shot_args = {
        'pl_module': pl_module,
        'dataloader': imagenet.val_dataloader(),
        'num_max_bpe_tokens': cfg.data.num_max_bpe_tokens,
        'device': device,
        'name': 'imagenet',
    }
    with torch.no_grad():
        result_dict = benchmark_func(**zero_shot_args)
        logger.info(f'* Eval result = {json.dumps(result_dict)}')

if __name__ == "__main__":
    main()