import logging
from pytorch_lightning import Callback, LightningDataModule
from typing import *
import torch
from pytorch_lightning.utilities import rank_zero_only
from run_image_text_retrieval import zero_shot_retrieval

logger = logging.getLogger(__name__)

class RetrievalCallback(Callback):
    def __init__(self, datamodule:LightningDataModule, name:str):
        super().__init__()
        self.datamodule = datamodule
        self.name = name

    @torch.no_grad()
    @rank_zero_only
    def on_validation_start(self, trainer, pl_module, **kwargs) -> None:
        result = zero_shot_retrieval(pl_module.module.model, self.datamodule.val_dataloader(), pl_module.device)

        pl_module.log(
            f"val/{self.name}",
            result['average_score'],
            rank_zero_only=True,
            logger=True,
            on_epoch=True,
        )
