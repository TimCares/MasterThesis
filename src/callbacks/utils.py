import logging
from pytorch_lightning import Callback, Trainer, LightningModule
from typing import *
from time import time

logger = logging.getLogger(__name__)

class WallClockCallback(Callback):
    def __init__(self):
        self.gpu_wall_clock_time = 0.0
        self.start_time = None

    def on_train_batch_start(self, trainer:Trainer, pl_module:LightningModule, batch:Any, batch_idx:int) -> None:
        self.start_time = time()

    def on_train_batch_end(self, trainer:Trainer, pl_module:LightningModule,
                           outputs, batch: Any, batch_idx: int) -> None:
        elapsed_time = time() - self.start_time
        self.gpu_wall_clock_time += elapsed_time

    def on_train_end(self, trainer:Trainer, pl_module:LightningModule) -> None:
        logger.info(f"GPU Wall Clock Time: {self.gpu_wall_clock_time/60:.2f} minutes")
        logger.info(f"GPU Wall Clock Time: {self.gpu_wall_clock_time/3600:.2f} hours")
        trainer.logger.experiment.log({"gpu_wall_clock_time": self.gpu_wall_clock_time})
