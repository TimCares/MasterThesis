import logging
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from typing import *
from time import time

logger = logging.getLogger(__name__)

class WallClockCallback(Callback):
    def __init__(self):
        self.gpu_wall_clock_time = 0.0
        self.start_time = None
        self.n_batches = 0

    def on_train_batch_start(self, trainer:Trainer, pl_module:LightningModule, batch:Any, batch_idx:int) -> None:
        self.start_time = time()

    def on_train_batch_end(self, trainer:Trainer, pl_module:LightningModule,
                           outputs, batch: Any, batch_idx: int) -> None:
        elapsed_time = time() - self.start_time
        self.gpu_wall_clock_time += elapsed_time
        self.n_batches += 1

    def on_train_end(self, trainer:Trainer, pl_module:LightningModule) -> None:
        logger.info(f"GPU Wall Clock Time: {self.gpu_wall_clock_time/60:.2f} minutes")
        logger.info(f"GPU Wall Clock Time: {self.gpu_wall_clock_time/3600:.2f} hours")
        trainer.logger.experiment.log({"gpu_wall_clock_time": self.gpu_wall_clock_time})

        logger.info(f"Average GPU Wall Clock Time per batch: {self.gpu_wall_clock_time/self.n_batches:.2f} seconds")
        trainer.logger.experiment.log({"gpu_wall_clock_time_per_batch": self.gpu_wall_clock_time/self.n_batches})

    @property
    def state_key(self) -> str:
        return f"WallClock[gpu_wall_clock_time={self.gpu_wall_clock_time}, n_batches={self.n_batches}]"

    def load_state_dict(self, state_dict):
        self.gpu_wall_clock_time = state_dict["gpu_wall_clock_time"]
        self.n_batches = state_dict["n_batches"]

    def state_dict(self):
        sd = {
            "gpu_wall_clock_time": self.gpu_wall_clock_time,
            "n_batches": self.n_batches
        }
        return sd
    

class GracefulStoppingCallback(Callback):
    def __init__(self, ckpt_path:str):
        self.ckpt_path = ckpt_path

    def on_train_batch_start(self, trainer:Trainer, pl_module:LightningModule, batch:Any, batch_idx:int) -> None:
        if trainer.received_sigterm:
            logger.info("Received SIGTERM. Gracefully stopping and saving checkpoint...")
            trainer.save_checkpoint(filepath=self.ckpt_path)
            trainer.should_stop = True
            logger.info("Checkpoint saved.")


class ResumeCheckModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_resumed = False

    def check_if_just_resumed(self):
        if self.has_resumed:
            logger.info("Resumed from checkpoint. Skipping repeated saving.")
            self.has_resumed = False
            return True
        return False # else

    def on_validation_end(self, trainer, pl_module):
        if self.check_if_just_resumed():
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.check_if_just_resumed():
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        self.has_resumed = trainer.ckpt_path is not None
