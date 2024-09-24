import sys
sys.path.append('beit2')
import hydra
from omegaconf import OmegaConf, DictConfig, open_dict
import os
import torch
import logging
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from models.text_classification import TextClassificationConfig, TextClassificationLightningModule
from datamodules import DATAMODULE_REGISTRY
from callbacks import WallClockCallback

from fairseq.dataclass.utils import merge_with_parent


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "fine_tuning"))
def main(cfg: DictConfig) -> None:
    # first things first: set up logging
    # that way we log everything that happens during setup and do not miss anything
    wandb_logger = WandbLogger(project='MMRL',
                               name=cfg.run_name,
                               save_dir=cfg.log_dir,
                               log_model=False,)
    
    cfg.model = merge_with_parent(dc=TextClassificationConfig(), cfg=cfg.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    logger.info('Starting training.')
    module = TextClassificationLightningModule(cfg=cfg)

    OmegaConf.resolve(cfg=cfg) # resolving done in-place

    callbacks = [
        ModelSummary(),
        LearningRateMonitor(logging_interval="step"),
        WallClockCallback(),
    ]
    if 'checkpoint' in cfg:
        callbacks.append(ModelCheckpoint(
                **OmegaConf.to_container(cfg.checkpoint, resolve=True)
        ))
        # checkpoint last, so that zero shot has been performed before saving 
        # (ModelCheckpoint usually executed last automatically, but just to be sure)

    torch.set_float32_matmul_precision("high") # or: "highest"
    trainer = Trainer(
        **OmegaConf.to_container(cfg.lightning_trainer, resolve=True),
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    dataset_args = OmegaConf.to_container(cfg.data, resolve=True)
    dataset_key = dataset_args.pop('_name')
    dataset = DATAMODULE_REGISTRY[dataset_key](**dataset_args)

    if 'load_checkpoint' in cfg and cfg.load_checkpoint is not None:
        logger.info(f'Resuming from checkpoint: {cfg.load_checkpoint}')
        ckpt_path = os.path.join(cfg.model_path, cfg.load_checkpoint)
    else:
        ckpt_path = None

    trainer.fit(module, datamodule=dataset, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
