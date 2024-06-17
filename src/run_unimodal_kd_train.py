import hydra
import wandb
from omegaconf import OmegaConf, open_dict, DictConfig
import os
import torch
import logging
import re
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger
from models.kd_data2vec import KDData2VecConfig, KDData2VecPreTrainingLightningModule
from datamodules import DATAMODULE_REGISTRY
from callbacks import ZeroShotRetrievalCallback, WallClockCallback

from fairseq.dataclass.utils import merge_with_parent


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "training"))
def main(cfg: DictConfig) -> None:
    # first things first: set up logging
    # that way we log everything that happens during setup and do not miss anything
    wandb_logger = WandbLogger(project='MMRL',
                               name=cfg.run_name,
                               save_dir=cfg.log_dir,
                               log_model=False,)
    
    cfg.model = merge_with_parent(dc=KDData2VecConfig(), cfg=cfg.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    logger.info('Starting training.')
    module = KDData2VecPreTrainingLightningModule(cfg=cfg)

    logger.info(f"Running with modalities: {cfg.model.supported_modalities}")

    OmegaConf.resolve(cfg=cfg) # resolving done in-place

    if 'zero_shot_val' in cfg:
        val_cfg = cfg.zero_shot_val
        zero_shot_modules = dict()
        val_dataloader_args = val_cfg.dataloader
        for name in val_cfg.datamodules:
            with open_dict(val_dataloader_args):
                datamodule_cfg = val_cfg.datamodules[name]
                # override general dataloader args with dataloader specific args (if present)
                args = OmegaConf.merge(val_dataloader_args, datamodule_cfg)

            zero_shot_modules[name] = DATAMODULE_REGISTRY[name](**args)
            logger.info(f"Zero-shot datamodule {name}: {args}")

    dataloader_args = cfg.data.dataloader

    callbacks = [
        ModelSummary(),
        LearningRateMonitor(logging_interval="step"),
        WallClockCallback(), # before zero-shot, so that we measure only the training batch time
    ]
    if 'zero_shot_val' in cfg:
        callbacks.append(
            ZeroShotRetrievalCallback(
                datamodules=zero_shot_modules,
                val_every_n_batches=val_cfg.val_every_n_batches,),
        )
    # checkpoint last, so that zero shot has been performed before saving 
    # (ModelCheckpoint usually executed last automatically, but just to be sure)
    callbacks.append(
        ModelCheckpoint(
                **OmegaConf.to_container(cfg.checkpoint, resolve=True)
            )
    )        

    torch.set_float32_matmul_precision("high") # or: "highest"
    trainer_args = OmegaConf.to_container(cfg.lightning_trainer, resolve=True)
    if 'deepspeed' in trainer_args['strategy']:
        trainer_args['strategy'] = DeepSpeedStrategy(
            stage=int(re.search(r'\d', trainer_args['strategy']).group()),
            offload_optimizer='offload' in trainer_args['strategy'],
            allgather_bucket_size=5e8, # size as recommended by pytorch lightning deepspeed docs
            reduce_bucket_size=5e8, # size as recommended by pytorch lightning deepspeed docs
        )
    trainer = Trainer(
        **trainer_args,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    wandb.save('models/kd_data2vec.py') # saves the model file to wandb
    wandb.save('run_unimodal_kd_train.py') # saves the model file to wandb

    datamodules = []
    for datamodule_key in cfg.data.datamodules.keys():
        dataset_args = cfg.data.datamodules[datamodule_key]
        with open_dict(dataset_args):
            dataset_args.update(dataloader_args)
        datamodules.append(DATAMODULE_REGISTRY[datamodule_key](**dataset_args))
        logger.info(f"Train datamodule {datamodule_key}: {dataset_args}")
    
    assert len(datamodules) == 1
    datamodule = datamodules[0]

    logger.info("Setting up datamodules:")
    datamodule.prepare_data()
    datamodule.setup("fit")
    logger.info("Datamodule setup complete.")

    if 'load_checkpoint' in cfg and cfg.load_checkpoint is not None:
        logger.info(f'Resuming from checkpoint: {cfg.load_checkpoint}')
        ckpt_path = os.path.join(cfg.model_path, cfg.load_checkpoint)
    else:
        ckpt_path = None

    trainer.fit(module,
                datamodule=datamodule,
                ckpt_path=ckpt_path)
    
    model_path = os.path.join(cfg.checkpoint.dirpath, 'last.ckpt')
    if 'deepspeed' in cfg.lightning_trainer.strategy and os.path.isdir(model_path):
        from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
        output_path = os.path.join(cfg.checkpoint.dirpath, 'fp32_last.ckpt')
        convert_zero_checkpoint_to_fp32_state_dict(model_path, output_path)
        os.remove(model_path)


if __name__ == "__main__":
    main()
