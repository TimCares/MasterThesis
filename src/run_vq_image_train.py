import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
import os
import torch
import logging
import git
from datetime import datetime
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.loggers import WandbLogger
import sys
sys.path.append("beit2")
from models import MODEL_REGISTRY
from datamodules import DATAMODULE_REGISTRY
from callbacks import WallClockCallback, GracefulStoppingCallback, ResumeCheckModelCheckpoint, CodebookUsageCallback
from fairseq.dataclass.utils import merge_with_parent


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "training"))
def main(cfg: DictConfig) -> None:
    if cfg.id is None:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha[:7]
        now = datetime.now()
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        id = f"{sha}__{timestamp}"
    else:
        assert cfg.load_checkpoint is not None, "If id is set, load_checkpoint must be set as well."
        id = cfg.id

    wandb_logger = WandbLogger(
        project='MMRL',
        name=cfg.run_name,
        save_dir=cfg.log_dir,
        log_model=False,
        id=id,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=True,
        resume='allow',
    )
    
    cfg_cls = MODEL_REGISTRY[cfg.model_name]['cfg']
    module_cls = MODEL_REGISTRY[cfg.model_name]['module']
    
    cfg.model = merge_with_parent(dc=cfg_cls(), cfg=cfg.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    logger.info('Starting training.')
    module = module_cls(cfg)

    OmegaConf.resolve(cfg=cfg) # resolving done in-place

    callbacks = [
        GracefulStoppingCallback(ckpt_path=os.path.join(cfg.checkpoint.common.dirpath, 'last.ckpt')),
        ModelSummary(),
        LearningRateMonitor(logging_interval="step"),
        WallClockCallback(),
        CodebookUsageCallback(n_codebook_embed=cfg.model.n_codebook_embed),
    ]
    
    common_checkpoint_args = OmegaConf.to_container(cfg.checkpoint.common, resolve=True)
    for ckpt in cfg.checkpoint.checkpoints:
        args = OmegaConf.to_container(ckpt, resolve=True) | common_checkpoint_args
        callbacks.append(ResumeCheckModelCheckpoint(**args))

    torch.set_float32_matmul_precision("high") # or: "highest"
    trainer_args = OmegaConf.to_container(cfg.lightning_trainer, resolve=True)
    if 'strategy' not in trainer_args:
        if 'deepspeed' in trainer_args:
            from pytorch_lightning.strategies import DeepSpeedStrategy
            trainer_args['strategy'] = DeepSpeedStrategy(**trainer_args.pop('deepspeed'))
        elif 'ddp' in trainer_args:
            from pytorch_lightning.strategies import DDPStrategy
            trainer_args['strategy'] = DDPStrategy(**trainer_args.pop('ddp'))
        else:
            trainer_args['strategy'] = 'auto'
        
    trainer = Trainer(
        **trainer_args,
        enable_checkpointing=True,
        callbacks=callbacks,
        logger=wandb_logger,
    )
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.save(f'models/{cfg.model_name}.py') # saves the model file to wandb

    datamodule_args = OmegaConf.to_container(cfg.data, resolve=True)
    name = datamodule_args.pop('_name')
    datamodule = DATAMODULE_REGISTRY[name](**datamodule_args)
    logger.info(f"Datamodule {name}: {datamodule_args}")

    logger.info("Setting up datamodules:")
    datamodule.prepare_data()
    datamodule.setup("fit")
    logger.info("Datamodule setup complete.")

    if 'load_checkpoint' in cfg and cfg.load_checkpoint is not None:
        logger.info(f'Resuming from checkpoint: {cfg.load_checkpoint}')
        ckpt_path = cfg.load_checkpoint
    else:
        ckpt_path = None

    trainer.fit(module,
                datamodule=datamodule,
                ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
