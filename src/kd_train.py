import hydra
from omegaconf import OmegaConf, open_dict, DictConfig
import os
import logging
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from multimodal_data2vec import KDMMData2VecConfig, KDData2VecPreTrainingLightningModule, TestLightningModule
from datamodules import DATAMODULE_REGISTRY
from multi_data_loader import MultiDataModule
from validate import ZeroShotCallback

from utils import merge_with_parent


logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "training"))
def main(cfg: DictConfig) -> None:
    cfg.model = merge_with_parent(dc=KDMMData2VecConfig(), cfg=cfg.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    if cfg.dry_run is not None and cfg.dry_run:
        logger.info('Starting dry run.')
        module = TestLightningModule(cfg=cfg)
    else:
        logger.info('Starting training.')
        module = KDData2VecPreTrainingLightningModule(cfg=cfg)

    OmegaConf.resolve(cfg=cfg) # resolving done in-place

    datamodules_key = [name for name in cfg.data if name.startswith('_')][0] # only ever one "_kd_datamodules" key
    dataloader_keys = [name for name in cfg.data if not name.startswith('_')]
    
    # resolving before is needed here to select a subset
    # unresolved field in the selected keys can't be resolved after this selection
    dataloader_args = {key: cfg.data[key] for key in dataloader_keys}

    datamodules = []
    if cfg.dry_run is not None and cfg.dry_run:
        datamodules.append(DATAMODULE_REGISTRY['dummy']())
    else:
        for dataset_args in cfg.data[datamodules_key]:
            with open_dict(dataset_args):
                args = OmegaConf.merge(dataset_args, dataloader_args)
            
            datamodules.append(DATAMODULE_REGISTRY[datamodules_key[1:]](**args))
            logger.info(args)
    
    datamodule = MultiDataModule(datamodules=datamodules)

    val_cfg = cfg.zero_shot_val
    zero_shot_modules = dict()
    for name in val_cfg.datamodules:
        with open_dict(val_cfg.datamodules[name]):
            args = OmegaConf.merge(val_cfg.datamodules[name], dataloader_args)

        zero_shot_modules[name[1:]] = DATAMODULE_REGISTRY[name[1:]](**args)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ZeroShotCallback(n_neighbors=val_cfg.n_neighbors,
                         datamodules=zero_shot_modules,
                         data_path=val_cfg.data_path,
                         val_every_n_batches=val_cfg.val_every_n_batches,
                         num_max_bpe_tokens=val_cfg.num_max_bpe_tokens,
                         is_multimodal_aligned=val_cfg.is_multimodal_aligned,),
        ModelCheckpoint(
                **OmegaConf.to_container(cfg.checkpoint, resolve=True)
            )
        # checkpoint last, so that zero shot has been performed before saving 
        # (ModelCheckpoint usually executed last automatically, but just to be sure)
    ]

    logger.info("Setting up datamodules:")
    datamodule.prepare_data()
    datamodule.setup("fit")
    logger.info("Datamodule setup complete.")

    trainer = Trainer(
        **OmegaConf.to_container(cfg.lightning_trainer, resolve=True),
        enable_checkpointing=True,
        callbacks=callbacks,
    )

    if 'load_checkpoint' in cfg and cfg.load_checkpoint is not None:
        ckpt_path = os.path.join(cfg.model_path, cfg.load_checkpoint)
    else:
        ckpt_path = None

    trainer.fit(module, train_dataloaders=datamodule.train_dataloader(), ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
