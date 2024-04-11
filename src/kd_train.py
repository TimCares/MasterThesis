import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import os
import logging
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from multimodal_data2vec import KDMMData2VecConfig, KDData2VecPreTrainingLightningModule
from datamodules import REGISTRY as DATAMODULE_REGISTRY
from multi_data_loader import MultiDataModule
from validate import ZeroShotCallback

from utils import merge_with_parent


logger = logging.getLogger(__name__)

@hydra.main(config_path=os.path.join("..", "configs", "training"))
def main(cfg: DictConfig) -> None:
    cfg.model = merge_with_parent(dc=KDMMData2VecConfig(), cfg=cfg.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')

    model = KDData2VecPreTrainingLightningModule(cfg=cfg)

    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)) # only effect: evaluate/resolve all interpolations

    datamodules_keys = [name for name in cfg.data if name.startswith('_')]
    dataloader_keys = [name for name in cfg.data if not name.startswith('_')]
    
    # resolving before is needed here to select a subset
    # unresolved field in the selected keys can't be resolved after this selection
    dataloader_args = {key: cfg.data[key] for key in dataloader_keys}

    datamodules = []
    for name in datamodules_keys:
        args = OmegaConf.merge(cfg.data[name], dataloader_args)
        
        datamodules.append(DATAMODULE_REGISTRY[name[1:]](**args))
    
    datamodule = MultiDataModule(datamodules=datamodules)

    val_cfg = cfg.zero_shot_val
    zero_shot_modules = dict()
    for name in val_cfg.datamodules:
        args = OmegaConf.merge(val_cfg.datamodules[name], dataloader_args)

        zero_shot_modules[name[1:]] = DATAMODULE_REGISTRY[name[1:]](**args)

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ZeroShotCallback(n_neighbors=val_cfg.n_neighbors,
                         datamodules=zero_shot_modules,
                         data_path=val_cfg.data_path,
                         num_max_bpe_tokens=val_cfg.num_max_bpe_tokens,
                         is_multimodal_aligned=val_cfg.is_multimodal_aligned,),
        ModelCheckpoint(
                **OmegaConf.to_container(cfg.checkpoint)
            )
    ]

    logger.info("Setting up datamodules:")
    #datamodule.setup("fit")
    logger.info("Datamodule setup complete.")

    trainer = Trainer(
        **OmegaConf.to_container(cfg.lightning_trainer),
        callbacks=callbacks,
    )

    if 'load_checkpoint' in cfg and cfg.load_checkpoint is not None:
        ckpt_path = os.path.join(cfg.model_path, cfg.load_checkpoint)
    else:
        ckpt_path = None

    # trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    # trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
