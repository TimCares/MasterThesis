import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import os
import logging

from multimodal_data2vec import KDMMData2VecConfig, KDMMData2Vec
from fairseq.dataclass.utils import merge_with_parent

from pytorch_lightning import seed_everything, Trainer


logger = logging.getLogger(__name__)

@hydra.main(config_path=os.path.join("..", "configs"), config_name="base")
def main(cfg: DictConfig) -> None:
    cfg.training.model = merge_with_parent(dc=KDMMData2VecConfig(), cfg=cfg.training.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')
    model = KDMMData2Vec(cfg=cfg.training.model)
    logger.info(model)

if __name__ == "__main__":
    main()
