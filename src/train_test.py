import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import os
import logging

from multimodal_data2vec import KDMMData2VecConfig,  KDMMData2Vec
from fairseq.dataclass.utils import merge_with_parent

from pytorch_lightning import seed_everything, Trainer


logger = logging.getLogger(__name__)

@hydra.main(config_path=os.path.join("..", "configs", "training"))
def main(cfg: DictConfig) -> None:
    cfg.model = merge_with_parent(dc=KDMMData2VecConfig(), cfg=cfg.model, remove_missing=False)
    if 'seed' in cfg and cfg.seed is not None:
        seed_everything(seed=cfg.seed, workers=True)
    else:
        logger.info('No seed set.')
    
    if not cfg.dry_run:
        model = KDMMData2Vec(cfg=cfg.model)
    else:
        model = None # TODO



if __name__ == "__main__":
    main()
