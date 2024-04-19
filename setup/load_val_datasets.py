import hydra
from omegaconf import OmegaConf, open_dict, DictConfig
import os
import logging

from torchaudio.datasets import LIBRISPEECH, SPEECHCOMMANDS
from torchvision.datasets import CIFAR10, CIFAR100

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=os.path.join("..", "configs", "training"))
def main(cfg:DictConfig) -> None:
    data_path = cfg.data_path

    for url in ["test-other", "train-clean-100", "train-clean-360"]:
        LIBRISPEECH(root=data_path, url=url, download=True) # no val set, but simple to just download here

    for subset in ["training", "testing"]:
        SPEECHCOMMANDS(data_path, subset=subset, download=True)

    for subset in ["train", "test"]:
        CIFAR10(data_path, train=subset, download=True)
        CIFAR100(data_path, train=subset, download=True)

        # IMDB(root=imdb_path, split=subset) -> done during setup of datamodules -> fits better there

if __name__ == "__main__":
    main()