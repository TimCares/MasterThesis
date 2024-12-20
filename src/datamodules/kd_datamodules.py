from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, List
from functools import partial
from datasets_ import ImageKDDataset, AudioKDDataset, TextKDDataset
import logging

logger = logging.getLogger(__name__)

class KDDataModule(LightningDataModule):
    def __init__(self,
                 data_path:str,
                 dataset:str,
                 dataset_mode:str,
                 rank:int,
                 world_size:int,
                 num_workers:int,
                 shuffle:bool,
                 drop_last:bool,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.dataset = dataset
        self.dataset_mode = dataset_mode
        self.rank = rank
        self.world_size = world_size
        self.batch_size = 1
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prepared = False

        logger.info(f"Using dataset: {self.dataset}")
    
    def set_train_dataset(self):
        if self.dataset_mode == 'image':
            dataset_cls = ImageKDDataset
        elif self.dataset_mode == 'audio':
            dataset_cls = AudioKDDataset
        elif self.dataset_mode == 'text':
            dataset_cls = TextKDDataset
        else:
            raise ValueError(f"Unknown dataset mode: {self.dataset_mode}")
        self.train_dataset = dataset_cls(data_path=self.data_path, dataset=self.dataset,
                                         rank=self.rank, world_size=self.world_size)

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self.train_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)


KD_DATAMODULE_REGISTRY = {
    'kd_image_datamodule': partial(KDDataModule, dataset_mode='image'),
    'kd_audio_datamodule': partial(KDDataModule, dataset_mode='audio'),
    'kd_text_datamodule': partial(KDDataModule, dataset_mode='text'),
}
