from typing import List
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader 
from torch.utils.data.dataset import ConcatDataset
import torch
from data2vec_fairseq.data.modality import Modality

class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        datamodules: List[LightningDataModule],
        batch_size:int,
        num_workers:int,
        shuffle:bool=True,
        drop_last:bool=True,
        modality:Modality=Modality.VL,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.modality = modality

    def prepare_data(self):
        for datamodule in self.datamodules:
            datamodule.prepare_data()

    def setup(self, stage=None):
        train_datasets = []
        for datamodule in self.datamodules:
            datamodule.setup(stage)
            if hasattr(datamodule, 'train_dataset'):
                train_datasets.append(datamodule.train_dataset)

        self.train_dataset = ConcatDataset(train_datasets)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)
    
    def collater(self, samples):
        return self.datamodules[0].train_dataset.collater(samples)

    def teardown(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            del self.train_dataset
