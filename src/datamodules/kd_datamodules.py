from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from src.datasets import KDOpenWebTextDataset

class KDBaseDataModule(LightningDataModule):
    def __init__(self,
                 data_path:str,
                 num_workers:int,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.batch_size = 1
        self.num_workers = num_workers
        self.shuffle = False
        self.drop_last = False
        self.prepared = False
    
    def set_train_dataset(self):
        raise NotImplementedError("set train dataset")

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


class KDOpenWebTextDataModule(KDBaseDataModule):
    def __init__(self,
                 data_path:str,
                 num_workers:int,
                 *args,
                 **kwargs):
        super().__init__(data_path, num_workers, *args, **kwargs)

    def set_train_dataset(self):
        self.train_dataset = KDOpenWebTextDataset(self.data_path)



KD_REGISTRY = {
    'kd-openwebtext': KDOpenWebTextDataModule
}