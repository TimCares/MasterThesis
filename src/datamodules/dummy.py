import torch
from torch.utils.data import Dataset
from .unimodal_datamodules import BaseDataModule
from data2vec_fairseq.data.modality import Modality

class DummyDataset(Dataset):
    def __init__(self, size=50000, dim=20):
        """
        Args:
            size (int): Number of data points in the dataset.
            dim (int): Dimensionality of each data point.
        """
        self.size = size
        self.dim = dim

    def load(self):
        pass

    def collater(self, batch):
        # we use text as an example here
        data = {
            'text': torch.stack([batch[i]['input'] for i in range(len(batch))]),
            'padding_mask': torch.ones(len(batch), 512).float(),
            'modes': [Modality.TEXT],
            'id': torch.arange(len(batch)),
            'target': torch.stack([batch[i]['target'] for i in range(len(batch))]),
        }
        return data

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Retrieve the data point at the specified index
        return {'input': torch.randn(512, self.dim), 'target': torch.randn(self.dim)}


class DummyDataModule(BaseDataModule):
    def __init__(self):
        super().__init__(data_path="",
                        batch_size=64,
                        num_workers=1,
                        shuffle=False,
                        drop_last=False,)
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset =  DummyDataset(size=10000, dim=20)

DUMMY_DATAMODULE_REGISTRY = {
    'dummy': DummyDataModule
}