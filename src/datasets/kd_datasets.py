"""
Because the data for the KD-Datasets is already stored in batches for memory efficiency and faster loading times,
each item returned by the datasets in this file is already a prepared batch. Therefore, 
"""
from .base_datasets import BaseDataset
import os
import json
import torch

class KDDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,):
        super().__init__(data_path, 'train') # Knowledge-Distillation is always training

    def __getitem__(self, index: int):
        return torch.load(self.items[index])
    
    def collater(self, batch):
        return batch # "batch" is already a prepared batch, so no work necessary here

class KDOpenWebTextDataset(KDDataset):
    def __init__(
            self,
            data_path:str,):
        super().__init__(data_path)
        self.path_to_data = os.path.join(data_path, 'kd_common_voice')

    def load(self):
        with open(os.path.join(self.path_to_data, 'index.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index = data['index']
        self.meta = data['datamodule']
        self.batch_size = self.meta['batch_size']
        self.items = [os.path.join(self.data_path, data_info['path']) for data_info in self.index]
    
    