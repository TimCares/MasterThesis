"""
Because the data for the KD-Datasets is already stored in batches for memory efficiency and faster loading times,
each item returned by the datasets in this file is already a prepared batch. Therefore, 
"""
from .base_datasets import BaseDataset
import os
import json
import torch
import logging

logger = logging.getLogger(__name__)
class KDDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            dataset:str,):
        super().__init__(data_path, 'train') # Knowledge-Distillation is always training
        self.dataset = dataset
        available_datasets = os.listdir(self.data_path)
        prefix = 'kd_'
        prefix_len = len(prefix)
        available_kd_datasets = [ds[len(prefix):] for ds in available_datasets if ds.startswith(prefix)] # 3: -> remove "kd_"
        logger.info(f"Available kd datasets: {available_kd_datasets}")
        assert self.dataset in available_kd_datasets, f"Dataset {self.dataset} not available for KD, possible choices: {available_kd_datasets}"

        self.path_to_data = os.path.join(self.data_path, f"kd_{self.dataset}")

    def load(self):
        with open(os.path.join(self.path_to_data, 'index.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index = data['index']
        self.meta = data['datamodule']
        self.batch_size = self.meta['batch_size']
        self.items = [os.path.join(self.data_path, data_info['path']) for data_info in self.index]

    def __getitem__(self, index: int):
        return torch.load(self.items[index])
    
    def collater(self, batch):
        return batch[0] # "batch" is just a one element list with an already prepared batch, so only indexing necessary here
    