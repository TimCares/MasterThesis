"""
Because the data for the KD-Datasets is already stored in batches for memory efficiency and faster loading times,
each item returned by the datasets in this file is already a prepared batch. For image and audio, the samples need
to be loaded from disk and prepared in a seperate collater function.
"""
from .base_datasets import BaseDataset, AudioDataset
from torchvision.datasets.folder import default_loader
from .data_utils import get_transforms
import os
import json
import torch
import soundfile as sf
import logging

logger = logging.getLogger(__name__)
class KDDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            dataset:str,
            rank:int,
            world_size:int,):
        super().__init__(data_path, 'train') # Knowledge-Distillation is always training
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        available_datasets = os.listdir(self.data_path)
        prefix = 'kd_'
        prefix_len = len(prefix)
        available_kd_datasets = [ds[prefix_len:] for ds in available_datasets if ds.startswith(prefix)]
        assert self.dataset in available_kd_datasets, f"Dataset {self.dataset} not available for KD, possible choices: {available_kd_datasets}"

        self.path_to_data = os.path.join(self.data_path, f"kd_{self.dataset}")

    def load(self):
        with open(os.path.join(self.path_to_data, 'index.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index = data['index']
        self.meta = data['datamodule']
        self.batch_size = self.meta['batch_size']
        all_items = [os.path.join(self.data_path, data_info['path']) for data_info in self.index]
        self.items = all_items[self.rank::self.world_size] # split data across ranks

    def __getitem__(self, index: int):
        return torch.load(self.items[index])
    
    def collater(self, batch):
        return batch[0] # "batch" is just a one element list with an already prepared batch, so only indexing necessary here
    

class TextKDDataset(KDDataset):
    pass # no need to implement anything, just use the base class (kept for consistency)


class ImageKDDataset(KDDataset):
    def load(self):
        super().load()
        self.transform = get_transforms(no_transform=self.meta['no_transform'],
                                        beit_transforms=self.meta['beit_transforms'], 
                                        transform_jitter=self.meta['transform_jitter'])
        
        self.loader = default_loader
        
    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)

    def __getitem__(self, index: int):
        batch = super().__getitem__(index)

        image_paths = batch.pop('data_path')
        images = [self._get_image(image_path) for image_path in image_paths]
        batch['image'] = torch.stack(images)
        return batch


class AudioKDDataset(KDDataset):
    def load(self):
        super().load()

        self._dataset = AudioDataset( # only created for the collater, TODO: maybe refactor later
            data_path=self.meta['data_path'],
            split='train',
            sample_rate=self.meta['sample_rate'],
            max_sample_size=self.meta['max_sample_size'],
            min_sample_size=self.meta['min_sample_size'],
            normalize=self.meta['normalize'],
            pad=self.meta['pad'],
        )

    def _get_audio(self, audio_path: str):
        wav, curr_sample_rate = sf.read(audio_path, dtype="float32")
        assert curr_sample_rate == self.meta['sample_rate']
        feats = torch.from_numpy(wav).float()
        return self._dataset.postprocess(feats, curr_sample_rate)

    def __getitem__(self, index: int):
        batch = super().__getitem__(index)

        audio_paths = batch.pop('data_path')

        audios = [self._get_audio(audio_path) for audio_path in audio_paths]

        batch['audio'] = self._dataset.collater(audios)
        return batch


KD_DATASET_REGISTRY = {
    'kd_image_dataset': ImageKDDataset,
    'kd_audio_dataset': AudioKDDataset,
    'kd_text_dataset': TextKDDataset,
}