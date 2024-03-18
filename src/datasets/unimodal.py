import logging
import torch
import torchtext
from torchtext.datasets import WikiText103
from torchaudio.datasets import LIBRISPEECH, LibriLightLimited
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import IPython.display as ipd
import matplotlib.pyplot as plt
import math
from config import DATA_PATH
import os
import json
import numpy as np
from fairseq.data import FairseqDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RawLibriLightDataset(Dataset):
    def __init__(self, dataset:str="1h", normalize:bool=True):
        self.data = LibriLightLimited(root=DATA_PATH, subset=dataset, download=True)
        self.normalize = normalize

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        waveform = self.data[idx][0][0] # idx, waveform, first batch element (only one element there)
        
        # after "https://arxiv.org/pdf/2006.11477.pdf#page10" Section 2, Feature Encoder
        if self.normalize:
            mean = waveform.mean()
            std = waveform.std()
            waveform = (waveform - mean) / std
        return waveform, waveform.shape[-1]

class RawLibrispeechDataset(Dataset):
    def __init__(self, dataset:str="train-clean-100", normalize:bool=True):
        self.data = LIBRISPEECH(root=DATA_PATH, url=dataset, download=True)
        self.normalize = normalize

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        waveform = self.data[idx][0][0] # idx, waveform, first batch element (only one element there)
        
        # after "https://arxiv.org/pdf/2006.11477.pdf#page10" Section 2, Feature Encoder
        if self.normalize:
            mean = waveform.mean()
            std = waveform.std()
            waveform = (waveform - mean) / std
        return waveform, waveform.shape[-1]
    
class RawPaddingCollateFn():
    def __init__(self, batch_size:int=32):
        self.batch_size = batch_size

    def __call__(self, batch):
        max_len = max(batch, key=lambda x: x[1])[1]

        padding_mask = torch.zeros(self.batch_size, max_len)

        padded_batch = []
        for idx, (x, waveform_length) in enumerate(batch):
            padding_mask[idx, waveform_length:] = 1 # belongs to padding -> 1 (true)
            padded_batch.append(torch.nn.functional.pad(x, (0, max_len - x.shape[-1]), value=-100))

        padded_batch = torch.stack(padded_batch)

        return padded_batch, padding_mask
    
def get_raw_libri_light_dataset(dataset:str="1h", batch_size:int=32, shuffle:bool=True, num_workers:int=1, normalize_waveform:bool=True):
    libri_light = RawLibriLightDataset(dataset=dataset, normalize=normalize_waveform)
    return DataLoader(libri_light, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=RawPaddingCollateFn(batch_size))

def get_raw_librispeech_dataset(dataset:str="train-clean-100", batch_size:int=32, shuffle:bool=True, num_workers:int=1, normalize_waveform:bool=True):
    librispeech = RawLibrispeechDataset(dataset=dataset, normalize=normalize_waveform)
    return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=RawPaddingCollateFn(batch_size))

class SpectrogramConsequitiveMasking():
    def __init__(self, scale:bool=False, mask_percentage:float=0.05):
        self.scale = scale
        self.mask_percentage = mask_percentage
        self.mel_spectrogram = T.MelSpectrogram()
        self.masking = T.TimeMasking(time_mask_param=100_000, iid_masks=True, p=mask_percentage)

    def __call__(self, batch):
        specs = []
        specs_masked = []
        max_time_steps = 0
        for x in batch:
            if self.scale:
                spec = (self.mel_spectrogram(x[0]) + 1e-6).log2()
            else:
                spec = self.mel_spectrogram(x[0])
            specs.append(spec)
            specs_masked.append(self.masking(spec))
            current_time_steps = spec.shape[-1]
            if current_time_steps > max_time_steps:
                max_time_steps = current_time_steps

        specs = torch.cat([torch.nn.functional.pad(w, (0, max_time_steps - w.shape[-1]), value=-1) for w in specs])
        specs_masked = torch.cat([torch.nn.functional.pad(w, (0, max_time_steps - w.shape[-1]), value=-1) for w in specs_masked])

        return specs_masked, specs
    
class ConsequitiveMasking():
    def __init__(self, mask_percentage:float=0.05):
        self.mask_percentage = mask_percentage

    def __call__(self, batch):
        waveforms = []
        waveforms_masked = []
        max_time_steps = 0
        
        for x in batch:
            waveform = x[0]
            waveforms.append(waveform)
            waveform_masked = waveform.clone()

            current_time_steps = waveform.shape[-1]
            mask_length = int(current_time_steps * self.mask_percentage)
            start = torch.randint(0, current_time_steps - mask_length, (1, )).item()
            for idx, s in enumerate(start):
                waveform_masked[idx, ..., s:s+mask_length] = 0.0
            waveforms_masked.append(waveform_masked)
            
            if current_time_steps > max_time_steps:
                max_time_steps = current_time_steps

        waveforms = torch.cat([torch.nn.functional.pad(w, (0, max_time_steps - w.shape[-1]), value=-1) for w in waveforms])
        waveforms_masked = torch.cat([torch.nn.functional.pad(w, (0, max_time_steps - w.shape[-1]), value=-1) for w in waveforms_masked])

        return waveforms_masked, waveforms

def get_librispeech_dataset(spectrogram:bool=True, scale:bool=False, dataset:str="train-clean-100", mask_percentage:float=0.05, consequitive:bool=False,
                 batch_size:int=32, shuffle:bool=True, num_workers:int=1):
    librispeech = LIBRISPEECH(root="./data", url=dataset, download=True)

    if spectrogram and consequitive:
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=SpectrogramConsequitiveMasking(scale, mask_percentage))
            
    elif spectrogram and not consequitive:
        mel_spectrogram = T.MelSpectrogram()
        max_batches = math.ceil(len(librispeech)/batch_size)
        n_patches_per_batch = torch.randint(2, 5, (max_batches,))
        i=0

        def collate_fn(batch):
            X = batch.clone()
            X = mel_spectrogram(X)
            num_time_steps = X.shape[-1]
            n_patches = n_patches_per_batch[i%max_batches].item()
            i+=1
            mask_length = int(num_time_steps * mask_percentage)
            length_per_patch = mask_length//n_patches

            for idx in enumerate(batch.shape[0]):
                start = torch.randint(0, num_time_steps - length_per_patch, (n_patches,))
                for s in start:
                    s=s.item()
                    X[idx, ..., s:s+length_per_patch] = 0
            return X, batch
        
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    elif not spectrogram and consequitive:
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=ConsequitiveMasking(mask_percentage))

    else:
        max_batches = math.ceil(len(librispeech)/batch_size)
        n_patches_per_batch = torch.randint(2, 5, (max_batches,))
        i=0

        def collate_fn(batch):
            X = batch.clone()
            num_time_steps = X.shape[-1]
            n_patches = n_patches_per_batch[i%max_batches].item()
            i+=1
            mask_length = int(num_time_steps * mask_percentage)
            length_per_patch = mask_length//n_patches
            
            for idx in enumerate(batch.shape[0]):
                start = torch.randint(0, num_time_steps - length_per_patch, (n_patches,))
                for s in start:
                    s=s.item()
                    X[idx, ..., s:s+length_per_patch] = 0
            return X, batch
        
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    



class SeperateModalitiesDataset(FairseqDataset):
    def __init__(
        self,
    ):
        FairseqDataset.__init__(self)
        image_dataset = None
        text_dataset = None
        audio_dataset = None
        self.datasets = [image_dataset, text_dataset, audio_dataset]
        self.total_length = sum([len(d) for d in self.datasets])
        self.dataset_index = 0
        self.batch_size = None
        

    def __getitem__(self, index):
        dataset = self.datasets[self.dataset_index]
        self.dataset_index = self.dataset_index % 3
        item = dataset[index % len(dataset)]
        return item
    
    def collater(self, samples):
        return self.datasets[self.dataset_index].collater(samples)

    def __len__(self) -> int:
        return self.total_length

    def __repr__(self) -> str:
        pass
    
    def num_tokens(self, index):
        return 1

    def size(self, index):
        return 1

    @property
    def sizes(self):
        return np.full((len(self),), 1)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        return order[0]