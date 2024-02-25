import logging
import torch
import torchtext
from torchtext.datasets import WikiText103
from torchaudio.datasets import LIBRISPEECH
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import IPython.display as ipd
import matplotlib.pyplot as plt
import math

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_librispeech_dataset(spectrogram:bool=True, dataset:str="train-clean-100", mask_percentage:float=0.05, consequitive:bool=False,
                 batch_size:int=32, shuffle:bool=True, num_workers:int=1):
    librispeech = LIBRISPEECH(root="./data", url=dataset, download=True)

    if spectrogram and consequitive:
        mel_spectrogram = T.MelSpectrogram()
        # 100_000 so that every possible wavelength can be masked up to the maximum mask percentage (mask_percentage)
        masking = T.TimeMasking(time_mask_param=100_000, iid_masks=True, p=mask_percentage)

        def collate_fn(batch):
            return masking(mel_spectrogram(batch)), batch
        
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
            
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
            max_length_per_patch = mask_length//n_patches
            start = torch.randint(0, num_time_steps - max_length_per_patch, (n_patches,))
            for idx, s in enumerate(start):
                X[idx, ..., s:s+max_length_per_patch] = 0
            return X, batch
        
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

    elif not spectrogram and consequitive:
        max_batches = math.ceil(len(librispeech)/batch_size)

        def collate_fn(batch):
            X = batch.clone()
            num_time_steps = X.shape[-1]
            mask_length = int(num_time_steps * mask_percentage)
            starts = torch.randint(0, num_time_steps - mask_length, (batch_size, ))
            for idx, s in enumerate(starts):
                X[idx, ..., s:s+mask_length] = 0
            return X, batch
        
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

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
            max_length_per_patch = mask_length//n_patches
            start = torch.randint(0, num_time_steps - max_length_per_patch, (n_patches,))
            for idx, s in enumerate(start):
                X[idx, ..., s:s+max_length_per_patch] = 0
            return X, batch
        
        return DataLoader(librispeech, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
