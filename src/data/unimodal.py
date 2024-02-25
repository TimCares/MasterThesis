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
