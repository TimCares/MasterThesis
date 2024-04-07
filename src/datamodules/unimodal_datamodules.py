from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from src.datasets import IMDBDataset, CIFARDataset, ImageNetDataset, LibriSpeechDataset, SpeechCommandsDataset, EnWik9Dataset, OpenWebTextDataset

class BaseDataModule(LightningDataModule):
    def __init__(self,
                 data_path:str,
                 batch_size:int,
                 num_workers:int,
                 shuffle:bool=True,
                 drop_last:bool=True,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.prepared = False
    
    def set_train_dataset(self):
        raise NotImplementedError("set train dataset")

    def set_val_dataset(self):
        raise NotImplementedError("set val dataset")

    def set_test_dataset(self):
        raise NotImplementedError("set test dataset")

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          collate_fn=self.train_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          collate_fn=self.val_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          collate_fn=self.test_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=self.shuffle,
                          drop_last=self.drop_last,)
    

class IMDBDataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 num_max_bpe_tokens:int,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_test_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = IMDBDataset(data_path=self.data_path, split='train', num_max_bpe_tokens=self.num_max_bpe_tokens)

    def set_test_dataset(self):
        self.test_dataset = IMDBDataset(data_path=self.data_path, split='test', num_max_bpe_tokens=self.num_max_bpe_tokens)


class EnWik9DataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 num_max_bpe_tokens:int,
                 sample_break_mode:str,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.sample_break_mode = sample_break_mode

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_val_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = EnWik9Dataset(data_path=self.data_path, split='train', num_max_bpe_tokens=self.num_max_bpe_tokens,
                                           sample_break_mode=self.sample_break_mode)

    def set_val_dataset(self):
        self.val_dataset = EnWik9Dataset(data_path=self.data_path, split='val', num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         sample_break_mode=self.sample_break_mode)
        

class OpenWebTextDataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 num_max_bpe_tokens:int,
                 sample_break_mode:str,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.sample_break_mode = sample_break_mode

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = OpenWebTextDataset(data_path=self.data_path, split='train', num_max_bpe_tokens=self.num_max_bpe_tokens,
                                                sample_break_mode=self.sample_break_mode)


class CIFARDataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 type:str,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.type = type

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_test_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = CIFARDataset(data_path=self.data_path, split='train', type=self.type)

    def set_test_dataset(self):
        self.test_dataset = CIFARDataset(data_path=self.data_path, split='test', type=self.type)


class ImageNetDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 beit_transforms,
                 no_transform,
                 transform_jitter,
                 precompute_mask_config,
                 crop_scale,
                 local_cache_path,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.transform_jitter = transform_jitter
        self.precompute_mask_config = precompute_mask_config
        self.crop_scale = crop_scale
        self.local_cache_path = local_cache_path

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_val_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
            self.val_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = ImageNetDataset(data_path=self.data_path, 
                                             split='train',
                                             beit_transforms=self.beit_transforms,
                                             no_transform=self.no_transform,
                                             transform_jitter=self.transform_jitter,
                                             precompute_mask_config=self.precompute_mask_config,
                                             crop_scale=self.crop_scale,
                                             local_cache_path=self.local_cache_path,)

    def set_val_dataset(self):
        self.val_dataset = ImageNetDataset(data_path=self.data_path,
                                           split='val',
                                           beit_transforms=self.beit_transforms,
                                           no_transform=self.no_transform,
                                           transform_jitter=self.transform_jitter,
                                           precompute_mask_config=self.precompute_mask_config,
                                           crop_scale=self.crop_scale,
                                           local_cache_path=self.local_cache_path,)
        

class LibriSpeechDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 sample_rate:int,
                 max_sample_size:int,
                 min_sample_size:int,
                 precompute_mask_config,
                 type:str,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.precompute_mask_config = precompute_mask_config
        self.type = type

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_test_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = LibriSpeechDataset(data_path=self.data_path,
                                                split='train',
                                                sample_rate=self.sample_rate,
                                                max_sample_size=self.max_sample_size,
                                                min_sample_size=self.min_sample_size,
                                                precompute_mask_config=self.precompute_mask_config,
                                                type=self.type,)

    def set_test_dataset(self):
        self.test_dataset = LibriSpeechDataset(data_path=self.data_path,
                                               split='test',
                                               sample_rate=self.sample_rate,
                                               max_sample_size=self.max_sample_size,
                                               min_sample_size=self.min_sample_size,
                                               precompute_mask_config=self.precompute_mask_config,
                                               type='test-other',)
        

class SpeechCommandsDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 min_sample_size: int,
                 normalize: bool,
                 pad: bool,
                 precompute_mask_config:Dict[str, Any]={},
                 *args,
                 **kwargs
                 ):
        super().__init__(data_path, *args, **kwargs)
        self.precompute_mask_config = precompute_mask_config
        self.min_sample_size = min_sample_size
        self.normalize = normalize
        self.pad = pad

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            self.set_test_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = SpeechCommandsDataset(data_path=self.data_path,
                                                   split='train',
                                                   min_sample_size=self.min_sample_size,
                                                   normalize=self.normalize,
                                                   pad=self.pad,
                                                   precompute_mask_config=self.precompute_mask_config,)

    def set_test_dataset(self):
        self.test_dataset = SpeechCommandsDataset(data_path=self.data_path,
                                                  split='test',
                                                  precompute_mask_config=self.precompute_mask_config,)


UNIMODAL_REGISTRY = {
    'imdb': IMDBDataModule,
    'enwik9': EnWik9DataModule,
    'openwebtext': OpenWebTextDataModule,
    'cifar': CIFARDataModule,
    'imagenet': ImageNetDataModule,
    'librispeech': LibriSpeechDataModule,
    'speechcommands': SpeechCommandsDataModule
}