from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, List
from datasets_ import IMDBDataset, ImageNetDataset, LibriSpeechDataset, SpeechCommandsDataset, OpenWebTextDataset, QQPDataset
from datasets_ import DATASET_REGISTRY
from functools import partial

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
        pass # optional: not all datasets have a validation set

    def set_test_dataset(self):
        pass # optional: not all datasets have a test set

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
        if not hasattr(self, 'val_dataset'):
            return None
        return DataLoader(self.val_dataset,
                          collate_fn=self.val_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=self.drop_last,)

    def test_dataloader(self):
        if not hasattr(self, 'test_dataset'):
            return None
        return DataLoader(self.test_dataset,
                          collate_fn=self.test_dataset.collater,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          sampler=None,
                          shuffle=False,
                          drop_last=self.drop_last,)
    
    def teardown(self, stage: str) -> None:
        if stage == 'fit' or stage is None:
            if hasattr(self, 'train_dataset'):
                del self.train_dataset
            if hasattr(self, 'val_dataset'):
                del self.val_dataset
        if stage == 'test' or stage is None:
            if hasattr(self, 'test_dataset'):
                del self.test_dataset


class IMDBDataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 num_max_bpe_tokens:int,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens

    def prepare_data(self): # only for validation datasets
        self.set_train_dataset()
        self.set_test_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = IMDBDataset(data_path=self.data_path, split='train', num_max_bpe_tokens=self.num_max_bpe_tokens)

    def set_test_dataset(self):
        self.test_dataset = IMDBDataset(data_path=self.data_path, split='test', num_max_bpe_tokens=self.num_max_bpe_tokens)

class QQPDataModule(BaseDataModule):
    def __init__(self, 
                 data_path:str,
                 num_max_bpe_tokens:int,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens

    def prepare_data(self): # only for validation datasets
        self.set_train_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = QQPDataset(data_path=self.data_path, num_max_bpe_tokens=self.num_max_bpe_tokens)


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
        assert type in ['cifar10', 'cifar100'], "Cifar dataset type must be in ['cifar10', 'cifar100']."
        self.type = type

    def prepare_data(self): # only for validation datasets
        self.set_train_dataset()
        self.set_test_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = DATASET_REGISTRY[self.type](data_path=self.data_path, split='train')

    def set_test_dataset(self):
        self.test_dataset = DATASET_REGISTRY[self.type](data_path=self.data_path, split='test')


class ImageNetDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 raw_image,
                 beit_transforms,
                 no_transform,
                 transform_jitter,
                 precompute_mask_config,
                 crop_scale,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.raw_image = raw_image
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.transform_jitter = transform_jitter
        self.precompute_mask_config = precompute_mask_config
        self.crop_scale = crop_scale

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
                                             raw_image=self.raw_image,
                                             beit_transforms=self.beit_transforms,
                                             no_transform=self.no_transform,
                                             transform_jitter=self.transform_jitter,
                                             crop_scale=self.crop_scale,
                                             precompute_mask_config=self.precompute_mask_config,
                                             )

    def set_val_dataset(self):
        self.val_dataset = ImageNetDataset(data_path=self.data_path,
                                           split='val',
                                           raw_image=self.raw_image,
                                           beit_transforms=False,
                                           no_transform=True,
                                           transform_jitter=False,
                                           crop_scale=self.crop_scale,
                                           precompute_mask_config=None,
                                           )
        

class LibriSpeechDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 sample_rate:int,
                 max_sample_size:int,
                 min_sample_size:int,
                 normalize:bool,
                 pad:bool,
                 types_train:Tuple[str],
                 types_test:Tuple[str]=None,
                 return_path:bool=False,
                 precompute_mask_config=None,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.normalize = normalize
        self.pad = pad
        self.types_train = types_train
        self.types_test = types_test
        self.precompute_mask_config = precompute_mask_config
        self.return_path = return_path

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()
            if self.types_test is not None:
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
                                                normalize=self.normalize,
                                                pad=self.pad,
                                                types=self.types_train,
                                                precompute_mask_config=self.precompute_mask_config,
                                                return_path=self.return_path)

    def set_test_dataset(self):
        self.test_dataset = LibriSpeechDataset(data_path=self.data_path,
                                               split='test',
                                               sample_rate=self.sample_rate,
                                               max_sample_size=self.max_sample_size,
                                               min_sample_size=0,
                                               normalize=self.normalize,
                                               pad=self.pad,
                                               types=self.types_test,
                                               precompute_mask_config=None,
                                               return_path=self.return_path)
        

class SpeechCommandsDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 min_sample_size: int,
                 normalize: bool,
                 pad: bool,
                 *args,
                 **kwargs
                 ):
        super().__init__(data_path, *args, **kwargs)
        self.min_sample_size = min_sample_size
        self.normalize = normalize
        self.pad = pad

    def prepare_data(self): # only for validation datasets
        self.set_train_dataset()
        self.set_test_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()
        if stage == 'test' or stage is None:
            self.test_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = SpeechCommandsDataset(data_path=self.data_path,
                                                   split='train',
                                                   normalize=self.normalize,
                                                   pad=self.pad,)

    def set_test_dataset(self):
        self.test_dataset = SpeechCommandsDataset(data_path=self.data_path,
                                                  split='test',
                                                  normalize=self.normalize,
                                                  pad=self.pad,)


UNIMODAL_DATAMODULE_REGISTRY = {
    'imdb': IMDBDataModule,
    'openwebtext': OpenWebTextDataModule,
    'cifar10': partial(CIFARDataModule, type='cifar10'),
    'cifar100': partial(CIFARDataModule, type='cifar100'),
    'imagenet': ImageNetDataModule,
    'librispeech': LibriSpeechDataModule,
    'speechcommands': SpeechCommandsDataModule,
    'qqp': QQPDataModule,
}
