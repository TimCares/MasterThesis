from typing import Tuple, Dict, Any
from .unimodal_datamodules import BaseDataModule
from datasets_ import COCOCaptions, VisualGenome, VQAv2, NLVR2, Flickr30Dataset, CommonVoice, Flickr8KAudioDataset

class COCOCaptionsDataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 transform_jitter=False,
                 beit_transforms=False,
                 no_transform=False,
                 crop_scale=(0.6, 1.0),
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale

    def set_train_dataset(self):
        self.train_dataset = COCOCaptions(data_path=self.data_path,
                                          split='train',
                                          num_max_bpe_tokens=self.num_max_bpe_tokens,
                                          transform_jitter=self.transform_jitter,
                                          beit_transforms=self.beit_transforms,
                                          no_transform=self.no_transform,
                                          crop_scale=self.crop_scale,)

    def set_val_dataset(self):
        self.val_dataset = COCOCaptions(data_path=self.data_path,
                                        split='val',
                                        num_max_bpe_tokens=self.num_max_bpe_tokens,
                                        transform_jitter=False,
                                        beit_transforms=False,
                                        no_transform=True,)

    def set_test_dataset(self):
        self.test_dataset = COCOCaptions(data_path=self.data_path,
                                         split='test',
                                         num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         transform_jitter=False,
                                         beit_transforms=False,
                                         no_transform=True,)
        

class VisualGenomeDataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 transform_jitter=False,
                 beit_transforms=False,
                 no_transform=False,
                 crop_scale=(0.6, 1.0),
                 n_caption_groups:int=1,
                 concat_captions:bool=True,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale
        self.n_caption_groups = n_caption_groups
        self.concat_captions = concat_captions

    def prepare_data(self):
        if not self.prepared:
            self.set_train_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()


    def set_train_dataset(self):
        self.train_dataset = VisualGenome(data_path=self.data_path,
                                          split='train',
                                          num_max_bpe_tokens=self.num_max_bpe_tokens,
                                          transform_jitter=self.transform_jitter,
                                          beit_transforms=self.beit_transforms,
                                          no_transform=self.no_transform,
                                          crop_scale=self.crop_scale,
                                          n_caption_groups=self.n_caption_groups,
                                          concat_captions=self.concat_captions,)
        

class VQAv2DataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 transform_jitter=False,
                 beit_transforms=False,
                 no_transform=False,
                 crop_scale=(0.6, 1.0),
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale


    def set_train_dataset(self):
        self.train_dataset = VQAv2(data_path=self.data_path,
                                   split='train',
                                   num_max_bpe_tokens=self.num_max_bpe_tokens,
                                   transform_jitter=self.transform_jitter,
                                   beit_transforms=self.beit_transforms,
                                   no_transform=self.no_transform,
                                   crop_scale=self.crop_scale,)

    def set_val_dataset(self):
        self.val_dataset = VQAv2(data_path=self.data_path,
                                 split='val',
                                 num_max_bpe_tokens=self.num_max_bpe_tokens,
                                 transform_jitter=False,
                                 beit_transforms=False,
                                 no_transform=True,)

    def set_test_dataset(self):
        self.test_dataset = VQAv2(data_path=self.data_path,
                                  split='test', # TODO: add test-dev?
                                  num_max_bpe_tokens=self.num_max_bpe_tokens,
                                  transform_jitter=False,
                                  beit_transforms=False,
                                  no_transform=True,)
        

class NLVR2DataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 transform_jitter=False,
                 beit_transforms=False,
                 no_transform=False,
                 crop_scale=(0.6, 1.0),
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale


    def set_train_dataset(self):
        self.train_dataset = NLVR2(data_path=self.data_path,
                                   split='train',
                                   num_max_bpe_tokens=self.num_max_bpe_tokens,
                                   transform_jitter=self.transform_jitter,
                                   beit_transforms=self.beit_transforms,
                                   no_transform=self.no_transform,
                                   crop_scale=self.crop_scale,)

    def set_val_dataset(self):
        self.val_dataset = NLVR2(data_path=self.data_path,
                                split='val',
                                num_max_bpe_tokens=self.num_max_bpe_tokens,
                                transform_jitter=False,
                                beit_transforms=False,
                                no_transform=True,)

    def set_test_dataset(self):
        self.test_dataset = NLVR2(data_path=self.data_path,
                                  split='test',
                                  num_max_bpe_tokens=self.num_max_bpe_tokens,
                                  transform_jitter=False,
                                  beit_transforms=False,
                                  no_transform=True,)


class Flickr30DataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens


    def set_train_dataset(self):
        self.train_dataset = Flickr30Dataset(data_path=self.data_path,
                                             split='train',
                                             num_max_bpe_tokens=self.num_max_bpe_tokens,)

    def set_val_dataset(self):
        self.val_dataset = Flickr30Dataset(data_path=self.data_path,
                                           split='val',
                                           num_max_bpe_tokens=self.num_max_bpe_tokens,)

    def set_test_dataset(self): # to be used for zero-shot retrieval
        self.test_dataset = Flickr30Dataset(data_path=self.data_path,
                                            split='test',
                                            num_max_bpe_tokens=self.num_max_bpe_tokens,)
        

class Flickr8AudioDataModule(BaseDataModule):
    def __init__(self,
                 data_path:str,
                 transform_jitter:bool,
                 beit_transforms:bool,
                 no_transform:bool,
                 crop_scale:Tuple[float, float],
                 sample_rate:int,
                 max_sample_size:int,
                 min_sample_size:int,
                 normalize:bool,
                 pad:bool,
                 batch_size:int,
                 num_workers:int,
                 shuffle:bool=True,
                 drop_last:bool=True,
                 precompute_mask_config:Dict[str, Any]={},):
        super().__init__(data_path=data_path,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         shuffle=shuffle,
                         drop_last=drop_last,)
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.normalize = normalize
        self.pad = pad
        self.precompute_mask_config = precompute_mask_config

    def prepare_data(self):
        if not self.prepared:
            self.set_test_dataset()

            self.prepared = True

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.test_dataset.load()


    def set_train_dataset(self):
        self.train_dataset = Flickr8KAudioDataset(data_path=self.data_path,
                                                  split='train',
                                                  sample_rate=self.sample_rate,
                                                  max_sample_size=self.max_sample_size,
                                                  min_sample_size=self.min_sample_size,
                                                  normalize=self.normalize,
                                                  pad=self.pad,
                                                  precompute_mask_config=self.precompute_mask_config)

    def set_val_dataset(self):
        self.val_dataset = Flickr8KAudioDataset(data_path=self.data_path,
                                                split='val',
                                                sample_rate=self.sample_rate,
                                                max_sample_size=self.max_sample_size,
                                                min_sample_size=self.min_sample_size,
                                                normalize=self.normalize,
                                                pad=self.pad,
                                                precompute_mask_config=None)

    def set_test_dataset(self): # to be used for zero-shot retrieval
        self.test_dataset = Flickr8KAudioDataset(data_path=self.data_path,
                                                 split='test',
                                                 sample_rate=self.sample_rate,
                                                 max_sample_size=self.max_sample_size,
                                                 min_sample_size=self.min_sample_size,
                                                 normalize=self.normalize,
                                                 pad=self.pad,
                                                 precompute_mask_config=None)
            

class CommonVoiceDataModule(BaseDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 sample_rate,
                 max_sample_size,
                 min_sample_size,
                 normalize,
                 pad,
                 batch_size:int,
                 num_workers:int,
                 shuffle:bool=True,
                 drop_last:bool=True,
                 precompute_mask_config:Dict[str, Any]={}):
        super().__init__(data_path, batch_size, num_workers, shuffle, drop_last)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.normalize = normalize
        self.pad = pad
        self.precompute_mask_config = precompute_mask_config

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
        self.train_dataset = CommonVoice(data_path=self.data_path,
                                         split='train',
                                         num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         sample_rate=self.sample_rate,
                                         max_sample_size=self.max_sample_size,
                                         min_sample_size=self.min_sample_size,
                                         normalize=self.normalize,
                                         pad=self.pad,
                                         precompute_mask_config=self.precompute_mask_config,)
        
    def set_test_dataset(self):
        self.test_dataset = CommonVoice(data_path=self.data_path,
                                        split='retrieval', # is the test split for zero-shot retrieval
                                        num_max_bpe_tokens=self.num_max_bpe_tokens,
                                        sample_rate=self.sample_rate,
                                        max_sample_size=self.max_sample_size,
                                        min_sample_size=self.min_sample_size,
                                        normalize=self.normalize,
                                        pad=self.pad,
                                        precompute_mask_config=None,)


MULTIMODAL_DATAMODULE_REGISTRY = {
    "coco_captions": COCOCaptionsDataModule,
    "visual_genome": VisualGenomeDataModule,
    "vqa_v2": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "flickr30": Flickr30DataModule,
    "flickr8_audio": Flickr8AudioDataModule,
    "common_voice": CommonVoiceDataModule,
}