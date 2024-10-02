from typing import Tuple, Dict, Any
from .unimodal_datamodules import BaseDataModule
from data2vec_fairseq.data.modality import Modality
from datasets_ import COCOCaptions, VisualGenome, VQAv2, NLVR2, Flickr30Dataset, CommonVoice, Flickr8KAudioDataset, ConceptualCaptions, SBUCaptions
from functools import partial

class BaseImageTextDataModule(BaseDataModule):
    @property
    def modality(self) -> Modality:
        return Modality.VL
    
class BaseTextAudioDataModule(BaseDataModule):
    @property
    def modality(self) -> Modality:
        return Modality.LA
    
class BaseImageAudioDataModule(BaseDataModule):
    @property
    def modality(self) -> Modality:
        return Modality.VA

class COCOCaptionsDataModule(BaseImageTextDataModule):
    def __init__(self,
                data_path,
                num_max_bpe_tokens,
                task="captioning",
                color_jitter=None,
                beit_transforms=False,
                crop_scale=(0.6, 1.0),
                text_token_mask_prob=0.0,
                *args,
                **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.task = task
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale
        self.text_token_mask_prob = text_token_mask_prob

    def set_train_dataset(self):
        self.train_dataset = COCOCaptions(data_path=self.data_path,
                                          split='train',
                                          num_max_bpe_tokens=self.num_max_bpe_tokens,
                                          task=self.task,
                                          color_jitter=self.color_jitter,
                                          beit_transforms=self.beit_transforms,
                                          crop_scale=self.crop_scale,
                                          text_token_mask_prob=self.text_token_mask_prob,)

    def set_val_dataset(self):
        self.val_dataset = COCOCaptions(data_path=self.data_path,
                                        split='val',
                                        num_max_bpe_tokens=self.num_max_bpe_tokens,
                                        task=self.task,
                                        color_jitter=False,
                                        beit_transforms=False,
                                        crop_scale=(1.0, 1.0),
                                        text_token_mask_prob=self.text_token_mask_prob,)

    def set_test_dataset(self):
        self.test_dataset = COCOCaptions(data_path=self.data_path,
                                         split='test',
                                         num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         task=self.task,
                                         color_jitter=False,
                                         beit_transforms=False,
                                         crop_scale=(1.0, 1.0),
                                         text_token_mask_prob=0.0,)
        

class VisualGenomeDataModule(BaseImageTextDataModule):
    def __init__(self,
                 data_path,
                 concat_captions,
                 num_max_bpe_tokens,
                 color_jitter=None,
                 beit_transforms=False,
                 crop_scale=(0.6, 1.0),
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.concat_captions = concat_captions
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()


    def set_train_dataset(self):
        self.train_dataset = VisualGenome(data_path=self.data_path,
                                          split='train',
                                          concat_captions=self.concat_captions,
                                          num_max_bpe_tokens=self.num_max_bpe_tokens,
                                          color_jitter=self.color_jitter,
                                          beit_transforms=self.beit_transforms,
                                          crop_scale=self.crop_scale,)
        

class ConceptualCaptionsDataModule(BaseImageTextDataModule):
    def __init__(self,
                type,
                data_path,
                num_max_bpe_tokens,
                color_jitter=None,
                beit_transforms=False,
                crop_scale=(0.6, 1.0),
                text_token_mask_prob=0.0,
                *args,
                **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.type = type
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale
        self.text_token_mask_prob = text_token_mask_prob

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = ConceptualCaptions(
            type=self.type,
            data_path=self.data_path,
            split='train',
            num_max_bpe_tokens=self.num_max_bpe_tokens,
            color_jitter=self.color_jitter,
            beit_transforms=self.beit_transforms,
            crop_scale=self.crop_scale,
            text_token_mask_prob=self.text_token_mask_prob,)
        

class SBUCaptionsDataModule(BaseImageTextDataModule):
    def __init__(self,
                data_path,
                num_max_bpe_tokens,
                color_jitter=None,
                beit_transforms=False,
                crop_scale=(0.6, 1.0),
                *args,
                **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale

    def prepare_data(self):
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset.load()

    def set_train_dataset(self):
        self.train_dataset = SBUCaptions(data_path=self.data_path,
                                         split='train',
                                         num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         color_jitter=self.color_jitter,
                                         beit_transforms=self.beit_transforms,
                                         crop_scale=self.crop_scale,)
        

class VQAv2DataModule(BaseImageTextDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 color_jitter=False,
                 beit_transforms=False,
                 crop_scale=(0.6, 1.0),
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale


    def set_train_dataset(self):
        self.train_dataset = VQAv2(data_path=self.data_path,
                                   split='train',
                                   num_max_bpe_tokens=self.num_max_bpe_tokens,
                                   color_jitter=self.color_jitter,
                                   beit_transforms=self.beit_transforms,
                                   crop_scale=self.crop_scale,)

    def set_val_dataset(self):
        self.val_dataset = VQAv2(data_path=self.data_path,
                                 split='val',
                                 num_max_bpe_tokens=self.num_max_bpe_tokens,
                                 color_jitter=False,
                                 beit_transforms=False,
                                 crop_scale=(1.0, 1.0),)

    def set_test_dataset(self):
        self.test_dataset = VQAv2(data_path=self.data_path,
                                  split='test',
                                  num_max_bpe_tokens=self.num_max_bpe_tokens,
                                  color_jitter=False,
                                  beit_transforms=False,
                                  crop_scale=(1.0, 1.0),)
        

class NLVR2DataModule(BaseImageTextDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 color_jitter=False,
                 beit_transforms=False,
                 crop_scale=(0.6, 1.0),
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale


    def set_train_dataset(self):
        self.train_dataset = NLVR2(data_path=self.data_path,
                                   split='train',
                                   num_max_bpe_tokens=self.num_max_bpe_tokens,
                                   color_jitter=self.color_jitter,
                                   beit_transforms=self.beit_transforms,
                                   crop_scale=self.crop_scale,)

    def set_val_dataset(self):
        self.val_dataset = NLVR2(data_path=self.data_path,
                                split='val',
                                num_max_bpe_tokens=self.num_max_bpe_tokens,
                                color_jitter=False,
                                beit_transforms=False,
                                crop_scale=(1.0, 1.0),)

    def set_test_dataset(self):
        self.test_dataset = NLVR2(data_path=self.data_path,
                                  split='test',
                                  num_max_bpe_tokens=self.num_max_bpe_tokens,
                                  color_jitter=False,
                                  beit_transforms=False,
                                  crop_scale=(1.0, 1.0),)


class Flickr30DataModule(BaseImageTextDataModule):
    def __init__(self,
                 data_path,
                 num_max_bpe_tokens,
                 color_jitter=None,
                 beit_transforms=False,
                 crop_scale=(0.6, 1.0),
                 text_token_mask_prob=0.0,
                 *args,
                 **kwargs):
        super().__init__(data_path, *args, **kwargs)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.color_jitter = color_jitter
        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale
        self.text_token_mask_prob = text_token_mask_prob

    def set_train_dataset(self):
        self.train_dataset = Flickr30Dataset(data_path=self.data_path,
                                             split='train',
                                             num_max_bpe_tokens=self.num_max_bpe_tokens,
                                             color_jitter=self.color_jitter,
                                             beit_transforms=self.beit_transforms,
                                             crop_scale=self.crop_scale,
                                             text_token_mask_prob=self.text_token_mask_prob,)

    def set_val_dataset(self):
        self.val_dataset = Flickr30Dataset(data_path=self.data_path,
                                           split='val',
                                           num_max_bpe_tokens=self.num_max_bpe_tokens,
                                           color_jitter=False,
                                           beit_transforms=False,
                                           crop_scale=(1.0, 1.0),
                                           text_token_mask_prob=self.text_token_mask_prob,)

    def set_test_dataset(self):
        self.test_dataset = Flickr30Dataset(data_path=self.data_path,
                                            split='test',
                                            num_max_bpe_tokens=self.num_max_bpe_tokens,
                                            color_jitter=False,
                                            beit_transforms=False,
                                            crop_scale=(1.0, 1.0),
                                            text_token_mask_prob=self.text_token_mask_prob,)
        

class Flickr8AudioDataModule(BaseImageAudioDataModule):
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
        if not hasattr(self, 'test_dataset'):
            self.set_test_dataset()

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
            

class CommonVoiceDataModule(BaseTextAudioDataModule):
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
        if not hasattr(self, 'train_dataset'):
            self.set_train_dataset()
        if not hasattr(self, 'test_dataset'):
            self.set_test_dataset()

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
    "conceptual_captions": ConceptualCaptionsDataModule,
    "conceptual_captions_3m": partial(ConceptualCaptionsDataModule, type='cc3m'),
    "conceptual_captions_12m": partial(ConceptualCaptionsDataModule, type='cc12m'),
    "vqa_v2": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "flickr30": Flickr30DataModule,
    "flickr8_audio": Flickr8AudioDataModule,
    "common_voice": CommonVoiceDataModule,
    "sbu": SBUCaptionsDataModule,
}