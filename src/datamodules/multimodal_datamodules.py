from datamodules.unimodal_datamodules import BaseDataModule
from datasets import COCOCaptions

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
                                        transform_jitter=self.transform_jitter,
                                        beit_transforms=self.beit_transforms,
                                        no_transform=self.no_transform,
                                        crop_scale=self.crop_scale,)

    def set_test_dataset(self):
        self.test_dataset = COCOCaptions(data_path=self.data_path,
                                         split='test',
                                         num_max_bpe_tokens=self.num_max_bpe_tokens,
                                         transform_jitter=self.transform_jitter,
                                         beit_transforms=self.beit_transforms,
                                         no_transform=self.no_transform,
                                         crop_scale=self.crop_scale,)
            