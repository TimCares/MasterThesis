# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import partial
from typing import Tuple
import logging
import math
import random
import time
import json

import numpy as np
import os

import torch

from torchvision import datasets
from torchvision.transforms import v2 as transforms
import torchvision.transforms.functional as F
from PIL import Image
from fairseq.data import FairseqDataset, Dictionary, data_utils
from multiprocessing_bpe_encoder import BPEEncoder

from shutil import copyfile

logger = logging.getLogger(__name__)


def load(path, loader, cache):
    if hasattr(caching_loader, "cache_root"):
        cache = caching_loader.cache_root

    cached_path = cache + path

    num_tries = 3
    for curr_try in range(num_tries):
        try:
            if curr_try == 2:
                return loader(path)
            if not os.path.exists(cached_path) or curr_try > 0:
                os.makedirs(os.path.dirname(cached_path), exist_ok=True)
                copyfile(path, cached_path)
                os.chmod(cached_path, 0o777)
            return loader(cached_path)
        except Exception as e:
            logger.warning(str(e))
            if "Errno 13" in str(e):
                caching_loader.cache_root = f"/scratch/{random.randint(0, 69420)}"
                logger.warning(f"setting cache root to {caching_loader.cache_root}")
                cached_path = caching_loader.cache_root + path
            if curr_try == (num_tries - 1):
                raise
            time.sleep(2)


def caching_loader(cache_root: str, loader):
    if cache_root is None:
        return loader

    if cache_root == "slurm_tmpdir":
        cache_root = os.environ["SLURM_TMPDIR"]
        assert len(cache_root) > 0

    if not cache_root.endswith("/"):
        cache_root += "/"

    return partial(load, loader=loader, cache=cache_root)


class RandomResizedCropAndInterpolationWithTwoPic:
    """Crop the given PIL Image to random size and aspect ratio with random interpolation.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size,
        second_size=None,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation="bilinear",
        second_interpolation="lanczos",
    ):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            logger.warning("range should be of kind (min, max)")

        if interpolation == "random":
            from PIL import Image

            self.interpolation = (Image.BILINEAR, Image.BICUBIC)
        else:
            self.interpolation = self._pil_interp(interpolation)

        self.second_interpolation = (
            self._pil_interp(second_interpolation)
            if second_interpolation is not None
            else None
        )
        self.scale = scale
        self.ratio = ratio

    def _pil_interp(self, method):
        from PIL import Image

        if method == "bicubic":
            return Image.BICUBIC
        elif method == "lanczos":
            return Image.LANCZOS
        elif method == "hamming":
            return Image.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return Image.BILINEAR

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(
                img, i, j, h, w, self.size, interpolation
            ), F.resized_crop(
                img, i, j, h, w, self.second_size, self.second_interpolation
            )


class COCOCaptionsPlots(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        local_cache_path=None,
        shuffle=True,
        key="imgs",
        beit_transforms=False,
        no_transform=False,
        compute_mask=False,
        patch_size: int = 16,
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        crop_scale:Tuple[float, float]=(0.08, 1.0),
    ):
        FairseqDataset.__init__(self)

        self.shuffle = shuffle
        self.key = key

        self.loader = caching_loader(local_cache_path, datasets.folder.default_loader)

        self.transform_source = None

        # self.transform_source = transforms.ColorJitter(0.4, 0.4, 0.4)

        if no_transform:
            if input_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(input_size / crop_pct)

            self.transform_train = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=3),
                    transforms.CenterCrop(input_size),
                ]
            )

            self.transform_train = transforms.Resize((input_size, input_size))
        elif beit_transforms:
            beit_transform_list = []
            beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
            beit_transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandomResizedCropAndInterpolationWithTwoPic(
                        size=input_size,
                        second_size=None,
                        interpolation="bicubic",
                        second_interpolation=None,
                        scale=crop_scale
                    ),
                ]
            )
            self.transform_train = transforms.Compose(beit_transform_list)
        else:
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        input_size, scale=crop_scale, interpolation=3
                    ),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                ]
            )
        self.final_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        root = root + f'/karpathy_{split}/' # for 'karpathy_<dataset>' folder structure
        folder_path = root.split('/karpathy')[0]

        assert os.path.exists(folder_path + '/dataset_coco_karpathy.json'), f'File "dataset_coco_karpathy.json"\
        not found in the folder {folder_path}!'

        with open(folder_path + '/dataset_coco_karpathy.json') as f:
            raw_meta_data = json.load(f)['images']

        assert root[0]=='/', 'root should be an absolute path'
        self.meta_data = []
        for meta_for_example in raw_meta_data:
            if meta_for_example['split'] == split:
                # convert the absolute path for reading
                meta_for_example['full_path'] = root + meta_for_example['filename']
                self.meta_data.append(meta_for_example)

        logger.info(
            f"initial transform: {self.transform_train}, "
            f"source transform: {self.transform_source}, "
            f"final transform: {self.final_transform}"
        )
        logger.info(f"loaded {len(self.meta_data)} examples")

        self.is_compute_mask = compute_mask
        self.patches = (input_size // patch_size) ** 2
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch

    def __getitem__(self, index):
        path = self.meta_data[index]['full_path']

        if self.loader is not None:
            img = self.loader(path)
        else:
            img = Image.open(path).convert("RGB")

        caption_choice_idx = torch.randint(low=0, high=5, size=(1,)) # TODO: Do in collater batch-wise, also do random switch between image target and text target

        caption = self.meta_data[index]['sentences'][caption_choice_idx]['raw']

        img = self.transform_train(img)

        source = None
        if self.transform_source is not None:
            source = self.final_transform(self.transform_source(img))

        if source is None:
            img = self.final_transform(img)

        v = {"id": index, self.key: source if source is not None else img}
        v["target"] = caption

        if self.is_compute_mask:
            if self.mask_length == 1:
                mask = data_utils.compute_block_mask_1d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                )
            else:
                mask = data_utils.compute_block_mask_2d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                    expand_adjcent=self.expand_adjacent,
                    mask_dropout=self.mask_dropout,
                    non_overlapping=self.non_overlapping,
                )

            v["precomputed_mask"] = mask

        return v

    def __len__(self):
        return len(self.meta_data)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s[self.key] for s in samples], dim=0)

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": {
                self.key: collated_img,
            },
        }

        if "target" in samples[0]:
            collated_target = torch.stack([s["target"] for s in samples], dim=0)
            res["net_input"]["target"] = collated_target

        if "precomputed_mask" in samples[0]:
            collated_mask = torch.cat([s["precomputed_mask"] for s in samples], dim=0)
            res["net_input"]["precomputed_mask"] = collated_mask

        return res

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
    

class COCOCaptions(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        local_cache_path=None,
        shuffle=True,
        key="imgs",
        transform_jitter=False,
        beit_transforms=False,
        no_transform=False,
        compute_mask=False,
        patch_size: int = 16,
        mask_prob: float = 0.75,
        mask_prob_adjust: float = 0,
        mask_length: int = 1,
        inverse_mask: bool = False,
        expand_adjacent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
        require_same_masks: bool = True,
        clone_batch: int = 1,
        crop_scale:Tuple[float, float]=(0.08, 1.0),
        tokens_per_sample: int = 512,
    ):
        FairseqDataset.__init__(self)

        self.shuffle = shuffle
        self.key = key

        self.loader = caching_loader(local_cache_path, datasets.folder.default_loader)

        self.transform_jitter = None
        if transform_jitter:
            self.transform_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)

        if no_transform:
            if input_size <= 224:
                crop_pct = 224 / 256
            else:
                crop_pct = 1.0
            size = int(input_size / crop_pct)

            self.transform_train = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=3),
                    transforms.CenterCrop(input_size),
                ]
            )

            self.transform_train = transforms.Resize((input_size, input_size))
        elif beit_transforms:
            beit_transform_list = []
            beit_transform_list.append(transforms.ColorJitter(0.4, 0.4, 0.4))
            beit_transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandomResizedCropAndInterpolationWithTwoPic(
                        size=input_size,
                        second_size=None,
                        interpolation="bicubic",
                        second_interpolation=None,
                        scale=crop_scale
                    ),
                ]
            )
            self.transform_train = transforms.Compose(beit_transform_list)
        else:
            self.transform_train = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        input_size, scale=crop_scale, interpolation=3
                    ),  # 3 is bicubic
                    transforms.RandomHorizontalFlip(),
                ]
            )
        self.final_transform = transforms.Compose(
            [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.to_tensor = transforms.ToImage()

        root_original = root
        root = root + f'/karpathy_{split}/' # for 'karpathy_<dataset>' folder structure
        folder_path = root.split('/karpathy')[0]

        assert os.path.exists(folder_path + '/dataset_coco_karpathy.json'), f'File "dataset_coco_karpathy.json"\
        not found in the folder {folder_path}!'

        with open(folder_path + '/dataset_coco_karpathy.json') as f:
            raw_meta_data = json.load(f)['images']

        assert root[0]=='/', 'root should be an absolute path'
        self.meta_data = []
        for meta_for_example in raw_meta_data:
            if meta_for_example['split'] == split:
                # convert the absolute path for reading
                meta_for_example['full_path'] = root + meta_for_example['filename']
                self.meta_data.append(meta_for_example)

        logger.info(
            f"initial transform: {self.transform_train}, "
            f"jitter transform: {self.transform_jitter}, "
            f"final transform: {self.final_transform}"
        )
        logger.info(f"loaded {len(self.meta_data)} examples")

        self.is_compute_mask = compute_mask
        self.patches = (input_size // patch_size) ** 2
        self.mask_prob = mask_prob
        self.mask_prob_adjust = mask_prob_adjust
        self.mask_length = mask_length
        self.inverse_mask = inverse_mask
        self.expand_adjacent = expand_adjacent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping
        self.require_same_masks = require_same_masks
        self.clone_batch = clone_batch

        encoder_json_path = os.path.join(root_original, "encoder.json")
        vocab_bpe_path = os.path.join(root_original, "vocab.bpe")
        self.bpe_encoder = BPEEncoder(encoder_json_path, vocab_bpe_path)
        self.dictionary = Dictionary.load(os.path.join(root_original, "dict.txt"))
        self.tokens_per_sample = tokens_per_sample

    def __getitem__(self, index):
        path = self.meta_data[index]['full_path']

        if self.loader is not None:
            img = self.loader(path)
        else:
            img = Image.open(path).convert("RGB")

        captions = self.meta_data[index]['sentences']

        img = self.transform_train(img)


        v = {"id": index, self.key: img}
        v["captions"] = captions

        if self.is_compute_mask:
            if self.mask_length == 1:
                mask = data_utils.compute_block_mask_1d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                )
            else:
                mask = data_utils.compute_block_mask_2d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                    expand_adjcent=self.expand_adjacent,
                    mask_dropout=self.mask_dropout,
                    non_overlapping=self.non_overlapping,
                )

            v["precomputed_mask"] = mask

        return v

    def __len__(self):
        return len(self.meta_data)

    def collater(self, samples):
        n_samples = len(samples)
        if n_samples == 0:
            return {}

        collated_img = torch.stack([self.to_tensor(s[self.key]) for s in samples], dim=0)

        collated_img = self.transform_train(collated_img)

        if self.transform_jitter is not None:
            collated_img = self.final_transform(self.transform_jitter(collated_img))
        else:
            collated_img = self.final_transform(collated_img)

        caption_idx = torch.randint(low=0, high=5, size=(n_samples,))

        text = [samples[i]['captions'][caption_idx[i]]['raw'] for i in range(n_samples)]
        text_encoded = self.bpe_encoder.encode_lines(text, tokens_per_sample=self.tokens_per_sample-1) # one less for beginning-of-sentence token
        # text_encoded is now a list of tensors, each tensor is a list of token ids

        # padding
        text_encoded = data_utils.collate_tokens(text_encoded, self.dictionary.pad(), left_pad=False)
        
        bos_tensor = torch.full((n_samples, 1), self.dictionary.bos())

        text_encoded = torch.cat([bos_tensor, text_encoded], dim=1) # add beginning-of-sentence token

        padding_mask = text_encoded == self.dictionary.pad()

        switch = torch.randint(low=0, high=2, size=(1,)).bool().item() # 2 is exclusive

        
        if switch:
            net_input = {
                "teacher": text_encoded,
                "student": collated_img,
                "padding_mask": padding_mask,
                "teacher_caption": switch,
        }
        else:
            net_input = {
                "teacher": collated_img,
                "student": text_encoded,
                "padding_mask": padding_mask,
                "teacher_caption": switch,
        }
        if self.is_compute_mask:
            collated_mask = torch.cat([s["precomputed_mask"] for s in samples], dim=0)
            net_input["precomputed_mask"] = collated_mask

        res = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        return res

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