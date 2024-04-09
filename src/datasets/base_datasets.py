import os
import logging
import torch
import random
import time
import json
from typing import *
import numpy as np
from shutil import copyfile
from functools import partial
import soundfile as sf
from enum import Enum, auto
from .data_utils import get_transforms 
from .data_utils import get_bpe_encoder as get_bpe_encoder_from_utils
import torch.nn.functional as F

from examples.data2vec.data import PathDataset
from fairseq.data import FairseqDataset, data_utils
from fairseq.data.data_utils import compute_block_mask_1d, compute_block_mask_2d

from fairseq.data import (
    Dictionary,
    NestedDictionaryDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    TokenBlockDataset,
)

from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets import ImageFolder

logger = logging.getLogger(__name__)

class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, split:str):
        self.data_path = data_path
        self.split = split

    def get_bpe_encoder(self):
        return get_bpe_encoder_from_utils(self.data_path)

    def load(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.items)
    
    @property
    def modes(self) -> List[Modality]:
        raise NotImplementedError

    def collater(self, samples):
        batch_tensors = {}
        for tensor_key in samples[0]:
            if isinstance(samples[0][tensor_key], torch.Tensor):
                batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
            else:
                batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

        batch_tensors['modes'] = self.modes
        return batch_tensors
    
    def log(self, msg:str):
        logger.info(f"[{self.__class__.__name__}]: {msg}")
    

class NLPDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int,
            sample_break_mode:str='none',):
        super().__init__(data_path=data_path, 
                         split=split)
        self.nlp_dir_path = os.path.join(self.data_path, 'language')
        os.makedirs(self.nlp_dir_path, exist_ok=True)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        self.sample_break_mode = sample_break_mode

    @property
    def modes(self) -> List[Modality]:
        return [Modality.TEXT]

    def index_exists(self, dataset_path):
        if os.path.exists(os.path.join(dataset_path, 'train.bin')) and os.path.exists(os.path.join(dataset_path, 'train.idx')):
            self.log(f"Data already exists under: {dataset_path}")
            return True
        else:
            return False
          
    def load(self):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.data_path, self.split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            combine=True,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(self.split, split_path)
            )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.num_max_bpe_tokens - 2,  # two less for bos and eos
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence and append end-of-sentence tokens
        dataset = PrependTokenDataset(dataset, self.dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.dictionary.eos())

        input_dict = {
            "language_tokens": RightPadDataset(
                dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "padding_mask": RightPaddingMaskDataset(dataset),
        }

        self.dataset = NestedDictionaryDataset(input_dict)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples):
        return self.dataset.collater(samples)
    

class AudioDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            sample_rate:int=16_000,
            max_sample_size:int=320_000,
            min_sample_size:int=0,
            normalize:bool=True,
            pad:bool=True,
            feature_encoder_spec=[],
            ):
        super().__init__(data_path=data_path,
                         split=split)
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.normalize = normalize
        self.pad = pad
        self.feature_encoder_spec = feature_encoder_spec
        self._features_size_map = {}

    @property
    def modes(self) -> List[Modality]:
        raise [Modality.AUDIO]

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        audios = [s["audio"] for s in samples]
        sizes = [len(s) for s in audios]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_audio = audios[0].new_zeros(len(audios), target_size)
        padding_mask = (
            torch.BoolTensor(collated_audio.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(audios, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_audio[i] = source
            elif diff < 0:
                assert self.pad
                collated_audio[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_audio[i] = self._crop_to_max_size(source, target_size)

        input = {
            "audio": collated_audio,
            #"id": torch.LongTensor([s["id"] for s in samples]),
            }
        
        if self.pad:
            input["padding_mask"] = padding_mask

        if "precomputed_mask" in samples[0]:
            target_size = self._get_mask_indices_dims(target_size)
            collated_mask = torch.cat(
                [
                    self._crop_to_max_size(s["precomputed_mask"], target_size, dim=1)
                    for s in samples
                ],
                dim=0,
            )
            input["precomputed_mask"] = collated_mask

        return input

    def _get_mask_indices_dims(self, size, padding=0, dilation=1):
        if size not in self.feature_encoder_spec:
            L_in = size
            for (_, kernel_size, stride) in self.feature_encoder_spec:
                L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
                L_out = 1 + L_out // stride
                L_in = L_out
            self._features_size_map[size] = L_out
        return self._features_size_map[size]

    def _crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        slices = []
        for d in range(dim):
            slices.append(slice(None))
        slices.append(slice(start, end))

        return t[slices]
    
    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats
    

class ImageDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split,
            beit_transforms,
            no_transform,
            transform_jitter,
            crop_scale,
            dataset_type,
            local_cache_path,
            precompute_mask_config,):
        super().__init__(data_path=data_path,
                         split=split)
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.transform_jitter = transform_jitter
        self.crop_scale = crop_scale
        self.dataset_type = dataset_type
        self.local_cache_path = local_cache_path
        self.precompute_mask_config = precompute_mask_config

    @property
    def modes(self) -> List[Modality]:
        return [Modality.IMAGE]
    

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

# adapted from: https://github.com/facebookresearch/fairseq/blob/34973a94d09ecc12092a5ecc8afece5e536b7692/examples/data2vec/data/mae_image_dataset.py
class MaeImageDataset(FairseqDataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size,
        local_cache_path=None,
        shuffle=True,
        key="image",
        beit_transforms=False,
        no_transform=False,
        transform_jitter=False,
        crop_scale=(0.6, 1.0),
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
        dataset_type: str = "imagefolder",
    ):
        FairseqDataset.__init__(self)

        self.shuffle = shuffle
        self.key = key

        loader = caching_loader(local_cache_path, datasets.folder.default_loader)

        if split == "train":
            self.transform = get_transforms(no_transform=no_transform,
                                            beit_transforms=beit_transforms,
                                            transform_jitter=transform_jitter,
                                            crop_scale=crop_scale)
        else:
            self.transform = get_transforms(no_transform=True, beit_transforms=False, transform_jitter=False)

        if dataset_type == "imagefolder":
            self.dataset = ImageFolder(
                os.path.join(root, split), loader=loader
            )
        elif dataset_type == "path":
            self.dataset = PathDataset(
                root,
                loader,
                None,
                None,
            )
        else:
            raise Exception(f"invalid dataset type {dataset_type}")

        logger.info(
            f"transform: {self.transform}"
        )
        logger.info(f"loaded {len(self.dataset)} examples")

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
        img, _ = self.dataset[index]

        img = self.transform(img)

        v = {"id": index, self.key: img}

        if self.is_compute_mask:
            if self.mask_length == 1:
                mask = compute_block_mask_1d(
                    shape=(self.clone_batch, self.patches),
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    mask_prob_adjust=self.mask_prob_adjust,
                    inverse_mask=self.inverse_mask,
                    require_same_masks=True,
                )
            else:
                mask = compute_block_mask_2d(
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
        return len(self.dataset)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        collated_img = torch.stack([s[self.key] for s in samples], dim=0)

        res = {
            # "id": torch.LongTensor([s["id"] for s in samples]),
            self.key: collated_img,
        }

        if "target" in samples[0]:
            collated_target = torch.stack([s["target"] for s in samples], dim=0)
            res["label"] = collated_target

        if "precomputed_mask" in samples[0]:
            collated_mask = torch.cat([s["precomputed_mask"] for s in samples], dim=0)
            res["precomputed_mask"] = collated_mask

        return res
    

class BaseImageText(BaseDataset):
    def __init__(
        self,
        data_path,
        split,
        num_max_bpe_tokens,
        transform_jitter=False,
        beit_transforms=False,
        no_transform=False,
        crop_scale=(0.6, 1.0),
    ):
        super().__init__(data_path=data_path, 
                         split=split)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale
        self.path_to_data = None

        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))

        self.bos_token_id = self.dictionary.bos()
        self.eos_token_id = self.dictionary.eos()
        self.pad_token_id = self.dictionary.pad()
        self.loader = default_loader
        self.transform = get_transforms(no_transform=self.no_transform,
                                        beit_transforms=self.beit_transforms,
                                        transform_jitter=self.transform_jitter,
                                        crop_scale=self.crop_scale)
        
    @property
    def modes(self) -> List[Modality]:
        return [Modality.IMAGE, Modality.TEXT]
        
    def index_exists(self, dataset_path):
        for index_file in self.get_index_files():
            if not os.path.exists(dataset_path, index_file):
                return False
        self.log(f"Data already exists under: {dataset_path}")
        return True

    def load(self):
        index_files = self.get_index_files()
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(self.path_to_data, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                self.log("Load %d image-text pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items

    def get_index_files(self):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)

    def _get_text_segment(self, text_segment, max_len=None):
        assert isinstance(text_segment, list)
        tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["padding_mask"] = padding_mask

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_text_example(index, data)
        return data

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = '{' + "\n  Number of items: %s," % self.__len__()
        body += "\n  data root = %s," % self.data_path
        body += "\n  split = %s," % self.split
        body += "\n  dataset index files = %s" % str(self.index_files)
        body += "\n  num max bpe tokens = %s" % self.num_max_bpe_tokens
        body += "\n  transforms = ["
        for t in self.transform:
            body += "\n    %s" % str(t)
        body += "\n  ]"
        body += "\n}"

        return head + body
    

class BaseImageAudio(AudioDataset):
    def __init__(
        self,
        data_path,
        split,
        transform_jitter=False,
        beit_transforms=False,
        no_transform=True,
        crop_scale=(0.6, 1.0),
        sample_rate:int=16_000,
        max_sample_size:int=320_000,
        min_sample_size:int=32_000,
        normalize:bool=True,
        pad:bool=True,
        precompute_mask_config:Dict[str, Any]={},
    ):
        super().__init__(data_path=data_path, 
                         split=split, 
                         sample_rate=sample_rate, 
                         max_sample_size=max_sample_size, 
                         min_sample_size=min_sample_size, 
                         normalize=normalize, 
                         pad=pad,
                         **precompute_mask_config)
        self.transform_jitter = transform_jitter
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.crop_scale = crop_scale
        self.path_to_data = None

        self.loader = default_loader
        self.transform = get_transforms(no_transform=self.no_transform,
                                        beit_transforms=self.beit_transforms,
                                        transform_jitter=self.transform_jitter,
                                        crop_scale=self.crop_scale)
        
    @property
    def modes(self) -> List[Modality]:
        return [Modality.IMAGE, Modality.AUDIO]
        
    def index_exists(self, dataset_path):
        for index_file in self.get_index_files():
            if not os.path.exists(dataset_path, index_file):
                return False
        self.log(f"Data already exists under: {dataset_path}")
        return True

    def load(self):
        index_files = self.get_index_files()
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(self.path_to_data, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                self.log("Load %d image-audio pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items

    def get_index_files(self):
        raise NotImplementedError()

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)
    
    def _get_audio(self, audio_path: str):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio).float()
        return self.postprocess(audio, sample_rate)

    def _get_image_audio_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        audio_path = item["audio_path"]
        data['audio'] = self._get_audio(audio_path)

    def __getitem__(self, index: int):
        data = dict()
        self._get_image_audio_example(index, data)
        return data
    
    def collater(self, samples):
        input = super().collater(samples)
        input["image"] = torch.stack([s["image"] for s in samples], dim=0)
        return input

class BaseTextAudio(AudioDataset):
    def __init__(
        self,
        data_path,
        split,
        num_max_bpe_tokens,
        sample_rate:int=16_000,
        max_sample_size:int=320_000,
        min_sample_size:int=32_000,
        normalize:bool=True,
        pad:bool=True,
        precompute_mask_config:Dict[str, Any]={},
    ):
        super().__init__(data_path=data_path, 
                         split=split, 
                         sample_rate=sample_rate, 
                         max_sample_size=max_sample_size, 
                         min_sample_size=min_sample_size, 
                         normalize=normalize, 
                         pad=pad,
                         **precompute_mask_config)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = None

        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))

        self.bos_token_id = self.dictionary.bos()
        self.eos_token_id = self.dictionary.eos()
        self.pad_token_id = self.dictionary.pad()

    @property
    def modes(self) -> List[Modality]:
        return [Modality.TEXT, Modality.AUDIO]

    def index_exists(self, dataset_path):
        for index_file in self.get_index_files():
            if not os.path.exists(dataset_path, index_file):
                return False
        self.log(f"Data already exists under: {dataset_path}")
        return True

    def load(self):
        index_files = self.get_index_files()
        items = []
        self.index_files = index_files

        offset = 0
        for _index_file in index_files:
            index_file = os.path.join(self.path_to_data, _index_file)
            with open(index_file, mode="r", encoding="utf-8") as reader:
                for line in reader:
                    data = json.loads(line)
                    items.append(data)
                self.log("Load %d text-audio pairs from %s. " % (len(items) - offset, index_file))
                offset = len(items)
        self.items = items

    def get_index_files(self):
        raise NotImplementedError()

    def _get_text_segment(self, text_segment, max_len=None):
        assert isinstance(text_segment, list)
        tokens = text_segment[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        tokens = [self.bos_token_id] + tokens[:] + [self.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens
    
    def _get_audio(self, audio_path: str):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio).float()
        return self.postprocess(audio, sample_rate)

    def _get_text_audio_example(self, index: int, data: dict):
        item = self.items[index]
        text_segment = item["text_segment"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["language_tokens"] = language_tokens
        data["language_padding_mask"] = padding_mask

        audio_path = item["audio_path"]
        data['audio'] = self._get_audio(audio_path)

    def __getitem__(self, index: int):
        data = dict()
        self._get_text_audio_example(index, data)
        return data
    
    def collater(self, samples):
        input = super().collater(samples)
        
        # language data must be collated seperately, as super().collater only for audio (superclass in "AudioDataset")
        for key in ["language_tokens", "language_padding_mask"]:
            if isinstance(samples[0][key], torch.Tensor):
                input[key] = torch.stack([d[key] for d in samples], dim=0)
            else:
                input[key] = torch.tensor([d[key] for d in samples])

        return input