import os
import logging
import torch
import json
from typing import *
import numpy as np
import soundfile as sf
from .data_utils import get_transforms 
from bpe_encoder import get_bpe_encoder as get_bpe_encoder_from_utils
import torch.nn.functional as F

from fairseq.data.data_utils import compute_block_mask_1d, compute_block_mask_2d, load_indexed_dataset
from data2vec_fairseq.data.modality import Modality
from utils import pad_text_sequence

from fairseq.data import (
    Dictionary,
    NestedDictionaryDataset,
    TokenBlockDataset,
    PrependTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    IdDataset,
)

from torchvision.datasets.folder import default_loader

logger = logging.getLogger(__name__)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, split:str, precompute_mask_config:Dict[str, Any]=None):
        self.data_path = data_path
        self.split = split
        self.precompute_mask_config = precompute_mask_config

    def get_bpe_encoder(self):
        return get_bpe_encoder_from_utils(self.data_path)

    def load(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.items)
    
    @property
    def modality(self) -> Modality:
        raise NotImplementedError

    def collater(self, samples):
        batch_tensors = {}
        for tensor_key in samples[0]:
            if isinstance(samples[0][tensor_key], torch.Tensor):
                batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
            else:
                batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

        batch_tensors['modality'] = self.modality
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
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        self.sample_break_mode = sample_break_mode

    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def index_exists(self, dataset_path):
        prefix = os.path.join(dataset_path, self.split)
        if os.path.exists(os.path.join(prefix, f'{self.split}.bin')) and os.path.exists(os.path.join(prefix, f'{self.split}.idx')):
            self.log(f"Data already exists under: {dataset_path}")
            return True
        else:
            return False
          
    def load(self):
        """
        Load a given dataset split.
        """
        split_path = os.path.join(self.data_path, self.split, self.split)

        dataset = load_indexed_dataset(
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
            self.num_max_bpe_tokens - 1,  # one less for bos
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence
        dataset = PrependTokenDataset(dataset, self.dictionary.bos())
        #dataset = AppendTokenDataset(dataset, self.dictionary.eos()) # -> not done in data2vec

        input_dict = {
            "x": RightPadDataset(
                dataset,
                pad_idx=self.dictionary.pad(),
            ),
            "padding_mask": RightPaddingMaskDataset(dataset),
            "id": IdDataset(),
        }

        self.dataset = NestedDictionaryDataset(input_dict)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples):
        batch = self.dataset.collater(samples)
        batch["modality"] = self.modality
        return batch
    

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
    def modality(self) -> Modality:
        return Modality.AUDIO

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
            "x": collated_audio,
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

        input['modality'] = self.modality

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
            pretraining,
            color_jitter=None,
            aa="rand-m9-mstd0.5-inc1",
            reprob=0.25,
            remode="pixel",
            recount=1,
            beit_transforms:bool=False,
            crop_scale:Tuple[float, float]=(0.08, 1.0),
            precompute_mask_config=None,):
        super().__init__(data_path=data_path,
                         split=split,
                         precompute_mask_config=precompute_mask_config)
        self.pretraining = pretraining
        self.color_jitter = color_jitter
        self.aa = aa
        self.reprob = reprob
        self.remode = remode
        self.recount = recount

        self.beit_transforms = beit_transforms
        self.crop_scale = crop_scale

        self.loader = default_loader

        self.transform = get_transforms(
            pretraining=self.pretraining,
            train=self.split=="train",
            color_jitter=self.color_jitter,
            aa=self.aa,
            reprob=self.reprob,
            remode=self.remode,
            recount=self.recount,
            beit_transforms=self.beit_transforms,
            crop_scale=self.crop_scale,
        )

    def _get_image(self, image_path: str):
        image = self.loader(image_path)
        return self.transform(image)

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE
    

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
        self.transform = get_transforms(no_transform=self.no_transform, # TODO: Do as in ImageDataset -> Inherit from it?
                                        beit_transforms=self.beit_transforms,
                                        transform_jitter=self.transform_jitter,
                                        crop_scale=self.crop_scale)
        
    @property
    def modality(self) -> Modality:
        return Modality.VL
        
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

        language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=max_len,
                                                          pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)

        return language_tokens, padding_mask, max_len # language_tokens, padding_mask are lists, converted to tensors in collater

    def _get_image_text_example(self, index: int, data: dict):
        item = self.items[index]
        img_path = item["image_path"]
        img = self._get_image(img_path)
        data["image"] = img

        text_segment = item["text"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["text"] = language_tokens
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
        compute_mask = precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = precompute_mask_config 
        super().__init__(data_path=data_path, 
                         split=split, 
                         sample_rate=sample_rate, 
                         max_sample_size=max_sample_size, 
                         min_sample_size=min_sample_size, 
                         normalize=normalize, 
                         pad=pad,
                         **mask_args)
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
    def modality(self) -> Modality:
        return Modality.VA
        
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
        input['modality'] = self.modality
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
        compute_mask = precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = precompute_mask_config 
        super().__init__(data_path=data_path, 
                         split=split, 
                         sample_rate=sample_rate, 
                         max_sample_size=max_sample_size, 
                         min_sample_size=min_sample_size, 
                         normalize=normalize, 
                         pad=pad,
                         **mask_args)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = None

        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))

        self.bos_token_id = self.dictionary.bos()
        self.eos_token_id = self.dictionary.eos()
        self.pad_token_id = self.dictionary.pad()

    @property
    def modality(self) -> Modality:
        return Modality.LA

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

        language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=max_len,
                                                          pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)
        
        return language_tokens, padding_mask, max_len # language_tokens, padding_mask are lists, converted to tensors in collater
    
    def _get_audio(self, audio_path: str):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        audio = torch.from_numpy(audio).float()
        return self.postprocess(audio, sample_rate)

    def _get_text_audio_example(self, index: int, data: dict):
        item = self.items[index]
        text_segment = item["text"]
        language_tokens, padding_mask, _ = self._get_text_segment(text_segment)
        data["text"] = language_tokens
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
        for key in ["text", "language_padding_mask"]:
            if isinstance(samples[0][key], torch.Tensor):
                input[key] = torch.stack([d[key] for d in samples], dim=0)
            else:
                input[key] = torch.tensor([d[key] for d in samples])

        input['padding_mask'] = {
            'audio': input['padding_mask'], # originates from super().collater
            'text': input['language_padding_mask']
        }
        input.pop('language_padding_mask')

        input['modality'] = self.modality

        return input
