import os
import logging
import torch
import random
import time
import json
from typing import *
import subprocess
import shutil
import numpy as np
from shutil import copyfile
from functools import partial
from data_utils import get_transforms, curl_dataset, _write_data_into_jsonl
from bpe_encoder import BPEEncoder
from utils.wav2vec_manifest import create_manifests
from fairseq.data import (
    Dictionary,
    NestedDictionaryDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel
from fairseq.examples.data2vec.data import PathDataset
from fairseq.data import FairseqDataset
from fairseq.data.data_utils import compute_block_mask_1d, compute_block_mask_2d
from bpe_encoder import encode
from torchaudio.datasets import LIBRISPEECH, SPEECHCOMMANDS
from torchvision.transforms import v2 as transforms
from torchvision import datasets
import torchtext
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

logger = logging.getLogger(__name__)

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:str, split:str):
        self.data_path = data_path
        self.split = split

    def load(self):
        raise NotImplementedError
    
    def __len__(self):
        return len(self.items)

    def collater(self, samples):
        batch_tensors = {}
        for tensor_key in samples[0]:
            if isinstance(samples[0][tensor_key], torch.Tensor):
                batch_tensors[tensor_key] = torch.stack([d[tensor_key] for d in samples])
            else:
                batch_tensors[tensor_key] = torch.tensor([d[tensor_key] for d in samples], dtype=torch.long)

        return batch_tensors

class NLPDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int,
            sample_break_mode:str='none',):
        super().__init__(data_path, split)
        self.nlp_dir_path = os.path.join(self.data_path, 'language')
        os.makedirs(self.nlp_dir_path, exist_ok=True)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        self.sample_break_mode = sample_break_mode

          
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

        self.dataset = NestedDictionaryDataset(
            {
                "net_input": input_dict,
            },
        )

class EnWik9Dataset(NLPDataset):
    def __init__(
            self,
            data_path:str,
            batch_size:int,
            num_workers:int,
            num_max_bpe_tokens:int,
            sample_break_mode:str='none',):
        super().__init__(data_path, batch_size, num_workers, num_max_bpe_tokens)
        self.sample_break_mode = sample_break_mode

    def prepare_data(self):
        dataset_path = os.path.join(self.nlp_dir_path, 'enwik9')
        os.makedirs(dataset_path, exist_ok=True)
        name = curl_dataset("http://mattmahoney.net/dc/enwik9.zip")
        os.system(f"unzip {name}")
        os.remove(name)
        os.system(f"perl ../setup/clean_enwik9.pl enwik9 > enwik9.txt")
        os.remove("enwik9")
        encode(f'{self.data_path}/encoder.json', f'{self.data_path}/vocab.bpe', ['enwik9.txt'], ['enwik9.bpe'], keep_empty=True)
        os.remove("enwik9.txt")
        process = ['fairseq-preprocess', '--only-source', '--srcdict', f'{self.data_path}/dict.txt',
                    '--trainpref', 'enwik9.bpe', '--destdir', f'{dataset_path}', '--workers', f'{self.num_workers}']
        subprocess.run(process)
        os.remove("enwik9.bpe")

    def setup(self, stage):
        self._load_dataset('train', self.sample_break_mode)


class IMDBDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int,):
        super().__init__(data_path, split)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = os.path.join(self.data_path, 'language', 'imdb')
        self.out_jsonl_path = os.path.join(self.path_to_data, f'{self.split}.jsonl')

        encoder_json_path = os.path.join(self.data_path, 'encoder.json')
        vocab_bpe_path = os.path.join(self.data_path, 'vocab.bpe')
        dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        bpe_encoder = BPEEncoder(encoder_json_path, vocab_bpe_path)

        bos_token_id = dictionary.bos()
        eos_token_id = dictionary.eos()
        pad_token_id = dictionary.pad()
                
        os.makedirs(self.path_to_data, exist_ok=True)
        
        if os.path.exists(self.out_jsonl_path):
            print(f'Data already exists. Skip creating it.')
            return
        
        items = []
        data_loader = iter(torchtext.datasets.IMDB(root=self.path_to_data, split=self.split))
        for label, text in data_loader:
            tokens = bpe_encoder.encode(text)
            if len(tokens) > self.num_max_bpe_tokens - 2:
                tokens = tokens[:self.num_max_bpe_tokens - 2]
            tokens = [bos_token_id] + tokens + [eos_token_id]
            num_tokens = len(tokens)
            padding_mask = [0] * num_tokens + [1] * (self.num_max_bpe_tokens - num_tokens)
            language_tokens =  tokens + [pad_token_id] * (self.num_max_bpe_tokens - num_tokens)
            items.append({'language_tokens': language_tokens, 'padding_mask': padding_mask, 'label': label})

        _write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(f'{self.path_to_data}/datasets')
                
    def load(self):
        items = []
        with open(self.out_jsonl_path, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            print("Load %d text examples." % len(items))
        self.items = items

    def __getitem__(self, index):
        item = self.items[index]
        return item

    
class AudioDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            sample_rate:int,
            max_sample_size:int,
            min_sample_size:int,
            precompute_mask_config
            ):
        super().__init__(data_path, split)
        self.sample_rate = sample_rate
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.precompute_mask_config = precompute_mask_config


class LibriSpeechDataset(AudioDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            sample_rate:int,
            max_sample_size:int,
            min_sample_size:int,
            precompute_mask_config,
            type:str,
            ):
        super().__init__(data_path, split, sample_rate, max_sample_size, min_sample_size, precompute_mask_config)
        
        os.makedirs(self.data_path, exist_ok=True)

        LIBRISPEECH(root=self.data_path, url=type, download=True)

        tar_file_path = os.path.join(self.data_path, f"{type}.tar.gz")
        if os.path.exists(tar_file_path):
            os.remove(tar_file_path)

        self.manifest_path = os.path.join(self.data_path, 'LibriSpeech', type)
    
        create_manifests(root=self.manifest_path, valid_percent=0, dest=self.manifest_path)
        

    def load(self):
        manifest_path = os.path.join(self.manifest_path, "{}.tsv".format('train'))

        compute_mask = self.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = self.precompute_mask_config    

        self.dataset = FileAudioDataset(
            manifest_path=manifest_path,
            sample_rate=self.sample_rate,
            max_sample_size=self.max_sample_size,
            min_sample_size=self.min_sample_size,
            pad=False,
            normalize=True,
            num_buckets=0,
            text_compression_level=TextCompressionLevel.none,
            compute_mask=compute_mask,
            **mask_args,
        )

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples):
        collater_res = self.dataset.collater(samples)
        res = {
            'id': collater_res['id'],
            'audio': collater_res['net_input']['source'],
            'precomputed_mask': collater_res['net_input']['precomputed_mask'],
        }
        return res


class SpeechCommandsDataset(BaseDataset):
    def __init__(self, 
                 data_path:str,
                 split:str,
                 feature_encoder_spec:str,
                 ):
        super().__init__(data_path, split)

        self.feature_encoder_spec = feature_encoder_spec
        self.pad = True
        
        # as desribed in the paper to the dataset, each sample is at a maximum of 1 second
        # long and is sampled at 16kHz (https://arxiv.org/pdf/1804.03209.pdf)
        self.max_sample_size = 16_000

        path_to_data = os.path.join(self.data_path, 'SpeechCommands', 'speech_commands_v0.02')
        # List all entries in the given path
        all_entries = os.listdir(path_to_data)
        # Filter out directories that do not start with '_'
        class_names = [entry for entry in all_entries if os.path.isdir(os.path.join(path_to_data, entry)) and not entry.startswith('_')]
        
        self.class_to_id = {class_name: i for i, class_name in enumerate(class_names)}

        if self.split == "train":
            self.subset = "training"
        else:
            self.subset = "testing"

        SPEECHCOMMANDS(self.data_path, subset=self.subset, download=True)

        if os.path.exists(f"{self.data_path}/speech_commands_v0.02.tar.gz"):
            os.remove(f"{self.data_path}/speech_commands_v0.02.tar.gz")

    def load(self):
        self.items = SPEECHCOMMANDS(self.data_path, subset=self.subset)

    def __getitem__(self, index):
        item = self.items[index]
        return {"audio": item[0][0], "label": item[2], "id": index}
    
    def collater(self, samples):
        samples = [s for s in samples if s["audio"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["audio"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self._crop_to_max_size(source, target_size)

        input = {"audio": collated_sources,
                 "label": torch.LongTensor([self.class_to_id[s["label"]] for s in samples]),
                 "id": torch.LongTensor([s["id"] for s in samples])
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
        L_in = size
        for (_, kernel_size, stride) in self.feature_encoder_spec:
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

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
    

    
class ImageDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split,
            beit_transforms,
            no_transform,
            transform_jitter,
            precompute_mask_config,
            crop_scale,
            dataset_type,
            local_cache_path):
        super().__init__(data_path, split)
        self.beit_transforms = beit_transforms
        self.no_transform = no_transform
        self.transform_jitter = transform_jitter
        self.precompute_mask_config = precompute_mask_config
        self.crop_scale = crop_scale
        self.dataset_type = dataset_type
        self.local_cache_path = local_cache_path


class ImageNetDataset(ImageDataset):
    def __init__(
            self,
            data_path:str,
            split,
            beit_transforms,
            no_transform,
            transform_jitter,
            precompute_mask_config,
            crop_scale,
            dataset_type,
            local_cache_path):
        super().__init__(data_path, split, beit_transforms, no_transform,
                         transform_jitter, precompute_mask_config,
                         crop_scale, dataset_type, local_cache_path)
        self.path_to_data = os.path.join(self.data_path, 'imagenet')
        os.makedirs(self.path_to_data, exist_ok=True)
        name = "imagenet-object-localization-challenge"
        command = ["kaggle", "competitions", "download", "-c", name]
        subprocess.run(command)
        os.system(f"unzip {name}.zip")
        os.remove(f"{name}.zip")
        os.remove("LOC*")
        os.system(f"mv ILSVRC/Data/CLS-LOC/* {self.path_to_data}")
        os.system(f"mv ILSVRC/ImageSets/CLS-LOC/* {self.path_to_data}")
        os.system(f"rm -r ILSVRC")

    def load(self):
        compute_mask = self.precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = self.precompute_mask_config
        
        self.dataset = MaeImageDataset(
            root=self.path_to_data,
            split=self.split,
            input_size=(244, 244),
            local_cache_path=self.local_cache_path,
            key='image',
            beit_transforms=self.beit_transforms,
            no_transform=self.no_transform,
            transform_jitter=self.transform_jitter,
            crop_scale=self.crop_scale,
            compute_mask=compute_mask,
            dataset_type=self.dataset_type,
            **mask_args,
        )

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples):
        return self.dataset.collater(samples)
    
class CIFARDataset(BaseDataset):
    def __init__(self, 
                 data_path:str,
                 split:str,
                 type:str="cifar10",
                 ):
        super().__init__(data_path, split)
        self.type = type

        if self.type == "cifar10":
            CIFAR10(self.data_path, train=self.split == "train", download=True)
        else:
            CIFAR100(self.data_path, train=self.split == "train", download=True)

    def load(self):
        transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.Resize((224, 224)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        if self.type == "cifar10":
            self.items = CIFAR10(self.data_path, train=self.split == "train", transform=transform)
        else:
            self.items = CIFAR100(self.data_path, train=self.split == "train", transform=transform)

    def __getitem__(self, index):
        item = self.items[index]
        return {"image": item[0], "target": item[1]}


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

        self.transform = get_transforms(no_transform=no_transform,
                                        beit_transforms=beit_transforms,
                                        transform_jitter=transform_jitter,
                                        crop_scale=crop_scale)

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
