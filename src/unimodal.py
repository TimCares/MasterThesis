import os
import logging
import torch
from torchvision import datasets
import random
import time
from shutil import copyfile
from functools import partial
from data_utils import get_transforms
from torch.utils.data import Dataset, DataLoader
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

logger = logging.getLogger(__name__)

class BaseDataset(Dataset):
    def get_dataloader(self, split:str='train', **kwargs) -> DataLoader:
        pass


class NLPDataset(BaseDataset):
    def __init__(
            self,
            data_path,
            num_max_bpe_tokens,
            sample_break_mode:str='none',):
        
        self.data_path = data_path
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))

        self._load_dataset('train', sample_break_mode)
          
    def _load_dataset(self, split, sample_break_mode):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            combine=True,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.num_max_bpe_tokens - 2,  # two less for bos and eos
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=sample_break_mode,
        )
        print("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence and append end-of-sentence tokens
        dataset = PrependTokenDataset(dataset, self.dictionary.bos())
        dataset = AppendTokenDataset(dataset, self.dictionary.eos())

        input_dict = {
            "source": RightPadDataset(
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

    def get_dataloader(self, split:str='train', **kwargs) -> DataLoader:
        return DataLoader(self.dataset, collate_fn=self.dataset.collater, **kwargs)


class AudioDataset(BaseDataset):
    def __init__(
            self,
            data_path,
            sample_rate,
            max_sample_size,
            min_sample_size,
            precompute_mask_config
            ):
        
        self.data_path = data_path
        manifest_path = os.path.join(data_path, "{}.tsv".format('train'))

        compute_mask = precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = precompute_mask_config    

        self.dataset = FileAudioDataset(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            pad=False,
            normalize=True,
            num_buckets=0,
            text_compression_level=TextCompressionLevel.none,
            compute_mask=compute_mask,
            **mask_args,
        )

    def get_dataloader(self, split:str='train', **kwargs) -> DataLoader:
        return DataLoader(self.dataset, collate_fn=self.dataset.collater, **kwargs)
    
class ImageDataset(BaseDataset):
    def __init__(
            self,
            data_path,
            split,
            beit_transforms,
            no_transform,
            transform_jitter,
            precompute_mask_config,
            crop_scale,
            dataset_type,
            local_cache_path):
        
        compute_mask = precompute_mask_config is not None
        mask_args = {}
        if compute_mask:
            mask_args = precompute_mask_config  
        
        self.dataset = MaeImageDataset(
                root=data_path,
                split=split,
                input_size=(244, 244),
                local_cache_path=local_cache_path,
                key='image',
                beit_transforms=beit_transforms,
                no_transform=no_transform,
                transform_jitter=transform_jitter,
                crop_scale=crop_scale,
                compute_mask=compute_mask,
                dataset_type=dataset_type,
                **mask_args,
        )

    def get_dataloader(self, split:str='train', **kwargs) -> DataLoader:
        return DataLoader(self.dataset, collate_fn=self.dataset.collater, **kwargs)


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
        key="imgs",
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
            self.dataset = datasets.ImageFolder(
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

        v = {"id": index, 'img': img}

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
