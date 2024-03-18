import logging
import sys
import os
from typing import Optional, List
from dataclasses import dataclass, field
from omegaconf import MISSING, II
import numpy as np

from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.examples.data2vec.tasks.mae_image_pretraining import MaeImageDataset
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    AppendTokenDataset,
    RightPadDataset,
    RightPaddingMaskDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq import utils


logger = logging.getLogger(__name__)


@dataclass
class ImageMaskingConfig:
    patch_size: int = II("model.modalities.image.patch_size")
    mask_prob: float = II("model.modalities.image.mask_prob")
    mask_prob_adjust: float = II("model.modalities.image.mask_prob_adjust")
    mask_length: int = II("model.modalities.image.mask_length")
    inverse_mask: bool = II("model.modalities.image.inverse_mask")
    mask_dropout: float = II("model.modalities.image.mask_dropout")
    clone_batch: int = II("model.clone_batch")
    expand_adjacent: bool = False
    non_overlapping: bool = False


@dataclass
class MultimodalPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    multi_data: Optional[List[str]] = None
    input_size: int = 224
    local_cache_path: Optional[str] = None
    key: str = "imgs"

    beit_transforms: bool = False
    target_transform: bool = False
    no_transform: bool = False

    rebuild_batches: bool = True

    precompute_mask_config: Optional[ImageMaskingConfig] = None

    subsample: float = 1
    seed: int = II("common.seed")
    dataset_type: str = "imagefolder"


@register_task("multi_modal_pretraining", dataclass=MultimodalPretrainingConfig)
class MultimodalPretrainingTask(FairseqTask):
    """ 
    Task for multimodal pretraining of data2vec in a round-robin fashion.
    """

    cfg: MultimodalPretrainingConfig

    @classmethod
    def setup_task(cls, cfg: MultimodalPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (AudioPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        cfg = task_cfg or self.cfg
        
        image_dataset = MaeImageDataset(
            root=data_path if cfg.multi_data is None else cfg.multi_data,
            split=split,
            input_size=cfg.input_size,
            local_cache_path=cfg.local_cache_path,
            key=cfg.key,
            beit_transforms=cfg.beit_transforms,
            target_transform=cfg.target_transform,
            no_transform=cfg.no_transform,
            compute_mask=compute_mask,
            dataset_type=cfg.dataset_type,
            **mask_args,
        )
        text_dataset = None
        audio_dataset = None

        self.datasets[split] = RoundRobinZipDatasets([('vision', image_dataset), ('text', text_dataset), ('audio', audio_dataset)])

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize
    
    def _get_masked_lm_dataset(self, split, epoch, combine):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.source_dictionary,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError(
                "Dataset not found: {} ({})".format(split, split_path)
            )

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self.cfg.tokens_per_sample,
            self.cfg.seed,
        )

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.cfg.tokens_per_sample - 1,  # one less for <s>
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode=self.cfg.sample_break_mode,
        )
        logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        dataset = AppendTokenDataset(dataset, self.source_dictionary.eos())

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(dataset))


        input_dict = {
            "source": RightPadDataset(
                dataset,
                pad_idx=self.source_dictionary.pad(),
            ),
            "id": IdDataset(),
            "padding_mask": RightPaddingMaskDataset(dataset),
        }

        dataset_nested = NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": input_dict,
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(dataset, reduce=True),
            },
            sizes=[dataset.sizes],
        )

        return SortDataset(
            dataset_nested, sort_order=[shuffle, dataset.sizes]
        )