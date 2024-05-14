import os
import logging
import torch
import json
from typing import *
from functools import partial
import shutil
import glob
from .data_utils import get_transforms, write_data_into_jsonl
from .base_datasets import BaseDataset
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
from .wav2vec_manifest import create_manifests
from bpe_encoder import encode, get_bpe_encoder
from utils import pad_text_sequence
import torch.nn as nn

from fairseq.data import Dictionary, ConcatDataset
from fairseq_cli.preprocess import preprocess

from torchaudio.datasets import LIBRISPEECH, SPEECHCOMMANDS
import torchtext
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import default_loader

from .base_datasets import AudioDataset, ImageDataset, NLPDataset
from data2vec_fairseq.data.modality import Modality
from .imagenet_classes import IMAGENET2012_CLASSES

logger = logging.getLogger(__name__)

class OpenWebTextDataset(NLPDataset):
    def __init__(self,
                 data_path: str,
                 split: str, # ignored, as only train for pretraining
                 num_max_bpe_tokens: int,
                 sample_break_mode: str = 'none'):
        super().__init__(data_path=data_path,
                         split=split,
                         num_max_bpe_tokens=num_max_bpe_tokens, 
                         sample_break_mode=sample_break_mode)
        dataset_path = os.path.join(self.data_path, 'openwebtext')
        base_data_path = self.data_path
        self.data_path = dataset_path

        if self.index_exists(dataset_path=dataset_path):
            return

        pattern = os.path.join(base_data_path, 'urlsf_*.tar')
        files = glob.glob(pattern)

        if len(files)==0:
            raise FileNotFoundError(f"No tar files found under: {dataset_path}")

        self.log(f"Found {len(files)} tar files, inflating...")
        for file in files:
            os.system(f"tar -xf {file} -C {base_data_path}")
            os.remove(file)
        pattern = os.path.join(dataset_path, '*.xz')
        files = glob.glob(pattern)
        for file in files:
            os.system(f"unxz {file}")
        self.log("Inflated all tar files.")

        self.log("Cleaning...")
        files = os.listdir(dataset_path)
        for file in files:
            with open(os.path.join(dataset_path, file), 'r+', encoding='utf-8', errors='ignore') as reader:
                lines = reader.readlines()
                reader.seek(0)
                reader.truncate()
                for line in lines:
                    line = line.strip()
                    if '\x00' not in line and line != '' and line != '---': # remove null bytes and empty lines
                        reader.write(line)
                        reader.write('\n')

        self.log("Joining...")
        with open(os.path.join(dataset_path, 'openwebtext.txt'), 'w', encoding='utf-8') as f:
            for file in files:
                path_to_file = os.path.join(dataset_path, file)
                with open(path_to_file, 'r', encoding='utf-8') as f2:
                    f.write(f2.read())
                os.remove(path_to_file)

        self.log("Encoding...")
        in_file = os.path.join(dataset_path, 'openwebtext.txt')
        out_file = os.path.join(dataset_path, 'openwebtext.bpe')
        encode(f'{base_data_path}/encoder.json', f'{base_data_path}/vocab.bpe', [in_file], [out_file], keep_empty=True)
        os.remove(in_file)
        preprocess(srcdict=f'{base_data_path}/dict.txt', trainpref=out_file, destdir=dataset_path, workers=os.cpu_count(),
                   only_source=True)
        os.remove(out_file)


class IMDBDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int,):
        super().__init__(data_path=data_path,
                         split=split)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = os.path.join(self.data_path, 'imdb')
        self.out_jsonl_path = os.path.join(self.path_to_data, f'{self.split}.jsonl')

        dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        bpe_encoder = get_bpe_encoder(self.data_path)

        bos_token_id = dictionary.bos()
        pad_token_id = dictionary.pad()
                
        os.makedirs(self.path_to_data, exist_ok=True)
        
        if os.path.exists(self.out_jsonl_path):
            self.log(f'Data already exists. Skip creating it.')
            return
        
        items = []
        data_loader = iter(torchtext.datasets.IMDB(root=self.path_to_data, split=self.split))
        for label, text in data_loader:
            tokens = bpe_encoder.encode(text)
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_max_bpe_tokens,
                                                              pad_idx=pad_token_id, bos_idx=bos_token_id)
            label = label-1 # 1 -> 0, 2 -> 1
            items.append({'text': language_tokens, 'padding_mask': padding_mask, 'target': label})

        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(f'{self.path_to_data}/datasets')

    @property
    def modes(self) -> List[Modality]:
        return [Modality.TEXT]
                
    def load(self):
        items = []
        with open(self.out_jsonl_path, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            self.log("Load %d text examples." % len(items))
        self.items = items

    def __getitem__(self, index):
        item = self.items[index]
        return item
    
class QQPDataset(BaseDataset):
    def __init__(
            self,
            data_path:str,
            num_max_bpe_tokens:int,):
        super().__init__(data_path=data_path,
                         split='train') # only one split available
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = os.path.join(self.data_path, 'qqp')
        self.out_jsonl_path = os.path.join(self.path_to_data, f'{self.split}.jsonl')
        self.n_pairs = 25_000

        dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        bpe_encoder = get_bpe_encoder(self.data_path)

        bos_token_id = dictionary.bos()
        pad_token_id = dictionary.pad()
                
        os.makedirs(self.path_to_data, exist_ok=True)
        
        if os.path.exists(self.out_jsonl_path):
            self.log(f'Data already exists. Skip creating it.')
            return
        
        items = []
        data_loader = iter(torchtext.datasets.QQP(root=self.path_to_data))
        pairs_collected = 0
        for target, text1, text2 in data_loader:
            if pairs_collected == self.n_pairs:
                break
            if target == 0: # only collect duplicated questions
                continue
            pair = dict()
            for i, text in enumerate([text1, text2]):
                tokens = bpe_encoder.encode(text)
                language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_max_bpe_tokens,
                                                                  pad_idx=pad_token_id, bos_idx=bos_token_id)
                pair[f'text{i}'] = language_tokens
                pair[f'padding_mask{i}'] = padding_mask
            
            items.append(pair)
            pairs_collected += 1

        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(f'{self.path_to_data}/datasets')

    @property
    def modes(self) -> List[Modality]:
        return [Modality.TEXT]
                
    def load(self):
        items = []
        with open(self.out_jsonl_path, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            self.log("Load %d text examples." % len(items))
        self.items = items

    def __getitem__(self, index):
        item = self.items[index]
        return item


class LibriSpeechDataset(AudioDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            sample_rate:int,
            max_sample_size:int,
            min_sample_size:int,
            normalize:bool,
            pad:bool,
            types:Tuple[str],
            precompute_mask_config:Dict[str, Any]=None,
            return_path:bool=False,
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
        self.precompute_mask_config = precompute_mask_config
        self.return_path = return_path

        os.makedirs(self.data_path, exist_ok=True)

        manifest_paths = []
        for type in types:
            LIBRISPEECH(root=self.data_path, url=type, download=True)

            tar_file_path = os.path.join(self.data_path, f"{type}.tar.gz")
            if os.path.exists(tar_file_path):
                os.remove(tar_file_path)

            manifest_path = os.path.join(self.data_path, 'LibriSpeech', type)

            if not os.path.exists(os.path.join(manifest_path, "{}.tsv".format('train'))):
                create_manifests(root=manifest_path, valid_percent=0, dest=manifest_path)
            
            manifest_paths.append(manifest_path)
        self.manifest_paths = manifest_paths

    def load(self):
        datasets = []
        for manifest_path in self.manifest_paths:
            manifest_path = os.path.join(manifest_path, "{}.tsv".format('train')) # TODO: change to self.split?

            compute_mask = self.precompute_mask_config is not None
            mask_args = {}
            if compute_mask:
                mask_args = self.precompute_mask_config    

            dataset = FileAudioDataset(
                manifest_path=manifest_path,
                sample_rate=self.sample_rate,
                max_sample_size=self.max_sample_size,
                min_sample_size=self.min_sample_size,
                pad=self.pad,
                normalize=self.normalize,
                compute_mask=compute_mask,
                **mask_args,
            )
            datasets.append(dataset)

        self.dataset = ConcatDataset(datasets)

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples):
        data_paths = [s['path'] for s in samples]
        collater_res = self.dataset.collater(samples)
        res = {
            'id': collater_res['id'],
            'audio': collater_res['net_input']['source'],
            'padding_mask': collater_res['net_input']['padding_mask'],
            'modes': self.modes
        }
        if 'precomputed_mask' in collater_res['net_input']:
            res['precomputed_mask'] = collater_res['net_input']['precomputed_mask'],
        if self.return_path:
            res['data_path'] = data_paths
        return res


class SpeechCommandsDataset(AudioDataset):
    def __init__(self, 
                 data_path:str,
                 split:str,
                 normalize:bool,
                 pad:bool,
                 ):
        super().__init__(data_path=data_path,
                         split=split,
                         sample_rate=16_000, 
                         max_sample_size=16_000,
                         normalize=normalize,
                         pad=pad,)
        # as described in the paper to the dataset, each sample is at a maximum of 1 second
        # long and is sampled at 16kHz (https://arxiv.org/pdf/1804.03209.pdf)
        # min_sample_size default is 0, so we take all samples
        # However, the parameters sample_rate, min_sample_size have to effect here anyway.
        # This is because even though we inherit from AudioDataset, we only do 
        # so we can easily reuse the collater method of AudioDataset.

        if self.split == "train":
            self.subset = "training"
        else:
            self.subset = "testing"

        SPEECHCOMMANDS(self.data_path, subset=self.subset, download=True)

        path_to_data = os.path.join(self.data_path, 'SpeechCommands', 'speech_commands_v0.02')
        # List all entries in the given path
        all_entries = os.listdir(path_to_data)
        # Filter out directories that do not start with '_'
        class_names = [entry for entry in all_entries if os.path.isdir(os.path.join(path_to_data, entry)) and not entry.startswith('_')]
        
        self.class_to_id = {class_name: i for i, class_name in enumerate(class_names)}

        if os.path.exists(f"{self.data_path}/speech_commands_v0.02.tar.gz"):
            os.remove(f"{self.data_path}/speech_commands_v0.02.tar.gz")

    def load(self):
        self.items = SPEECHCOMMANDS(self.data_path, subset=self.subset)

    def __getitem__(self, index):
        item = self.items[index]
        return {"audio": item[0][0], "target": item[2], "id": index}
    
    def collater(self, samples):
        input = super().collater(samples)
        input["target"] = torch.LongTensor([self.class_to_id[s["target"]] for s in samples])
        return input


class ImageNetDataset(ImageDataset):
    def __init__(
            self,
            data_path:str,
            split,
            color_jitter=None,
            aa="rand-m9-mstd0.5-inc1",
            reprob=0.25,
            remode="pixel",
            recount=1,
            precompute_mask_config=None,
    ):
        super().__init__(data_path=data_path, 
                         split=split,
                         color_jitter=color_jitter,
                         aa=aa,
                         reprob=reprob,
                         remode=remode,
                         recount=recount,
                         precompute_mask_config=precompute_mask_config)
        self.path_to_data = os.path.join(self.data_path, 'imagenet')
        if not os.path.exists(self.path_to_data):
            raise FileNotFoundError(f"Directory {self.path_to_data} does not exists, "
                                    "please create it and add the correponding files from HuggingFace: "
                                    f"https://huggingface.co/datasets/imagenet-1k")
        
        self.path_to_split = os.path.join(self.path_to_data, self.split)
        os.makedirs(self.path_to_split, exist_ok=True)

        self.classes = {synset: i for i, synset in enumerate(IMAGENET2012_CLASSES.keys())}

        if not os.path.exists(os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl')):
            self._make_imagenet_dataset_index()


    def load(self):
        items = []
        with open(os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl'), 'r', encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                items.append(data)
            self.log(f"Loaded {len(items)} {self.split} examples.")
        self.items = items

    def __getitem__(self, index):
        item = self.items[index]
        image = self._get_image(image_path=item['image_path'])
        data = {
            'image': image,
            'id': index,
            'target': item['target']
        }
        return data

    def __len__(self):
        return len(self.items)
    
    def _make_imagenet_dataset_index(self):
        items = []
        for file in os.listdir(self.path_to_split):
            if self.split != 'test':
                root, _ = os.path.splitext(file)
                _, synset_id = os.path.basename(root).rsplit("_", 1)
            else:
                synset_id = -1
            items.append({
                'image_path': os.path.join(self.path_to_split, file),
                'target': self.classes[synset_id],
            })

        write_data_into_jsonl(items, os.path.join(self.path_to_data, f'imagenet.{self.split}.jsonl'))

    
class CIFARDataset(BaseDataset):
    def __init__(self, 
                 data_path:str,
                 split:str,
                 type:str="cifar10",
                 ):
        super().__init__(data_path=data_path, 
                         split=split)
        self.type = type

        if self.type == "cifar10":
            CIFAR10(self.data_path, train=self.split == "train", download=True)
        elif self.type == "cifar100":
            CIFAR100(self.data_path, train=self.split == "train", download=True)
        else:
            raise ValueError(f'CIFARDataset: Unknown dataset type: {self.type}, available options: ["cifar10", "cifar100"].')
        
    @property
    def modes(self) -> List[Modality]:
        return [Modality.IMAGE]

    def load(self):
        # only used for evaluation -> no augmentations
        transform = get_transforms(no_transform=True, beit_transforms=False, transform_jitter=False)

        if self.type == "cifar10":
            self.items = CIFAR10(self.data_path, train=self.split == "train", transform=transform)
        else:
            self.items = CIFAR100(self.data_path, train=self.split == "train", transform=transform)

    def __getitem__(self, index):
        item = self.items[index]
        return {"image": item[0], "target": item[1]}


UNIMODAL_DATASET_REGISTRY = {
    "openwebtext": OpenWebTextDataset,
    "imdb": IMDBDataset,
    "librispeech": LibriSpeechDataset,
    "speechcommands": SpeechCommandsDataset,
    "imagenet": ImageNetDataset,
    "cifar10": partial(CIFARDataset, type='cifar10'),
    "cifar100": partial(CIFARDataset, type='cifar100'),
    "qqp": QQPDataset,
}
