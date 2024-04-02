import os
import logging
import torch
import json
import re
from typing import *
import subprocess
import shutil
import glob
from .data_utils import get_transforms, download_and_unzip, write_data_into_jsonl, get_bpe_encoder
from .base_datasets import BaseDataset
from utils.wav2vec_manifest import create_manifests
from bpe_encoder import encode

from fairseq.data import Dictionary
from fairseq.data.audio.raw_audio_dataset import FileAudioDataset
from fairseq.data.text_compressor import TextCompressionLevel

from torchaudio.datasets import LIBRISPEECH, SPEECHCOMMANDS
import torchtext
from torchvision.datasets import CIFAR10, CIFAR100

from .base_datasets import AudioDataset, ImageDataset, MaeImageDataset, NLPDataset

logger = logging.getLogger(__name__)

class EnWik9Dataset(NLPDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int,
            sample_break_mode:str='none',):
        super().__init__(data_path, split, num_max_bpe_tokens, sample_break_mode)

        dataset_path = os.path.join(self.nlp_dir_path, 'enwik9')
        os.makedirs(dataset_path, exist_ok=True)
        download_and_unzip(urls=["http://mattmahoney.net/dc/enwik9.zip"], store_at='.')
        os.system(f"perl ../setup/clean_enwik9.pl enwik9 > enwik9.txt")
        os.remove("enwik9")
        encode(f'{self.data_path}/encoder.json', f'{self.data_path}/vocab.bpe', ['enwik9.txt'], ['enwik9.bpe'], keep_empty=True)
        os.remove("enwik9.txt")
        process = ['fairseq-preprocess', '--only-source', '--srcdict', f'{self.data_path}/dict.txt',
                    '--trainpref', 'enwik9.bpe', '--destdir', f'{dataset_path}', '--workers', f'{os.cpu_count()}']
        subprocess.run(process)
        os.remove("enwik9.bpe")

class OpenWebTextDataset(NLPDataset):
    def __init__(self,
                 data_path: str,
                 split: str,
                 num_max_bpe_tokens: int,
                 sample_break_mode: str = 'none'):
        super().__init__(data_path, split, num_max_bpe_tokens, sample_break_mode)
        dataset_path = os.path.join(self.nlp_dir_path, 'openwebtext')
        pattern = os.path.join(self.nlp_dir_path, '*.tar')
        files = glob.glob(pattern)
        self.log(f"Found {len(files)} tar files, inflating...")
        for file in files:
            os.system(f"tar -xf {file} -C {self.nlp_dir_path}")
            os.remove(file)
        pattern = os.path.join(dataset_path, '*.xz')
        files = glob.glob(pattern)
        for file in files:
            os.system(f"unxz {file}")
        self.log("Inflated all tar files.")

        pattern = rb'\x00+'

        self.log("Cleaning...")
        files = os.listdir(dataset_path)
        for file in files:
            with open(os.path.join(dataset_path, file), 'rb') as f:
                first_line = f.readline()
                rest_of_file = f.read()
            
            matches = list(re.finditer(pattern, first_line))
            if matches:
                first_line = first_line[matches[-1].end():]

            with open(os.path.join(dataset_path, file), 'wb') as f:
                f.write(first_line)
                f.write(rest_of_file)

            with open(os.path.join(dataset_path, file), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            with open(os.path.join(dataset_path, file), 'w', encoding='utf-8') as f:
                for line in lines:
                    stripped_line = line.strip()
                    if stripped_line != '' and stripped_line != '---':
                        f.write(line)

        self.log("Joining...")
        with open(os.path.join(dataset_path, 'openwebtext.txt'), 'w') as f:
            for file in files:
                path_to_file = os.path.join(dataset_path, file)
                with open(path_to_file, 'r') as f2:
                    f.write(f2.read())
                os.remove(path_to_file)

        self.log("Encoding...")
        in_file = os.path.join(dataset_path, 'openwebtext.txt')
        out_file = os.path.join(dataset_path, 'openwebtext.bpe')
        encode(f'{self.data_path}/encoder.json', f'{self.data_path}/vocab.bpe', [in_file], [out_file], keep_empty=True)
        os.remove(in_file)
        process = ['fairseq-preprocess', '--only-source', '--srcdict', f'{self.data_path}/dict.txt',
                    '--trainpref', out_file, '--destdir', f'{dataset_path}', '--workers', f'{os.cpu_count()}']
        subprocess.run(process)
        os.remove(out_file)


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

        dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        bpe_encoder = get_bpe_encoder(self.data_path)

        bos_token_id = dictionary.bos()
        eos_token_id = dictionary.eos()
        pad_token_id = dictionary.pad()
                
        os.makedirs(self.path_to_data, exist_ok=True)
        
        if os.path.exists(self.out_jsonl_path):
            self.log(f'Data already exists. Skip creating it.')
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

        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(f'{self.path_to_data}/datasets')
                
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
            type:str,
            **precompute_mask_config,
            ):
        super().__init__(data_path, split, sample_rate, max_sample_size, min_sample_size, 
                         True, False,
                         **precompute_mask_config)
        self.precompute_mask_config = precompute_mask_config

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
            pad=self.pad,
            normalize=self.normalize,
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


class SpeechCommandsDataset(AudioDataset):
    def __init__(self, 
                 data_path:str,
                 split:str,
                 min_sample_size:int,
                 normalize:bool,
                 pad:bool,
                 **precompute_mask_config,
                 ):
        super().__init__(data_path,
                         split,
                         16_000, 
                         16_000,
                         min_sample_size, normalize, pad, **precompute_mask_config)
        # as described in the paper to the dataset, each sample is at a maximum of 1 second
        # long and is sampled at 16kHz (https://arxiv.org/pdf/1804.03209.pdf)

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
        input = super().collater(samples)
        input["label"] = torch.LongTensor([self.class_to_id[s["label"]] for s in samples])
        return input


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
        # only used for evaluation -> no augmentations
        transform = get_transforms(no_transform=True, beit_transforms=False, transform_jitter=False)

        if self.type == "cifar10":
            self.items = CIFAR10(self.data_path, train=self.split == "train", transform=transform)
        else:
            self.items = CIFAR100(self.data_path, train=self.split == "train", transform=transform)

    def __getitem__(self, index):
        item = self.items[index]
        return {"image": item[0], "target": item[1]}
