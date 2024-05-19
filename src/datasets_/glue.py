import torchtext
from .base_datasets import BaseDataset
import os
from fairseq.data import Dictionary
from bpe_encoder import get_bpe_encoder, BPEEncoder
from utils import pad_text_sequence
from .data_utils import write_data_into_jsonl
import json
from data2vec_fairseq.data.modality import Modality
from typing import List, Dict, Any
import shutil
from torchvision.datasets.utils import download_url
import pandas as pd

class GLUE(BaseDataset):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int=512,):
        super().__init__(data_path=data_path,
                         split=split)
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = os.path.join(self.data_path, self._dataset_name + '_glue')
        self.out_jsonl_path = os.path.join(self.path_to_data, f'{self.split}.jsonl')

        if os.path.exists(self.out_jsonl_path):
            self.log(f"Found {self.out_jsonl_path}. Skip preprocessing.")
            return

        dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        bpe_encoder = get_bpe_encoder(self.data_path)

        self.bos_token_id = dictionary.bos()
        self.eos_token_id = dictionary.eos()
        self.pad_token_id = dictionary.pad()
                
        os.makedirs(self.path_to_data, exist_ok=True)

        self.num_tokens_upper_bound = self._get_max_length(iter(self._dataset(split=self.split)))
        if self.num_tokens_upper_bound > self.num_max_bpe_tokens:
            self.log(f"Upper bound length: {self.num_tokens_upper_bound} is greater than num_max_bpe_tokens: {self.num_max_bpe_tokens}.")
            self.num_tokens_upper_bound = self.num_max_bpe_tokens

        items = self._make_index(bpe_encoder)
            
        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(f'{self.path_to_data}/datasets')
            
    @property
    def modes(self) -> List[Modality]:
        return [Modality.TEXT]
    
    def _make_index(self, bpe_encoder: BPEEncoder) -> List[Dict[str, Any]]:
        raise NotImplementedError()
    
    def _dataset(self, split):
        raise NotImplementedError()
    
    @property
    def _dataset_name(self):
        raise NotImplementedError()
                
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
    
    def _get_max_length(self, data_loader) -> int:
        raise NotImplementedError()
    

class CoLA(GLUE):
    def _dataset(self, split):
        return torchtext.datasets.CoLA(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'cola'
    
    def _get_max_length(self, data_loader) -> int:
        return max([len(d[2]) for d in data_loader])
    
    def _make_index(self, bpe_encoder: BPEEncoder) -> List[Dict[str, Any]]:
        items = []
        for _, target, text in iter(self._dataset(split=self.split)):
            tokens = bpe_encoder.encode(text)
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                              pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)
            items.append({'text': language_tokens,
                          'padding_mask': padding_mask,
                          'target': target})
            
        return items
    

class SST(GLUE):
    def _dataset(self, split):
        return torchtext.datasets.SST2(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'sst'
    
    def _get_max_length(self, data_loader) -> int:
        return self.num_max_bpe_tokens
    
    def _make_index(self, bpe_encoder: BPEEncoder) -> List[Dict[str, Any]]:
        items = []
        for text, target in iter(self._dataset(split=self.split)):
            tokens = bpe_encoder.encode(text)
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                                pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)
            items.append({'text': language_tokens,
                          'padding_mask': padding_mask,
                          'target': target})
            
        return items
    

class QNLI(GLUE):
    def _dataset(self, split):
        return torchtext.datasets.QNLI(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'qnli'
    
    def _get_max_length(self, data_loader) -> int:
        return self.num_max_bpe_tokens
    
    def _make_index(self, bpe_encoder: BPEEncoder) -> List[Dict[str, Any]]:
        items = []
        n_trunc = 0
        for target, question, text in iter(self._dataset(split=self.split)):
            question_tokens = bpe_encoder.encode(question)
            text_tokens = bpe_encoder.encode(text)

            max_len_text = self.num_max_bpe_tokens - len(question_tokens) - 2 # -2 for bos token and separator (eos)
            if len(text_tokens) > max_len_text:
                n_trunc += 1
                text_tokens = text_tokens[:max_len_text]

            tokens = text_tokens + [self.eos_token_id] + question_tokens
            assert len(tokens) <= self.num_max_bpe_tokens
            
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                              pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)
            assert self.eos_token_id in language_tokens # both text and question should still be there after potential truncation

            items.append({'text': language_tokens,
                          'padding_mask': padding_mask,
                          'target': target})
        
        self.log(f"Truncated {n_trunc} examples.")
            
        return items
    

class RTE(GLUE):
    def _dataset(self, split):
        return torchtext.datasets.RTE(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'rte'
    
    def _get_max_length(self, data_loader) -> int:
        return max([len(d[1])+len(d[2]) for d in data_loader])
    
    def _make_index(self, bpe_encoder: BPEEncoder) -> List[Dict[str, Any]]:
        items = []
        n_trunc = 0
        for target, text1, text2 in iter(self._dataset(split=self.split)):
            text1_tokens = bpe_encoder.encode(text1)
            text2_tokens = bpe_encoder.encode(text2)

            max_len_text = self.num_max_bpe_tokens - len(text2_tokens) - 2 # -2 for bos token and separator (eos)
            if len(text1_tokens) > max_len_text:
                n_trunc += 1
                text1_tokens = text1_tokens[:max_len_text]

            tokens = text1_tokens + [self.eos_token_id] + text2_tokens
            assert len(tokens) <= self.num_max_bpe_tokens
            
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                              pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)
            assert self.eos_token_id in language_tokens # both text and question should still be there after potential truncation

            items.append({'text': language_tokens,
                          'padding_mask': padding_mask,
                          'target': target})
        
        self.log(f"Truncated {n_trunc} examples.")
            
        return items
    

class MRPC(RTE):
    def _dataset(self, split):
        return torchtext.datasets.MRPC(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'mrpc'

#https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip
class QQP(GLUE):
    def __init__(
            self,
            data_path:str,
            split:str,
            num_max_bpe_tokens:int=512,):
        self.data_path = data_path
        self.split = split
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.path_to_data = os.path.join(self.data_path, self._dataset_name)
        self.out_jsonl_path = os.path.join(self.path_to_data, f'{self.split}.jsonl')

        if os.path.exists(self.out_jsonl_path):
            self.log(f"Found {self.out_jsonl_path}. Skip preprocessing.")
            return

        dictionary = Dictionary.load(os.path.join(self.data_path, "dict.txt"))
        bpe_encoder = get_bpe_encoder(self.data_path)

        self.bos_token_id = dictionary.bos()
        self.eos_token_id = dictionary.eos()
        self.pad_token_id = dictionary.pad()
                
        os.makedirs(self.path_to_data, exist_ok=True)

        URL='https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip'
        download_url(url=URL, root=self.path_to_data)
        filepath = os.path.join(self.path_to_data, os.path.basename(URL))
        os.system(f"unzip {filepath} -d {self.path_to_data}")
        os.remove(filepath)

        items = self._make_index(bpe_encoder)
            
        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(os.path.join(self.path_to_data, 'QQP'))
    
    @property
    def _dataset_name(self):
        return 'qqp'
    
    def _make_index(self, bpe_encoder: BPEEncoder) -> List[Dict[str, Any]]:
        path = os.path.join(self.path_to_data, 'QQP', f'{self.split}.tsv')
        df = pd.read_csv(path, delimiter='\t')[['question1', 'question2', 'is_duplicate']]

        items = []
        n_trunc = 0
        for _, example in df.iterrows():
            question1_tokens = bpe_encoder.encode(example['question1'])
            question2_tokens = bpe_encoder.encode(example['question2'])

            max_len_text = self.num_max_bpe_tokens - len(question2_tokens) - 2 # -2 for bos token and separator (eos)
            if len(question1_tokens) > max_len_text:
                n_trunc += 1
                question1_tokens = question1_tokens[:max_len_text]

            tokens = question1_tokens + [self.eos_token_id] + question2_tokens
            assert len(tokens) <= self.num_max_bpe_tokens
            
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_max_bpe_tokens,
                                                              pad_idx=self.pad_token_id, bos_idx=self.bos_token_id)
            assert self.eos_token_id in language_tokens # both text and question should still be there after potential truncation

            items.append({'text': language_tokens,
                          'padding_mask': padding_mask,
                          'target': example['is_duplicate']})
            
        self.log(f"Truncated {n_trunc} examples.")
        return items
    

class STSB(RTE):
    def _dataset(self, split):
        return torchtext.datasets.STSB(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'stsb'
    

class MNLI(RTE):
    def _dataset(self, split):
        return torchtext.datasets.MNLI(root=self.path_to_data, split=split)
    
    def _get_max_length(self, data_loader) -> int:
        return self.num_max_bpe_tokens # dataset is very large, looping through it is too slow
    
    @property
    def _dataset_name(self):
        return 'mnli'
    

GLUE_DATASET_REGISTRY = {
    'cola_glue': CoLA,
    'sst_glue': SST,
    'qnli_glue': QNLI,
    'rte_glue': RTE,
    'mrpc_glue': MRPC,
    'qqp_glue': QQP,
    'stsb_glue': STSB,
    'mnli_glue': MNLI,
}
