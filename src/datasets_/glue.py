import torchtext
from .base_datasets import BaseDataset
from typing import Tuple
import os
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
                
        os.makedirs(self.path_to_data, exist_ok=True)

        self.num_tokens_upper_bound = self._get_max_length(iter(self._dataset(split=self.split)))
        if self.num_tokens_upper_bound > self.num_max_bpe_tokens:
            self.log(f"Upper bound length: {self.num_tokens_upper_bound} is greater than num_max_bpe_tokens: {self.num_max_bpe_tokens}.")
            self.num_tokens_upper_bound = self.num_max_bpe_tokens

        items = self._make_index()
            
        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(f'{self.path_to_data}/datasets')

    def prepare_sentence_pair(self, sentence1:str, sentence2:str) -> Tuple[List[int], List[int], List[int], int]:
        tokens1 = self.tokenize_text(sentence1)
        tokens2 = self.tokenize_text(sentence2)
        
        trunc = 0
        skip = 0
        max_len_text = self.num_max_bpe_tokens - len(tokens1) - 3 # -3 for bos token and 2x separator (eos)
        if len(text_tokens) > max_len_text:
            trunc = 1
            text_tokens = text_tokens[:max_len_text]

        tokens = text_tokens + [self.sep_token_id] + tokens2
        assert len(tokens) <= self.num_max_bpe_tokens
        
        language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                          pad_idx=self.pad_token_id, bos_idx=self.cls_token_id,
                                                          eos_idx=self.sep_token_id)
        if self.sep_token_id not in language_tokens:# both text and question should still be there after potential truncation
            skip = 1
        return language_tokens, padding_mask, trunc, skip
            
    @property
    def modality(self) -> Modality:
        return Modality.TEXT
    
    def _make_index(self) -> List[Dict[str, Any]]:
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
    
    def _make_index(self) -> List[Dict[str, Any]]:
        items = []
        for _, target, text in iter(self._dataset(split=self.split)):
            tokens = self.tokenize_text(text)
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                              pad_idx=self.pad_token_id, bos_idx=self.cls_token_id,
                                                              eos_idx=self.sep_token_id)
            items.append({'x': language_tokens,
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
    
    def _make_index(self) -> List[Dict[str, Any]]:
        items = []
        for text, target in iter(self._dataset(split=self.split)):
            tokens = self.tokenize_text(text)
            language_tokens, padding_mask = pad_text_sequence(tokens=tokens, num_max_bpe_tokens=self.num_tokens_upper_bound,
                                                              pad_idx=self.pad_token_id, bos_idx=self.cls_token_id,
                                                              eos_idx=self.sep_token_id)
            items.append({'x': language_tokens,
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
    
    def _make_index(self) -> List[Dict[str, Any]]:
        items = []
        n_trunc = 0
        n_skip = 0
        for target, text1, text2 in iter(self._dataset(split=self.split)):
            language_tokens, padding_mask, trunc, skip = self.prepare_sentence_pair(sentence1=text1, sentence2=text2)
            if skip:
                n_skip += 1
                continue
            if trunc:
                n_trunc += 1

            items.append({'x': language_tokens,
                          'padding_mask': padding_mask,
                          'target': target})
        
        self.log(f"Truncated {n_trunc} examples.")
        self.log(f"Skipped {n_skip} examples.")
            
        return items
    

class RTE(QNLI):
    def _dataset(self, split):
        return torchtext.datasets.RTE(root=self.path_to_data, split=split)
    
    @property
    def _dataset_name(self):
        return 'rte'
    
    def _get_max_length(self, data_loader) -> int:
        return max([len(d[1])+len(d[2]) for d in data_loader])

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
                
        os.makedirs(self.path_to_data, exist_ok=True)

        URL='https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip'
        download_url(url=URL, root=self.path_to_data)
        filepath = os.path.join(self.path_to_data, os.path.basename(URL))
        os.system(f"unzip {filepath} -d {self.path_to_data}")
        os.remove(filepath)

        items = self._make_index()
            
        write_data_into_jsonl(items, self.out_jsonl_path)
        shutil.rmtree(os.path.join(self.path_to_data, 'QQP'))
    
    @property
    def _dataset_name(self):
        return 'qqp'
    
    def _make_index(self) -> List[Dict[str, Any]]:
        path = os.path.join(self.path_to_data, 'QQP', f'{self.split}.tsv')
        df = pd.read_csv(path, delimiter='\t')[['question1', 'question2', 'is_duplicate']]

        items = []
        n_trunc = 0
        n_skip = 0
        for _, example in df.iterrows():
            question1_tokens = self.tokenize_text(example['question1'])
            question2_tokens = self.tokenize_text(example['question2'])
            language_tokens, padding_mask, trunc, skip = self.prepare_sentence_pair(sentence1=question1_tokens, sentence2=question2_tokens)
            if skip:
                n_skip += 1
                continue
            if trunc:
                n_trunc += 1

            items.append({'x': language_tokens,
                          'padding_mask': padding_mask,
                          'target': example['is_duplicate']})
        
        self.log(f"Truncated {n_trunc} examples.")
        self.log(f"Skipped {n_skip} examples.")
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
