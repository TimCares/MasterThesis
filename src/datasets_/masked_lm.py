import os
import mmap
import numpy as np
import pickle
import multiprocessing
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any
from .base_datasets import BaseDataset
from data2vec_fairseq.data.modality import Modality

def init_worker(tokenizer_path):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def tokenize_line(line):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line.strip()))


class MaskedLMDataset(BaseDataset):
    def __init__(
        self,
        name: str,
        data_path: str,
        split: str,
        text_file: os.PathLike,
        tokenizer: Any,
        block_size: int=512,
        mask_prob: float=0.0,
    ):
        super().__init__(data_path=data_path, split=split)
        self.name = name
        self.text_file = text_file
        self.tokenizer_path = 'tokenizer'  # Directory to save the tokenizer
        tokenizer.save_pretrained(self.tokenizer_path)
        self.block_size = block_size
        self.mask_prob = mask_prob
        self.token_file = os.path.join(self.data_path, f'mlm_{self.name}_{self.split}.bin')
        self.index_file = os.path.join(self.data_path, f'mlm_{self.name}_{self.split}.idx')

        if not self.index_exists():
            self.preprocess()

        # Load the index
        with open(self.index_file, 'rb') as f_idx:
            self.index = pickle.load(f_idx)

        # Open the mmap file
        self.fp = open(self.token_file, 'rb')
        self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

        # Build sequences
        self.build_sequences()

        self.tokenizer = tokenizer

    def preprocess(self):
        """Tokenize the text file and store tokenized data into a binary mmap file."""
        # Calculate total lines for progress bar
        total_lines = sum(1 for _ in open(self.text_file, 'r', encoding='utf-8'))
        with open(self.text_file, 'r', encoding='utf-8') as f_in, open(self.token_file, 'wb') as f_out:
            index = []
            offset = 0
            batch_size = 10000  # Adjust as needed

            # Initialize multiprocessing pool
            pool = multiprocessing.Pool(initializer=init_worker, initargs=(self.tokenizer_path,))
            pbar = tqdm(total=total_lines, desc="Tokenizing")

            lines = []
            for line in f_in:
                lines.append(line)
                if len(lines) >= batch_size:
                    # Tokenize the batch in parallel
                    tokenized_lines = pool.map(tokenize_line, lines)
                    # Write tokenized data to file and update index
                    for tokens in tokenized_lines:
                        length = len(tokens)
                        np.array(tokens, dtype=np.int32).tofile(f_out)
                        index.append((offset, length))
                        offset += length * 4  # Each int32 token is 4 bytes
                        pbar.update(1)
                    lines = []

            # Process remaining lines
            if lines:
                tokenized_lines = pool.map(tokenize_line, lines)
                for tokens in tokenized_lines:
                    length = len(tokens)
                    np.array(tokens, dtype=np.int32).tofile(f_out)
                    index.append((offset, length))
                    offset += length * 4
                    pbar.update(1)

            pbar.close()
            pool.close()
            pool.join()

            # Save the index to a file
            with open(self.index_file, 'wb') as f_idx:
                pickle.dump(index, f_idx)

    def build_sequences(self):
        """Slice the tokenized data into blocks of a specific size."""
        self.examples = []
        current_chunk = []
        current_length = 0

        for idx in tqdm(range(len(self.index)), desc="Building sequences"):
            offset, length = self.index[idx]
            if current_length + length <= self.block_size:
                current_chunk.append((offset, length))
                current_length += length
            else:
                # Save the current chunk as a sequence
                self.examples.append(current_chunk)
                current_chunk = [(offset, length)]
                current_length = length

        # Add any remaining tokens as the last sequence
        if current_chunk:
            self.examples.append(current_chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Retrieve a sequence of tokens."""
        data = dict()
        chunks = self.examples[idx]
        tokens = []
        for offset, length in chunks:
            data = self.mmap_file[offset:offset + length * 4]
            tokens.extend(np.frombuffer(data, dtype=np.int32))
        tokens = torch.tensor(tokens, dtype=torch.long)

        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
            tokens[-1] = self.tokenizer.sep_token_id # Replace last token with [SEP]

        data = {
            "text": tokens,
            "padding_mask": torch.zeros_like(tokens),
        }
        if self.mask_prob > 0.0:
            text_tokens = self.tokenizer.convert_ids_to_tokens(tokens) # needed to detect subwords (prefix ##)
            mask_labels = self.whole_word_mask(text_tokens, mlm_probability=self.mask_prob)
            mask_labels = [0] + mask_labels + [0] * (self.block_size - len(mask_labels) - 1)
            data["mask_labels"] = mask_labels

        return tokens
    
    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def get_index_files(self):
        return (self.token_file, self.index_file)

    def index_exists(self):
        for file in self.get_index_files():
            if not os.path.exists(file):
                return False
        self.log(f"Data already exists under: {self.data_path}")
        return True
    
    def collater(self, samples):
        batch_tensors = super().collater(samples)
        if self.mask_prob > 0.0:
            batch_tensors['targets'] = batch_tensors['text'].clone()
            batch_tensors['targets'][~batch_tensors['mask_labels'].bool()] = -100
            batch_tensors['text'][batch_tensors['mask_labels'].bool()] = self.tokenizer.mask_token_id
        return batch_tensors