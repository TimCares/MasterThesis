import os
import mmap
import numpy as np
import multiprocessing
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import Any
from .base_datasets import BaseDataset
from data2vec_fairseq.data.modality import Modality
import subprocess
import tempfile
try:
    import pyarrow.plasma as plasma
except ImportError:
    plasma = None

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
        data_path = os.path.join(data_path, name)
        super().__init__(data_path=data_path, split=split)
        os.makedirs(data_path, exist_ok=True)
        self.name = name
        self.text_file = text_file
        self.tokenizer_path = 'tokenizer'  # Directory to save the tokenizer
        tokenizer.save_pretrained(self.tokenizer_path)
        self.max_seq_length = block_size
        self.block_size = block_size - 2  # Subtract 2 for CLS and SEP tokens
        self.mask_prob = mask_prob
        self.token_file = os.path.join(self.data_path, f'mlm_{self.name}_{self.split}.bin')
        self.index_file = os.path.join(self.data_path, f'mlm_{self.name}_{self.split}.idx')
        self.index_entry_size = 16  # Each index entry has two int64 (offset and length)

        if not self.index_exists():
            self.preprocess()
        
        self.tokenizer = tokenizer

    def __getstate__(self):
        """Exclude non-picklable objects from the pickled state."""
        state = self.__dict__.copy()
        state.pop('fp', None)  # Remove the file pointer
        state.pop('mmap_file', None)  # Remove the mmap object
        return state

    def __setstate__(self, state):
        """Reinitialize the non-picklable objects after unpickling."""
        self.__dict__.update(state)
        # Reopen the file and recreate the mmap object
        self.fp = open(self.token_file, 'rb')
        self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

    def load(self):
        self.build_sequences()

    def preprocess(self):
        """Tokenize the text file and store tokenized data into a binary mmap file."""
        n_unk_tokens = 0
        n_total_tokens = 0
        # Calculate total lines for progress bar
        total_lines = sum(1 for _ in open(self.text_file, 'r', encoding='utf-8'))
        with open(self.text_file, 'r', encoding='utf-8') as f_in, \
            open(self.token_file, 'wb') as f_out, \
            open(self.index_file, 'wb') as f_idx:
            
            offset = 0
            batch_size = 10000

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
                        tokens = np.array(tokens, dtype=np.int32)
                        n_unk_tokens += (tokens == self.unk_token_id).sum()
                        n_total_tokens += length
                        tokens.tofile(f_out)
                        # Write offset and length as int64 to index file
                        f_idx.write(np.array([offset, length], dtype=np.int64).tobytes())
                        offset += length * 4  # Each int32 token is 4 bytes
                        pbar.update(1)
                    lines = []

            # Process remaining lines
            if lines:
                tokenized_lines = pool.map(tokenize_line, lines)
                for tokens in tokenized_lines:
                    length = len(tokens)
                    tokens = np.array(tokens, dtype=np.int32)
                    n_unk_tokens += (tokens == self.unk_token_id).sum()
                    n_total_tokens += length
                    tokens.tofile(f_out)
                    f_idx.write(np.array([offset, length], dtype=np.int64).tobytes())
                    offset += length * 4
                    pbar.update(1)

            pbar.close()
            pool.close()
            pool.join()

        self.log(f'Preprocessing complete. Processed {n_total_tokens} tokens, '
                 f'found {n_unk_tokens}({n_unk_tokens/n_total_tokens*100:.05f}%) unknown tokens.')

    def build_sequences(self):
        """Slice the tokenized data into blocks of a specific size."""
        # Memory-map the index file
        self.index_fp = open(self.index_file, 'rb')
        self.index_mmap = mmap.mmap(self.index_fp.fileno(), 0, access=mmap.ACCESS_READ)

        items = []
        current_offset = 0
        current_length = 0

        num_lines = self.get_num_lines()

        for idx in tqdm(range(num_lines), desc="Building sequences"):
            offset, length = self.get_index_entry(idx)
            if current_length == 0:
                current_offset = offset

            # Handle lines longer than block_size
            if length >= self.block_size:
                if current_length > 0:
                    items.append([current_offset, current_length]) # stop current chunk if it exists

                # Split the line into chunks of block_size
                num_splits = (length + self.block_size - 1) // self.block_size
                for i in range(num_splits):
                    split_offset = offset + i * self.block_size * 4  # 4 bytes per int32 token
                    split_length = min(self.block_size, length - i * self.block_size)
                    items.append([split_offset, split_length])
                
                current_length = 0
            else:
                if current_length + length <= self.block_size:
                    current_length += length
                else:
                    # Save the current chunk as a sequence
                    items.append([current_offset, current_length])
                    current_offset = offset
                    current_length = length

        # Add any remaining tokens as the last sequence
        if current_length > 0:
            items.append([current_offset, current_length])

        self._items = PlasmaArray(
            np.array(items, dtype=np.int64)
        )

        # Close the index mmap and file since it's no longer needed
        self.index_mmap.close()
        self.index_fp.close()
        del self.index_mmap
        del self.index_fp

    @property
    def items(self):
        return self._items.array

    def get_num_lines(self):
        """Calculate the number of lines in the index file."""
        index_file_size = os.path.getsize(self.index_file)
        num_lines = index_file_size // self.index_entry_size
        return num_lines

    def get_index_entry(self, idx):
        """Retrieve an index entry (offset and length) from the mmap index file."""
        start = idx * self.index_entry_size
        end = start + self.index_entry_size
        data = self.index_mmap[start:end]
        offset, length = np.frombuffer(data, dtype=np.int64)
        return offset, length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """Retrieve a sequence of tokens."""
        # Ensure that the mmap file is initialized
        if not hasattr(self, 'mmap_file'):
            self.fp = open(self.token_file, 'rb')
            self.mmap_file = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

        result_dict = dict()
        offset, length = self.items[idx]
        tokens = np.frombuffer(self.mmap_file[offset:offset + length * 4], dtype=np.int32).tolist()

        num_tokens = len(tokens)
        if num_tokens > self.block_size:
            tokens = tokens[:self.block_size]
            num_tokens = self.block_size

        if self.mask_prob > 0.0:
            text_tokens = self.tokenizer.convert_ids_to_tokens(tokens) # needed to detect subwords (prefix ##)
            mask_labels = self.whole_word_mask(text_tokens, mlm_probability=self.mask_prob)
            mask_labels = [0] + mask_labels + [0] * (self.max_seq_length - num_tokens - 1)
            result_dict["mask_labels"] = mask_labels

        tokens = [self.tokenizer.cls_token_id] + tokens + [self.tokenizer.sep_token_id] # self.block_size + 2 = self.max_seq_length
        num_tokens += 2 # Add CLS and SEP tokens
        padding_mask = [0] * num_tokens + [1] * (self.max_seq_length - num_tokens)
        language_tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_seq_length - num_tokens)

        result_dict["text"] = language_tokens
        result_dict["padding_mask"] = padding_mask
        result_dict["id"] = idx

        return result_dict
    
    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def get_index_files(self):
        return (self.token_file, self.index_file)

    def index_exists(self):
        for file in self.get_index_files():
            if not os.path.exists(file):
                self.log(f"File {file} not found. Index does not exist, creating...")
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


# copied from fairseq/data/plasma_utils.py:

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
class PlasmaArray:
    """
    Wrapper around numpy arrays that automatically moves the data to shared
    memory upon serialization. This is particularly helpful when passing numpy
    arrays through multiprocessing, so that data is not unnecessarily
    duplicated or pickled.
    """

    def __init__(self, array):
        super().__init__()
        self.array = array
        self.disable = array.nbytes < 134217728  # disable for arrays <128MB
        self.object_id = None
        self.path = None

        # variables with underscores shouldn't be pickled
        self._client = None
        self._server = None
        self._server_tmp = None
        self._plasma = None

    @property
    def plasma(self):
        if self._plasma is None and not self.disable:
            self._plasma = plasma
        return self._plasma

    def start_server(self):
        if self.plasma is None or self._server is not None:
            return
        assert self.object_id is None
        assert self.path is None
        self._server_tmp = tempfile.NamedTemporaryFile()
        self.path = self._server_tmp.name
        self._server = subprocess.Popen(
            ["plasma_store", "-m", str(int(1.05 * self.array.nbytes)), "-s", self.path]
        )

    @property
    def client(self):
        if self._client is None:
            assert self.path is not None
            self._client = self.plasma.connect(self.path, num_retries=200)
        return self._client

    def __getstate__(self):
        """Called on pickle load"""
        if self.plasma is None:
            return self.__dict__
        if self.object_id is None:
            self.start_server()
            self.object_id = self.client.put(self.array)
        state = self.__dict__.copy()
        del state["array"]
        state["_client"] = None
        state["_server"] = None
        state["_server_tmp"] = None
        state["_plasma"] = None
        return state

    def __setstate__(self, state):
        """Called on pickle save"""
        self.__dict__.update(state)
        if self.plasma is None:
            return
        self.array = self.client.get(self.object_id)

    def __del__(self):
        if self._server is not None:
            self._server.kill()
            self._server = None
            self._server_tmp.close()
            self._server_tmp = None
