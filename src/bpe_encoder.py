#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys
from collections import Counter
from multiprocessing import Pool
import torch
import os
from datasets import download_url

from fairseq.data.encoders.gpt2_bpe_utils import get_encoder

def load_tokenizer_data(store_at:str="../data"):
    for filename in ["dict.txt", "encoder.json", "vocab.bpe"]:
        if not os.path.exists(os.path.join(store_at, filename)):
            url = f"https://dl.fbaipublicfiles.com/fairseq/data2vec2/{filename}"
            download_url(url=url, store_at=store_at)

def get_bpe_encoder(data_path):
    load_tokenizer_data(data_path)
    encoder_json_path = os.path.join(data_path, "encoder.json")
    vocab_bpe_path = os.path.join(data_path, "vocab.bpe")
    return BPEEncoder(encoder_json_path, vocab_bpe_path)

class BPEEncoder(object):
    def __init__(self, encoder_json_path, vocab_bpe_path):
        self.bpe = get_encoder(encoder_json_path, vocab_bpe_path)

    def encode(self, line):
        line = line.strip()
        return self.bpe.encode(line)

    def decode(self, tokens):
        return self.bpe.decode(tokens)

    def encode_lines(self, lines, tokens_per_sample=None, to_tensor=True):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            tokens = self.encode(line)
            if tokens_per_sample is not None and len(tokens) > tokens_per_sample:
                tokens = tokens[:tokens_per_sample]
            if to_tensor:
                tokens = torch.tensor(tokens)
            enc_lines.append(tokens)
        return enc_lines

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return dec_lines


class MultiprocessingEncoder(object):
    def __init__(self, encoder_path, vocab_path, keep_empty):
        self.encoder_path = encoder_path
        self.vocab_path = vocab_path
        self.keep_empty = keep_empty

    def initializer(self):
        global bpe
        bpe = get_encoder(self.encoder_path, self.vocab_path)

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0 and not self.keep_empty:
                return ["EMPTY", None]
            tokens = self.encode(line)
            enc_lines.append(" ".join(tokens))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-json",
        help="path to encoder.json",
    )
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help="path to vocab.bpe",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["-"],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=["-"],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    args = parser.parse_args()

    assert len(args.inputs) == len(
        args.outputs
    ), "number of input and output paths should match"
    encode(args.encoder_json, args.vocab_bpe, args.inputs, args.outputs, args.keep_empty)

def encode(encoder_path, vocab_path, inputs_arg, outputs_arg, keep_empty):
    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-"
            else sys.stdin
            for input in inputs_arg
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-"
            else sys.stdout
            for output in outputs_arg
        ]

        encoder = MultiprocessingEncoder(encoder_path, vocab_path, keep_empty)
        pool = Pool(initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

if __name__ == "__main__":
    main()
