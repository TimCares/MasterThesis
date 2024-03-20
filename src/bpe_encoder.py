#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.data.encoders.gpt2_bpe import get_encoder

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
