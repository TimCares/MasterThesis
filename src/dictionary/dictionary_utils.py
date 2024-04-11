# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import re

def post_process(sentence: str, symbol: str):
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re

        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(" +", " ", sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence


def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == "xla":
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()