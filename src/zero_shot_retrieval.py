# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from rich.progress import track
from typing import *
import torch
from flava.data.transforms import (
    default_image_pretraining_transforms,
    default_text_transform,
)
from torch import nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def compute_recall(similarity_scores: torch.Tensor, k: int = 5):
    dataset_size = similarity_scores.size(0)
    targets = torch.arange(dataset_size).view(dataset_size, -1)
    _, topk_idx = torch.topk(similarity_scores, k)
    recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall


def transform(image, target):
    _, image_transform = default_image_pretraining_transforms()
    transformed_image = image_transform(image)
    # Take the first caption for now
    transformed_text = default_text_transform()(target[0])
    return transformed_image, transformed_text


# def collator(batch):
#     texts = []
#     images = torch.stack([x[0]["image"] for x in batch], dim=0)
#     texts = torch.cat([torch.LongTensor(x[1]["input_ids"]) for x in batch], dim=0)
#     return images, texts


@torch.no_grad()
def zero_shot_retrieval(model, dataloader, device, name, modalities):
    mode_a_embeds = []
    mode_b_embeds = []

    for _, batch in track(enumerate(dataloader), description=f"Encoding {name}..."):
        a, b = batch
        _, a_emb = model.encode_image(a.to(device), projection=True)
        _, b_emb = model.encode_text(b.to(device), projection=True)
        mode_a_embeds.append(a_emb.detach().cpu())
        mode_b_embeds.append(b_emb.detach().cpu())

    mode_a_embeds = torch.cat(mode_a_embeds, 0)
    mode_b_embeds = torch.cat(mode_b_embeds, 0)

    mode_a_embeds = nn.functional.normalize(mode_a_embeds, dim=-1)
    mode_b_embeds = nn.functional.normalize(mode_b_embeds, dim=-1)

    similarity_scores = mode_a_embeds @ mode_b_embeds.t()
    similarity_scores_t = similarity_scores.t()

    a_to_b_r1 = compute_recall(similarity_scores, k=1)
    a_to_b_r5 = compute_recall(similarity_scores, k=5)
    b_to_a_r1 = compute_recall(similarity_scores_t, k=1)
    b_to_a_r5 = compute_recall(similarity_scores_t, k=5)

    logger.info(f"{name}: {modalities[0]}_to_{modalities[1]}_recall@1 {a_to_b_r1}")
    logger.info(f"{name}: {modalities[0]}_to_{modalities[1]}_recall@5 {a_to_b_r5}")
    logger.info(f"{name}: {modalities[1]}_to_{modalities[0]}_recall@1 {b_to_a_r1}")
    logger.info(f"{name}: {modalities[1]}_to_{modalities[0]}_recall@5 {b_to_a_r5}")


def make_zero_shot_retrieval(model:Callable, dataloaders:Dict[str, torch.utils.data.DataLoader], modalities:List[Tuple[str, str]]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    # names = ["coco", "flickr"]
    # dataloaders = [dataloader, dataloader]
    # modalities = [("image", 'text'), ("image", 'text')]
    for name, dataloader, modalities in zip(dataloaders.keys(), dataloaders.values(), modalities):
        logger.info(f"Zero-shot retrieval on: {name}")
        zero_shot_retrieval(model, dataloader, device, name, modalities)
