import torch
import einops
import opt_einsum
from typing import Dict, Tuple
from rich.progress import track
import logging

logger = logging.getLogger(__name__)

# code for cmli logits heavily borrowed from x-clip -> https://github.com/lucidrains/x-clip
# (https://github.com/lucidrains/x-clip/blob/main/x_clip/x_clip.py)

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(mask.bool(), 0.)
    numer = t.sum(dim = dim)
    denom = (~mask.bool()).sum(dim = dim).clamp(min = eps)
    return numer / denom

def chunked_matmul(
    text_features: torch.Tensor,
    image_features: torch.Tensor,
    text_chunk_size: int,
    image_chunk_size: int,
    display_progress: bool = False,
) -> torch.Tensor:
    if text_chunk_size == -1:
        text_chunk_size = text_features.shape[0]
    if image_chunk_size == -1:
        image_chunk_size = image_features.shape[0]
    
    assert text_features.shape[0] % text_chunk_size == 0
    assert image_features.shape[0] % image_chunk_size == 0

    result = []
    text_iterator = range(0, text_features.shape[0], text_chunk_size)
    if display_progress:
        text_iterator = track(text_iterator, description="Processing matmul chunks")
    for text_chunk_idx in text_iterator:
        text_chunk = text_features[text_chunk_idx:text_chunk_idx+text_chunk_size]

        row_result = []
        for image_chunk_idx in range(0, image_features.shape[0], image_chunk_size):
            image_chunk = image_features[image_chunk_idx:image_chunk_idx+image_chunk_size]

            chunk_result = opt_einsum.contract('x t d, y i d -> x y t i', text_chunk, image_chunk)
            row_result.append(chunk_result)
        
        result.append(torch.cat(row_result, dim=1))
    
    return torch.cat(result, dim=0)

def reduce_token_similarity(
        sim:torch.Tensor,
        padding_mask:torch.Tensor,
) -> Dict[str, torch.Tensor]:
    
    text_to_image = einops.reduce(sim, '... t i -> ... t', 'max')
    text_to_image_mask = einops.rearrange(padding_mask, 'b t -> b 1 t')
    text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

    image_to_text_mask = einops.rearrange(padding_mask, 'b t -> b 1 t 1')
    masked_sim = sim.masked_fill(image_to_text_mask.bool(), max_neg_value(sim.dtype))
    image_to_text = einops.reduce(einops.reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

    return {"i2t": image_to_text, "t2i": text_to_image}

def to_half(text_features:torch.Tensor, image_features:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    text_features = text_features.half()
    image_features = image_features.half()
    return text_features, image_features

def remove_cls(
    text_features:torch.Tensor, 
    image_features:torch.Tensor, 
    padding_mask:torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    text_features = text_features[:, 1:]
    image_features = image_features[:, 1:]
    padding_mask = padding_mask[:, 1:]
    return text_features, image_features, padding_mask

def infer_cmli_logits(
    text_features:torch.Tensor,
    image_features:torch.Tensor,
    padding_mask:torch.Tensor,
    logit_scale:float|torch.Tensor=1.0,
) -> Dict[str, torch.Tensor]:
    
    text_features, image_features = to_half(text_features, image_features)
    text_features, image_features, padding_mask = remove_cls(text_features, image_features, padding_mask)
    
    sim = logit_scale * opt_einsum.contract('x t d, y i d -> x y t i', text_features, image_features)

    return reduce_token_similarity(sim, padding_mask)

def infer_chunked_cmli_logits(
    text_features:torch.Tensor,
    image_features:torch.Tensor,
    padding_mask:torch.Tensor,
    text_chunk_size:int,
    image_chunk_size:int,
    display_progress:bool=False,
    logit_scale:float|torch.Tensor=1.0,
) -> Dict[str, torch.Tensor]:
    
    text_features, image_features = to_half(text_features, image_features)
    text_features, image_features, padding_mask = remove_cls(text_features, image_features, padding_mask)

    sim = logit_scale * chunked_matmul(
        text_features=text_features,
        image_features=image_features,
        text_chunk_size=text_chunk_size,
        image_chunk_size=image_chunk_size,
        display_progress=display_progress,
    )
    
    return reduce_token_similarity(sim, padding_mask)
