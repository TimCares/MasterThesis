import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import opt_einsum
from .layers import gather_features
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# code for cmli logits heavily borrowed from x-clip -> https://github.com/lucidrains/x-clip
# (https://github.com/lucidrains/x-clip/blob/main/x_clip/x_clip.py)

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(mask.bool(), 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def infer_cmli_logits(
    text_features:torch.Tensor,
    image_features:torch.Tensor,
    padding_mask:torch.Tensor,
    logit_scale:float|torch.Tensor=1.0,
) -> Dict[str, torch.Tensor]:
    
    text_features = text_features[:, 1:]
    image_features = image_features[:, 1:]
    padding_mask = padding_mask[:, 1:]
    sim = logit_scale * opt_einsum.contract('x t d, y i d -> x y t i', text_features, image_features)

    text_to_image = einops.reduce(sim, '... t i -> ... t', 'max')
    text_to_image_mask = einops.rearrange(padding_mask, 'b t -> 1 b 1 t')
    text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)

    image_to_text_mask = einops.rearrange(padding_mask, 'b t -> b 1 t 1')
    masked_sim = sim.masked_fill(image_to_text_mask.bool(), max_neg_value(sim.dtype))
    image_to_text = einops.reduce(einops.reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

    return {"it2": image_to_text, "t2i": text_to_image}

class CachedLabelContrastiveLoss(nn.Module):
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_labels(self, num_logits, device):
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels


# The implementation code is modified from open_clip (https://github.com/mlfoundations/open_clip.git)
class ClipLoss(CachedLabelContrastiveLoss):
    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features
            )

            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        labels = self.get_labels(logits_per_image.shape[0], device)

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        out_dict = {
            'loss': total_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict


class ClipMomentumMemoryBankLoss(nn.Module):
    def __init__(
            self,
            embed_size:int=768,
            size:int=16384, # 2^14
            half_precision:bool=False,
            device:str='cuda',
            world_size:int=1,): 
        super().__init__()
        self.world_size = world_size
        self.size = size
        assert self.size > 0, "Size of memory bank must be larger than batch size"
        
        if half_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        imb_tmp = torch.rand((self.size, embed_size), dtype=dtype, device=device, requires_grad=False)
        imb_tmp = imb_tmp / imb_tmp.norm(dim=-1, keepdim=True)
        self.register_buffer('image_memory_bank', imb_tmp)

        tmb_tmp = torch.rand((self.size, embed_size), dtype=dtype, device=device, requires_grad=False)
        tmb_tmp = tmb_tmp / tmb_tmp.norm(dim=-1, keepdim=True)
        self.register_buffer('text_memory_bank', tmb_tmp)
        self.index_pointer = 0

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor) -> None:
        """
        img_emb and text_emb must come from momentum encoder
        """
        if self.world_size > 1:
            img_emb, text_emb = gather_features(
                img_emb,
                text_emb,
            )

        bsz = img_emb.shape[0]
        assert bsz == text_emb.shape[0]
        assert self.size % bsz == 0

        end_idx = self.index_pointer + bsz
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb.detach()
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb.detach()

        self.index_pointer = end_idx % self.size
    
    def forward(
            self,
            image_features:torch.Tensor,
            text_features:torch.Tensor,
            image_features_m:torch.Tensor,
            text_features_m:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        return self.compute_loss(
            logit_scale=logit_scale, 
            image_features=image_features,
            text_features=text_features,
            image_features_m=image_features_m,
            text_features_m=text_features_m,
        )

    def compute_loss(
            self,
            image_features:torch.Tensor,
            text_features:torch.Tensor,
            image_features_m:torch.Tensor,
            text_features_m:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        device = image_features.device
        
        logits_per_image = logit_scale * image_features @ torch.cat([text_features_m, self.text_memory_bank], dim=0).t()
        logits_per_text = logit_scale * text_features @ torch.cat([image_features_m, self.image_memory_bank], dim=0).t()

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), labels)
            + F.cross_entropy(logits_per_text.float(), labels)
        ) / 2

        out_dict = {
            'loss': itc_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict

# adapted from VLMo -> https://github.com/microsoft/unilm/blob/master/vlmo/vlmo/modules/objectives.py
class ITMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                all_image_features,
                all_text_features,
                logits_per_image,
                logits_per_text,
                proj_head,):
        device = logits_per_image.device
        bsz = logits_per_image.shape[0]
        itm_labels = torch.cat([
            torch.ones(bsz), 
            torch.zeros(bsz), 
            torch.zeros(bsz)]).to(device)

        with torch.no_grad():
            weights_i2t = F.softmax(logits_per_image[:bsz].float(), dim=1)
            weights_t2i = F.softmax(logits_per_text[:bsz].float(), dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        neg_text_idx = torch.multinomial(weights_i2t, 1).squeeze()
        neg_image_idx = torch.multinomial(weights_t2i, 1).squeeze()

        pos_image_text_pairs = torch.concat([all_image_features[:bsz], all_text_features[:bsz]], dim=1)
        neg_image_text_pairs = torch.concat([all_image_features[:bsz], all_text_features[neg_text_idx]], dim=1)
        neg_text_image_samples = torch.concat([all_image_features[neg_image_idx], all_text_features[:bsz]], dim=1)

        examples = torch.concat([pos_image_text_pairs, neg_image_text_pairs, neg_text_image_samples], dim=0)

        logits = proj_head(examples)

        out_dict = {
            'loss': F.cross_entropy(logits, itm_labels.long()),
            'logits': logits,
            'targets': itm_labels,
        }
        return out_dict

# following FILIP -> https://arxiv.org/pdf/2111.07783
class CMLILoss(CachedLabelContrastiveLoss):
    def __init__(self, cache_labels=False, rank=0, world_size=1):
        super().__init__(cache_labels, rank, world_size)

    def get_labels(self, num_logits, device):
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels
     
    def _gather(self, image_features, text_features, padding_mask):
        if self.world_size > 1:
            all_image_features, all_text_features, all_padding_mask = gather_features(
                image_features, text_features, padding_mask
            )
        else:
            all_image_features = image_features
            all_text_features = text_features
            all_padding_mask = padding_mask
        return all_image_features, all_text_features, all_padding_mask
    
    def _mask_eos(self, padding_masks):
        last_zero_indices = (padding_masks == 0).cumsum(dim=1).argmax(dim=1)
        padding_masks[torch.arange(padding_masks.size(0)), last_zero_indices] = 1
        return padding_masks
    
    def forward(self, image_features, text_features, padding_mask, logit_scale=1.0):
        # following FILIP, we cast to half precision (fp16)
        image_features = image_features.half()
        text_features = text_features.half()

        padding_mask = self._mask_eos(padding_mask)
        image_features, text_features, padding_mask = self._gather(
            image_features, text_features, padding_mask
        )

        cmli_logits = infer_cmli_logits(
            text_features=text_features,
            image_features=image_features,
            padding_mask=padding_mask,
            logit_scale=logit_scale
        )

        logits_per_image = cmli_logits['i2t']
        
        logits_per_text = cmli_logits['t2i']

        labels = self.get_labels(logits_per_image.shape[0], logits_per_image.device)

        cmli_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        out_dict = {
            'loss': cmli_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict
    
class SparseCMLILoss(CMLILoss):
    def __init__(self, cache_labels=False, rank=0, world_size=1, include_cls_token=False, token_fraction=0.25):
        super().__init__(cache_labels, rank, world_size, include_cls_token)
        self.token_fraction = token_fraction
    
    def gather_top_tokens(self, input, attn, padding_mask=None):
        if padding_mask is not None:
            attn = self._mask_eos(attn[:, 0], padding_mask)
        else:
            attn = attn[:, 0]
        top_tokens = attn[:, 1:].topk(k=int(attn.shape[1]*self.token_fraction), dim=-1).indices
        top_tokens += 1
        top_tokens_index = top_tokens.unsqueeze(-1).expand(-1, -1, input.shape[-1])
        return torch.gather(input, 1, top_tokens_index)
    
    def _mask_eos(self, cls_attn, padding_masks):
        last_zero_indices = (padding_masks == 0).cumsum(dim=1).argmax(dim=1)
        cls_attn[torch.arange(cls_attn.size(0)), last_zero_indices] = 0
        return cls_attn
    
    def infer_cmli_logits(
        self,
        q_features:torch.Tensor,
        k_features:torch.Tensor,
        logit_scale:float|torch.Tensor=1.0,
    ) -> torch.Tensor:
        sim = logit_scale * opt_einsum.contract('x t d, y i d -> x y t i', q_features, k_features)

        logits = sim.max(dim=-1).values.mean(dim=-1)

        return logits
    
    def forward(self, image_features, text_features, padding_mask, image_attn, text_attn, logit_scale=1.0):
        # following FILIP, we cast to half precision (fp16)
        image_features = image_features.half()
        text_features = text_features.half()

        image_features = self.gather_top_tokens(image_features, image_attn)
        text_features = self.gather_top_tokens(text_features, text_attn, padding_mask)

        all_image_features, all_text_features = self._gather(
            image_features, text_features
        )

        logits_per_image = self.infer_cmli_logits(
            q_features=image_features,
            k_features=all_text_features,
            logit_scale=logit_scale
        )
        logits_per_text = self.infer_cmli_logits(
            q_features=text_features,
            k_features=all_image_features,
            logit_scale=logit_scale
        )

        labels = self.get_labels(logits_per_image.shape[0], logits_per_image.device)

        cmli_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        out_dict = {
            'loss': cmli_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict

class CMLITargetLoss(nn.Module):
    def forward(self, image, text, target, padding_mask):
        embed_dim = image.size(-1)

        text_norm = text / text.norm(dim=-1, keepdim=True)
        target_norm = target / target.norm(dim=-1, keepdim=True)
        # diagonal = torch.sum(A * B.transpose(-1, -2), dim=-1)
        similarity = text_norm.unsqueeze(1) @ target_norm.unsqueeze(0).transpose(-1, -2)
        similarity = torch.diagonal(similarity).permute(2, 0, 1)

        # exclude cls token (1:)
        similarity = similarity[:, 1:, 1:]

        # +1 because we exclude the cls token before
        token2patch_idx = similarity.argmax(dim=2)+1

        token2patch_idx = token2patch_idx.unsqueeze(-1).expand(-1, -1, embed_dim)

        padding_mask = ~padding_mask[:, 1:].contiguous().view(-1).bool()

        token_aliged_patches = torch.gather(target, 1, token2patch_idx).view(-1, embed_dim)[padding_mask]

        tokens = text[:, 1:].contiguous().view(-1, embed_dim)[padding_mask]

        all_tokens = torch.cat([text[:, :1], tokens], dim=0)
        all_patches = torch.cat([image[:, :1], token_aliged_patches], dim=0)

        kd_text_loss = F.mse_loss(input=all_tokens.float(), target=all_patches.float())

        image_all = image.view(-1, embed_dim).float() # (B, D, C) -> (B*D, C)
        target_all = target.view(-1, embed_dim).float() # (B, D, C) -> (B*D, C)

        kd_image_loss = F.mse_loss(input=image_all, target=target_all)

        total_loss = (kd_text_loss + kd_image_loss) / 2
        
        return total_loss
    

    
    # def forward2(self, image, text, target, padding_masks):

    #     text_norm = text / text.norm(dim=-1, keepdim=True)
    #     target_norm = target / target.norm(dim=-1, keepdim=True)

    #     similarity = text_norm.unsqueeze(1) @ target_norm.unsqueeze(0).transpose(-1, -2)
    #     similarity = torch.diagonal(similarity).permute(2, 0, 1)

    #     # exclude similarity to padding tokens
    #     similarity = similarity.masked_fill(padding_masks[:, :, None].bool(), float('-inf'))

    #     # exclude cls token (1:)
    #     similarity = similarity[:, 1:, 1:]

    #     token2patch_sim, token2patch_idx = similarity.max(dim=2)

    #     token_indices = token2patch_sim.max(dim=1).indices

    #     each_example = torch.arange(token2patch_idx.size(0))

    #     patch_indices = token2patch_idx[each_example, token_indices]

    #     # when indexing, add 1 because we exclude the cls token before
    #     image_patches = target[each_example, patch_indices+1]
    #     text_tokens = text[each_example, token_indices+1]

    #     kd_matching_timesteps_loss = F.mse_loss(input=text_tokens, target=image_patches)

    #     kd_image_cls_loss = F.mse_loss(input=image[:, 0], target=target[:, 0])

    #     if self.text_cls_token:
    #         kd_text_cls_loss= F.mse_loss(input=text[:, 0], target=target[:, 0])
    #         kd_text_loss = (kd_matching_timesteps_loss + kd_text_cls_loss) / 2
    #     else:
    #         kd_text_loss = kd_matching_timesteps_loss

    #     total_loss = (kd_text_loss + kd_image_cls_loss) / 2
        
    #     return total_loss