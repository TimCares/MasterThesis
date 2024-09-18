import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import opt_einsum
import pytorch_lightning as L
from timm.models.layers import trunc_normal_
from .layers import gather_features, GatherLayer
from .cmli import infer_cmli_logits, masked_mean, max_neg_value, to_half, remove_cls
from typing import Dict
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import logging

logger = logging.getLogger(__name__)

# code for cmli logits heavily borrowed from x-clip -> https://github.com/lucidrains/x-clip
# (https://github.com/lucidrains/x-clip/blob/main/x_clip/x_clip.py)

def mask_eos(padding_masks):
    last_zero_indices = (padding_masks == 0).cumsum(dim=1).argmax(dim=1)
    padding_masks[torch.arange(padding_masks.size(0)), last_zero_indices] = 1
    return padding_masks

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
    def forward(self, image_features, text_features, logit_scale, gather=True):
        device = image_features.device
        if self.world_size > 1 and gather:
            all_image_features, all_text_features = gather_features(
                image_features, text_features
            )

            logits_per_image = logit_scale * image_features @ all_text_features.T
            logits_per_text = logit_scale * text_features @ all_image_features.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        labels = self.get_labels(logits_per_image.shape[0], device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2
        
        out_dict = {
            'loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict
    

class KDClipLoss(CachedLabelContrastiveLoss):
    def forward(self, input_image, input_text, target, logit_scale, gather=True):
        device = input_image.device
        if self.world_size > 1 and gather:
            all_target = GatherLayer.apply(target)
            all_target = torch.cat(all_target)

            logits_per_image = logit_scale * input_image @ all_target.T
            logits_per_text = logit_scale * input_text @ all_target.T
        else:
            logits_per_image = logit_scale * input_image @ target.T
            logits_per_text = logit_scale * input_text @ target.T

        labels = self.get_labels(logits_per_image.shape[0], device)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2
        
        out_dict = {
            'loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
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
    

class KDClipMomentumMemoryBankLoss(nn.Module):
    def __init__(
            self,
            embed_size:int=768,
            size:int=16384, # 2^14
            device:str='cuda',
            world_size:int=1,): 
        super().__init__()
        self.world_size = world_size
        self.size = size
        
        target_mb_ = torch.rand((self.size, embed_size), device=device, requires_grad=False)
        target_mb_ = target_mb_ / target_mb_.norm(dim=-1, keepdim=True)
        self.register_buffer('target_memory_bank', target_mb_)
        self.index_pointer = 0

    def _update(self, target:torch.Tensor) -> None:
        if self.world_size > 1:
            all_target = GatherLayer.apply(target)
            all_target = torch.cat(all_target)

        bsz = all_target.shape[0]
        assert self.size % bsz == 0

        end_idx = self.index_pointer + bsz
        self.target_memory_bank[self.index_pointer:end_idx] = all_target.detach()

        self.index_pointer = end_idx % self.size
    
    def forward(
            self,
            input_image:torch.Tensor,
            input_text:torch.Tensor,
            target:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        return self.compute_loss(
            logit_scale=logit_scale, 
            input_image=input_image,
            input_text=input_text,
            target=target,
        )

    def compute_loss(
            self,
            input_image:torch.Tensor,
            input_text:torch.Tensor,
            target:torch.Tensor,
            logit_scale:torch.Tensor,) -> Dict[str, torch.Tensor]:
        device = input_image.device
        
        all_target = torch.cat([target, self.target_memory_bank], dim=0).t()
        logits_per_image = logit_scale * input_image @ all_target
        logits_per_text = logit_scale * input_text @ all_target

        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)

        total_loss = (image_loss + text_loss) / 2

        out_dict = {
            'loss': total_loss,
            'image_loss': image_loss,
            'text_loss': text_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict

# adapted from VLMo -> https://github.com/microsoft/unilm/blob/master/vlmo/vlmo/modules/objectives.py
class ITMLoss(nn.Module):
    def __init__(self, rank=0, world_size=1,):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def forward(self,
                image_features,
                text_features,
                cls_head,):
        device = image_features.device
        bsz = image_features.shape[0]
        itm_labels = torch.cat([
            torch.ones(bsz), 
            torch.zeros(bsz), 
            torch.zeros(bsz)]).to(device)

        if self.world_size > 1:
            image_features, text_features = gather_features(
                image_features, text_features
            )

        scores = torch.ones(bsz, bsz)
        scores.fill_diagonal_(0)

        neg_text_idx = torch.multinomial(scores, 1).squeeze()
        neg_image_idx = torch.multinomial(scores, 1).squeeze()

        pos_image_text_pairs = image_features[:bsz] + text_features[:bsz]
        neg_image_text_pairs = image_features[:bsz] + text_features[neg_text_idx]
        neg_text_image_samples = image_features[neg_image_idx] + text_features[:bsz]

        examples = torch.concat([pos_image_text_pairs, neg_image_text_pairs, neg_text_image_samples], dim=0)

        logits = cls_head(examples)

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
    
    def forward(self, image_features, text_features, padding_mask, logit_scale=1.0):

        padding_mask = mask_eos(padding_mask)
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
    def __init__(self, cache_labels=False, rank=0, world_size=1, token_fraction=0.25):
        super().__init__(cache_labels, rank, world_size)
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
        text_features:torch.Tensor,
        image_features:torch.Tensor,
        logit_scale:float|torch.Tensor=1.0,
    ) -> torch.Tensor:
    
        text_features = text_features[:, 1:]
        image_features = image_features[:, 1:]
        sim = logit_scale * opt_einsum.contract('x t d, y i d -> x y t i', text_features, image_features)

        text_to_image = einops.reduce(einops.reduce(sim, '... t i -> ... t', 'max'), '... i -> ...', 'mean')

        image_to_text = einops.reduce(einops.reduce(sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

        return {"i2t": image_to_text, "t2i": text_to_image}
    
    def forward(self, image_features, text_features, padding_mask, image_attn, text_attn, logit_scale=1.0):
        text_features, image_features = to_half(text_features, image_features)

        image_features = self.gather_top_tokens(image_features, image_attn)
        text_features = self.gather_top_tokens(text_features, text_attn, padding_mask)

        image_features, text_features = self._gather(
            image_features, text_features
        )

        cmli_logits = infer_cmli_logits(
            text_features=text_features,
            image_features=image_features,
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

class TargetCMLILoss(L.LightningModule):
    def __init__(self, embed_dim, cmli_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.cmli_dim = cmli_dim if cmli_dim is not None else embed_dim
        self.down_proj = nn.Linear(embed_dim, self.cmli_dim)
        self.down_proj.apply(init_bert_params)
    
    def forward(self, image, text, target, padding_mask):

        text_tokens = text[:, 1:]
        target_patches = target[:, 1:]
        padding_mask = mask_eos(padding_mask)[:, 1:]

        text_tokens_proj = self.down_proj(text_tokens)
        target_patches_proj = self.down_proj(target_patches)
        text_tokens_proj_norm = text_tokens_proj / text_tokens_proj.norm(dim=-1, keepdim=True)

        target_patches_proj_norm = target_patches_proj / target_patches_proj.norm(dim=-1, keepdim=True)
        
        similarity = opt_einsum.contract('b i d, b j d -> b i j', text_tokens_proj_norm, target_patches_proj_norm)

        similarity = similarity.max(dim=-1).values
        similarity = similarity.masked_fill(padding_mask.bool(), max_neg_value(similarity.dtype)).max(dim=-1).values

        kd_text_patch_loss = (1 - similarity).mean() # cosine embedding loss
        
        label = torch.ones(image.shape[0], device=image.device)
        kd_text_cls_loss = F.cosine_embedding_loss(input1=text[:, 0].float(), input2=target[:, 0].float(),
                                                   target=label)
        
        kd_text_loss = (kd_text_patch_loss + kd_text_cls_loss) / 2

        kd_image_loss = F.cosine_embedding_loss(input1=image[:, 0].float(), input2=target[:, 0].float(),
                                                target=label)

        total_loss = (kd_text_loss + kd_image_loss) / 2

        out_dict = {
            'kd_loss': total_loss,
            'kd_text_loss': kd_text_loss,
            'kd_text_patch_loss': kd_text_patch_loss,
            'kd_text_cls_loss': kd_text_cls_loss,
            'kd_image_loss': kd_image_loss,
        }
        
        return out_dict


class CosineCMLILoss(nn.Module):
    def __init__(self, align_margin=0.5):
        super().__init__()
        self.align_margin = align_margin
    
    def get_perm(self, B):
        true_order = torch.arange(B)
        perm = torch.randperm(B)

        while (true_order==perm).any().item():
            perm = torch.randperm(B)
        return perm

    def make_features(self, image_features, text_features, padding_mask):
        B = text_features.shape[0]
        perm = self.get_perm(B).to(image_features.device)

        image_features = image_features.repeat(2, 1, 1)
        text_features = torch.concat([text_features, text_features[perm]], dim=0)
        padding_mask = torch.concat([padding_mask, padding_mask[perm]], dim=0)

        target = torch.full((B*2, ), 1).to(text_features.device)
        target[B:] = -1
        return image_features, text_features, padding_mask, target
    
    def infer_cmli_logits(
        self,
        text_features:torch.Tensor,
        image_features:torch.Tensor,
        padding_mask:torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        text_features, image_features = to_half(text_features, image_features)
        text_features, image_features, padding_mask = remove_cls(text_features, image_features, padding_mask)
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        sim = opt_einsum.contract('b i d, b j d -> b i j', text_features, image_features)

        text_to_image = einops.reduce(sim, '... t i -> ... t', 'max')
        text_to_image = masked_mean(text_to_image, padding_mask, dim = -1)

        image_to_text_mask = einops.rearrange(padding_mask, 'b t -> b t 1')
        masked_sim = sim.masked_fill(image_to_text_mask.bool(), max_neg_value(sim.dtype))
        image_to_text = einops.reduce(einops.reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')

        return {"i2t": image_to_text, "t2i": text_to_image}
    
    def _loss(self, cosine_similarity, target):
        pos_loss = 1 - cosine_similarity
        neg_loss = torch.clamp(cosine_similarity - self.align_margin, min=0)
        loss = torch.where(target == 1, pos_loss, neg_loss)
        return loss.mean()

    def forward(self, image_features, text_features, padding_mask):

        image_features, text_features, padding_mask, target = self.make_features(
            image_features=image_features,
            text_features=text_features,
            padding_mask=padding_mask,
        )

        padding_mask = mask_eos(padding_mask)

        cmli_logits = self.infer_cmli_logits(
            text_features=text_features,
            image_features=image_features,
            padding_mask=padding_mask,
        )

        cos_cmli_i2t = self._loss(cmli_logits['i2t'], target)
        cos_cmli_t2i = self._loss(cmli_logits['t2i'], target)
        total_loss = (cos_cmli_i2t + cos_cmli_t2i) / 2
        
        out_dict = {
            'loss': total_loss
        }
        return out_dict
