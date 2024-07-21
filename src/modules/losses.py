import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import gather_features
from typing import Dict
import logging

logger = logging.getLogger(__name__)

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

        all_image_features = all_image_features if self.world_size > 1 else image_features
        all_text_features = all_text_features if self.world_size > 1 else text_features

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        out_dict = {
            'loss': total_loss,
            'all_image_features': all_image_features,
            'all_text_features': all_text_features,
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
    def __init__(self, cache_labels=False, rank=0, world_size=1, include_cls_token=False):
        super().__init__(cache_labels, rank, world_size)
        self.include_cls_token = include_cls_token
     
    def _gather(self, image_features, text_features, padding_masks):
        if self.world_size > 1:
            all_image_features, all_text_features, all_padding_masks = gather_features(
                image_features, text_features, padding_masks
            )
        else:
            all_image_features = image_features
            all_text_features = text_features
            all_padding_masks = padding_masks
        return all_image_features, all_text_features, all_padding_masks
    
    def forward(self, image, text, padding_masks, logit_scale=1.0):
        image, text, padding_masks = self._gather(image, text, padding_masks)

        similarity = logit_scale * image.unsqueeze(1) @ text.unsqueeze(0).transpose(-1, -2)

        # exclude similarity to padding tokens
        similarity = similarity.masked_fill(padding_masks[:, None, None, :].bool(), float('nan'))

        if self.include_cls_token:
            logits_per_cls_image = similarity[:, :, 0, 0]
            logits_per_cls_text = logits_per_cls_image.t()
        
        # exclude cls token (1:)
        similarity = similarity[:, :, 1:, 1:]

        logits_per_image = similarity.nan_to_num(float('-inf')).max(dim=3).values.mean(dim=2)
        logits_per_text = similarity.max(dim=2).values.nanmean(dim=2)

        labels = self.get_labels(logits_per_image.shape[0], logits_per_image.device)

        cmli_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        
        if self.include_cls_token:
            itc_loss = (
                F.cross_entropy(logits_per_cls_image, labels) +
                F.cross_entropy(logits_per_cls_text, labels)
                ) / 2
            total_loss = (cmli_loss + itc_loss) / 2
        else:
            total_loss = cmli_loss
        
        out_dict = {
            'loss': total_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict

class CMLITargetLoss(nn.Module):
    def __init__(self, text_cls_token=False):
        self.text_cls_token = text_cls_token
    
    def forward(self, image, text, target, padding_masks):

        similarity = text.unsqueeze(1) @ image.unsqueeze(0).transpose(-1, -2)

        # exclude similarity to padding tokens
        similarity = similarity.masked_fill(padding_masks[:, None, None, :].bool(), float('nan'))

        kd_image_cls_loss = F.mse_loss(input=image[:, 0], target=target[:, 0])
        
        # exclude cls token (1:)
        similarity = similarity[:, :, 1:, 1:]

        token2patch = similarity.max(dim=3).indices

        mask = torch.eye(image.shape[0], device=image.device, dtype=torch.bool)

        token2patch = token2patch[mask] # (bsz, num_tokens)

        F.mse_loss(input=text, target=image[token2patch])


        if self.text_cls_token:
            kd_text_cls_loss= F.mse_loss(input=text[:, 0], target=target[:, 0])
        
        out_dict = {
            'loss': total_loss,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'targets': labels,
        }
        return out_dict