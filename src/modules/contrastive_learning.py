import torch
import torch.nn.functional as F
from typing import Dict
from pytorch_lightning import LightningModule
import logging
from .layers import gather_features

logger = logging.getLogger(__name__)

class ContrastiveLearningMemoryBankModule(LightningModule):
    def __init__(
            self,
            embed_size:int=768,
            batch_size:int=256,
            size:int=16384, # 2^14
            device:str='cuda',
            half_precision:bool=False,
            world_size:int=1,
            rank:int=0,): 
        super().__init__()
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size*self.world_size
        assert size % self.batch_size == 0, "Size of memory bank must be multiple of batch size"
        self.size = size - batch_size # one batch is always new samples
        assert self.size > 0, "Size of memory bank must be larger than batch size"
        
        if half_precision:
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        imb_tmp = torch.rand((self.size, embed_size), dtype=dtype, device=self.device, requires_grad=False)
        imb_tmp = imb_tmp / imb_tmp.norm(dim=-1, keepdim=True)
        self.register_buffer('image_memory_bank', imb_tmp)

        tmb_tmp = torch.rand((self.size, embed_size), dtype=dtype, device=self.device, requires_grad=False)
        tmb_tmp = tmb_tmp / tmb_tmp.norm(dim=-1, keepdim=True)
        self.register_buffer('text_memory_bank', tmb_tmp)
        self.index_pointer = 0

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor) -> None:
        if self.world_size > 1:
            img_emb, text_emb = gather_features(
                img_emb, text_emb
            )
        end_idx = self.index_pointer + self.batch_size
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb.detach()
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb.detach()
        if end_idx == self.size:
            self.index_pointer = 0
        else:
            self.index_pointer = end_idx
    
    def forward(
            self,
            logit_scale:torch.Tensor,
            img_emb:torch.Tensor,
            text_emb:torch.Tensor,
            stage:str) -> Dict[str, torch.Tensor]:
        return self.compute_loss(logit_scale, img_emb, text_emb, stage)

    def compute_loss(
            self,
            logit_scale:torch.Tensor,
            img_emb:torch.Tensor,
            text_emb:torch.Tensor,
            stage:str) -> Dict[str, torch.Tensor]:
        assert img_emb.size(0) == text_emb.size(0)
        if stage == 'train': # if it is not the training stage, we can use any batch size, as we do not update the memory bank
            assert img_emb.size(0) == self.batch_size
        
        logits_per_image = logit_scale * img_emb @ torch.cat([text_emb, self.text_memory_bank], dim=0).t()
        logits_per_text = logit_scale * text_emb @ torch.cat([img_emb, self.image_memory_bank], dim=0).t()

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or self.device not in self.labels:
            labels = torch.arange(num_logits, device=self.device, dtype=torch.long)
            if self.world_size > 1:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[self.device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[self.device]

        img_itc_acc = (logits_per_image.argmax(dim=1) == labels).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == labels).float().mean()

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), labels)
            + F.cross_entropy(logits_per_text.float(), labels)
        ) / 2

        if stage == 'train': # we do not want to update the memory bank with batches/samples from the validation set
            self._update(img_emb, text_emb)

        out_dict = {
            'loss': itc_loss,
            'img_itc_acc': img_itc_acc,
            'text_itc_acc': text_itc_acc,
            'acc': (img_itc_acc + text_itc_acc) / 2,
        }
        return out_dict
