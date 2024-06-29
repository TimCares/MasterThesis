import torch
import torch.nn.functional as F
from typing import Tuple, Union
from pytorch_lightning import LightningModule
import logging

logger = logging.getLogger(__name__)

class ContrastiveLearningMemoryBankModule(LightningModule):
    def __init__(
            self,
            embed_size:int=768,
            batch_size:int=256,
            start_size:Union[int, None]=None,
            end_size:int=16384, # 2^14
            max_steps:Union[int, None]=None,
            device:str='cuda',
            half_precision:bool=False,): 
        super().__init__()
        assert end_size % batch_size == 0, "Size of memory bank must be multiple of batch size"
        self.start_size = start_size
        self.batch_size = batch_size
        self.end_size = end_size - batch_size # one batch is always new samples
        self.max_steps = max_steps

        if self.start_size is not None:
            assert self.max_steps is not None
            assert self.start_size < self.end_size
            assert self.start_size % batch_size == 0
            self.curr_size = self.start_size
        else:
            self.curr_size = self.end_size
        
        self.image_memory_bank = torch.zeros((self.end_size, embed_size), dtype=torch.float32, device=device)
        self.text_memory_bank = torch.zeros((self.end_size, embed_size), dtype=torch.float32, device=device)
        self.id_memory_bank = torch.zeros(self.end_size, dtype=torch.long, device=device)
        if half_precision:
            self.image_memory_bank = self.image_memory_bank.half()
            self.text_memory_bank = self.text_memory_bank.half()
        self.index_pointer = 0
        self.indexer = torch.zeros(self.end_size, dtype=torch.long, device=device)
        # "indexer" is used to only do contrastive loss on non-empty (non-zero) entries in the memory bank 
        # -> if the memory bank is full, "indexer" will be all ones -> all samples are used

    def _set_size(self, step:int) -> torch.Tensor:
        if self.start_size is None or self.max_steps is None or self.curr_size >= self.end_size:
            return self.curr_size
        n_increases = (self.end_size - self.start_size) // self.batch_size
        interval_increase = self.max_steps // n_increases
        if step % interval_increase == 0:
            self.index_pointer = self.curr_size
            new_size = self.curr_size + self.batch_size
            logger.info(f"Step {step}: Memory bank size increased to {new_size}")
            return new_size
        return self.curr_size

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor, step:int, id:torch.Tensor) -> None:
        self.curr_size = self._set_size(step)
        end_idx = self.index_pointer + self.batch_size
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb.detach()
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb.detach()
        self.id_memory_bank[self.index_pointer:end_idx] = id
        self.indexer[self.index_pointer:end_idx] = 1
        if end_idx == self.curr_size:
            self.index_pointer = 0
        else:
            self.index_pointer = end_idx
    
    def forward(
            self,
            logit_scale:torch.Tensor,
            img_emb:torch.Tensor,
            text_emb:torch.Tensor,
            id:torch.Tensor,
            step:int,
            stage:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute_loss(logit_scale, img_emb, text_emb, id, step, stage)

    def compute_loss(
            self,
            logit_scale:torch.Tensor,
            img_emb:torch.Tensor,
            text_emb:torch.Tensor,
            id:torch.Tensor,
            step:int,
            stage:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert img_emb.size(0) == text_emb.size(0)
        if stage == 'train': # if it is not the training stage, we can use any batch size, as we do not update the memory bank
            assert img_emb.size(0) == self.batch_size

        mask = self.indexer.bool()
        mask[self.curr_size:] = False
        
        logits_per_image = logit_scale * img_emb @ torch.cat([text_emb, self.text_memory_bank[mask]], dim=0).t()
        logits_per_text = logit_scale * text_emb @ torch.cat([img_emb, self.image_memory_bank[mask]], dim=0).t()

        ids = torch.cat([id, self.id_memory_bank[mask]], dim=0)
        target = (id.unsqueeze(1) == ids.unsqueeze(0)).long()

        img_itc_acc = target[torch.arange(target.size(0)), logits_per_image.argmax(dim=1)].mean(dtype=torch.float32)
        text_itc_acc = target[torch.arange(target.size(0)), logits_per_text.argmax(dim=1)].mean(dtype=torch.float32)        

        itc_loss = (
            F.binary_cross_entropy_with_logits(logits_per_image.float(), target)
            + F.binary_cross_entropy_with_logits(logits_per_text.float(), target)
        ) / 2

        if stage == 'train': # we do not want to update the memory bank with batches/samples from the validation set
            self._update(img_emb, text_emb, step, id)
        return itc_loss, img_itc_acc, text_itc_acc

    def log(self, *args, **kwargs):
        super().log(batch_size=self.batch_size, sync_dist=True, *args, **kwargs)
