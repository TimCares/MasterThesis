import torch
import torch.nn.functional as F
from typing import Tuple, Union
from pytorch_lightning import LightningModule


class ContrastiveLearningMemoryBankModule(LightningModule):
    def __init__(
            self,
            embed_size:int=768,
            batch_size:int=256,
            start_size:Union[int, None]=None,
            end_size:int=16384, # 2^14
            max_steps:Union[int, None]=None,
            device:str='cuda',
            half_precision:bool=False,
            log_similarity:bool=False,): 
        super().__init__()
        assert end_size % batch_size == 0, "Size of memory bank must be multiple of batch size"
        self.start_size = start_size
        self.batch_size = batch_size
        self.end_size = end_size - batch_size # one batch is always new samples
        self.max_steps = max_steps

        if self.start_size is not None:
            assert self.max_steps is not None
            assert self.start_size < self.end_size
            assert self.end_size % self.start_size == 0
            self.curr_size = self.start_size
        else:
            self.curr_size = self.end_size
        
        self.log_similarity = log_similarity
        self.image_memory_bank = torch.zeros((self.end_size, embed_size), dtype=torch.float32, device=device)
        self.text_memory_bank = torch.zeros((self.end_size, embed_size), dtype=torch.float32, device=device)
        if half_precision:
            self.image_memory_bank = self.image_memory_bank.half()
            self.text_memory_bank = self.text_memory_bank.half()
        self.index_pointer = 0
        self.indexer = torch.zeros(self.end_size, dtype=torch.long, device=device)
        # "indexer" is used to only do contrastive loss on non-empty (non-zero) entries in the memory bank 
        # -> if the memory bank is full, "indexer" will be all ones -> all samples are used

    def _set_size(self, step:int) -> None:
        if self.start_size is None or self.max_steps is None or step >= self.max_steps:
            return # do nothing
        n_increases = (self.end_size - self.start_size) // self.batch_size
        interval_increase = self.max_steps // n_increases
        if step % interval_increase == 0:
            self.index_pointer = self.curr_size
            self.curr_size += self.batch_size

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor, step:int) -> None:
        self._set_size(step)
        end_idx = self.index_pointer + self.batch_size
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb.detach()
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb.detach()
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
            step:int,
            stage:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.compute_loss(logit_scale, img_emb, text_emb, step, stage)

    def compute_loss(
            self,
            logit_scale:torch.Tensor,
            img_emb:torch.Tensor,
            text_emb:torch.Tensor,
            step:int,
            stage:str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert img_emb.size(0) == text_emb.size(0)
        if stage == 'train': # if it is not the training stage, we can use any batch size, as we do not update the memory bank
            assert img_emb.size(0) == self.batch_size

        mask = self.indexer.bool()
        mask[self.curr_size:] = False
        
        scale = logit_scale.exp()
        logits_per_image = img_emb @ torch.concat([text_emb, self.text_memory_bank[mask]], dim=0).t()
        logits_per_text = text_emb @ torch.concat([img_emb, self.image_memory_bank[mask]], dim=0).t()
        if self.log_similarity:
            self._log_similarity(logits_per_image, stage=stage)
        logits_per_image = logits_per_image * scale
        logits_per_text = logits_per_text * scale

        target = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), target)
            + F.cross_entropy(logits_per_text.float(), target)
        ) / 2

        if stage == 'train': # we do not want to update the memory bank with batches/samples from the validation set
            self._update(img_emb, text_emb, step)
        return itc_loss, img_itc_acc, text_itc_acc
    
    def _log_similarity(self, logits: torch.Tensor, stage:str='train') -> None:
        diagonal_mask = torch.eye(logits.size(0)).bool()
        mean_pos_sim = logits[diagonal_mask].mean()
        mean_neg_sim = logits[~diagonal_mask].mean()
        self.log(f"{stage}/itc_mean_pos_similarity", mean_pos_sim)
        self.log(f"{stage}/itc_mean_neg_similarity", mean_neg_sim)

    def log(self, *args, **kwargs):
        super().log(batch_size=self.batch_size, sync_dist=True, *args, **kwargs)
