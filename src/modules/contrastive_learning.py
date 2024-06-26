import torch
import torch.nn.functional as F
from typing import Tuple, Union
from pytorch_lightning import LightningModule


class ContrastiveLearningMemoryBankModule(LightningModule):
    def __init__(
            self,
            embed_size:int=768,
            batch_size:int=256,
            size:int=16384, # 2^14
            start_decay_rate:float=0.99, # decay rate of 1 means no decay -> all samples are weighted equally
            end_decay_rate:Union[float, None]=None, # 0.98-0.99999
            max_steps:Union[int, None]=None,
            device:str='cuda',
            half_precision:bool=False,
            log_similarity:bool=False,): 
        super().__init__()
        self.batch_size = batch_size
        self.size = size - batch_size # one batch is always new samples
        self.start_decay_rate = start_decay_rate
        self.log_similarity = log_similarity
        if self.start_decay_rate == 1:
            end_decay_rate = None
        self.end_decay_rate = end_decay_rate
        if self.end_decay_rate is not None:
            assert max_steps is not None
        self.max_steps = max_steps
        assert size % batch_size == 0, "Size of memory bank must be multiple of batch size"
        self.image_memory_bank = torch.zeros((self.size, embed_size), dtype=torch.float32, device=device)
        self.text_memory_bank = torch.zeros((self.size, embed_size), dtype=torch.float32, device=device)
        if half_precision:
            self.image_memory_bank = self.image_memory_bank.half()
            self.text_memory_bank = self.text_memory_bank.half()
        self.index_pointer = 0
        self.age = torch.zeros(self.size, dtype=torch.long, device=device)
        self.indexer = torch.zeros(self.size, dtype=torch.long, device=device)
        # "indexer" is used to only do contrastive loss on non-empty (non-zero) entries in the memory bank 
        # -> if the memory bank is full, "indexer" will be all ones -> all samples are used

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor) -> None:
        end_idx = self.index_pointer + self.batch_size
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb.detach()
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb.detach()
        self.age[self.index_pointer:end_idx] = 0
        self.age += 1 # new samples always have an age of 1
        self.indexer[self.index_pointer:end_idx] = 1
        if end_idx == self.size:
            self.index_pointer = 0
        else:
            self.index_pointer = end_idx

    def _get_decay(self, step:int) -> float:
        if self.end_decay_rate is None:
            return self.start_decay_rate
        if step >= self.max_steps:
            return self.end_decay_rate
        return self.start_decay_rate + (self.end_decay_rate - self.start_decay_rate) * (step / self.max_steps)
    
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
            
        sample_size = img_emb.size(0)

        mask = self.indexer.bool()
        # current samples always have an age of 0
        weights = torch.concat([torch.zeros(sample_size, device=self.age.device), self.age[mask]], dim=0)
        decay_rate = self._get_decay(step)
        weights = torch.pow(decay_rate, weights.float())
        weights = weights / weights.sum()
        
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
            F.cross_entropy(logits_per_image.float(), target, weight=weights)
            + F.cross_entropy(logits_per_text.float(), target, weight=weights)
        ) / 2

        if stage == 'train': # we do not want to update the memory bank with batches/samples from the validation set
            self._update(img_emb, text_emb)
        return itc_loss, img_itc_acc, text_itc_acc
    
    def _log_similarity(self, logits: torch.Tensor, stage:str='train') -> None:
        diagonal_mask = torch.eye(logits.size(0)).bool()
        mean_pos_sim = logits[diagonal_mask].mean()
        mean_neg_sim = logits[~diagonal_mask].mean()
        self.log(f"{stage}/itc_mean_pos_similarity", mean_pos_sim)
        self.log(f"{stage}/itc_mean_neg_similarity", mean_neg_sim)

    def log(self, *args, **kwargs):
        super().log(batch_size=self.batch_size, sync_dist=True, *args, **kwargs)
