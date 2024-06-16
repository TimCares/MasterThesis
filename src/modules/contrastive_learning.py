import torch
import torch.nn.functional as F
from typing import Tuple


class ContrastiveLearningMemoryBankModule:
    def __init__(
            self,
            embed_size:int=768,
            batch_size:int=256,
            size:int=16384, # 2^14
            decay_rate:float=0.99,
            schedule_decay_rate:bool=False): # decay rate of 1 means no decay -> all samples are weighted equally
        super().__init__()
        self.batch_size = batch_size
        self.size = size - batch_size # one batch is always new samples
        self.decay_rate = decay_rate
        self.schedule_decay_rate = schedule_decay_rate
        assert size % batch_size == 0, "Size of memory bank must be multiple of batch size"
        self.image_memory_bank = torch.zeros((size, embed_size), dtype=torch.float32)
        self.text_memory_bank = torch.zeros((size, embed_size), dtype=torch.float32)
        self.index_pointer = 0
        self.age = torch.zeros(size, dtype=torch.long)

    def _update(self, img_emb:torch.Tensor, text_emb:torch.Tensor) -> None:
        end_idx = self.index_pointer + self.batch_size
        self.image_memory_bank[self.index_pointer:end_idx] = img_emb
        self.text_memory_bank[self.index_pointer:end_idx] = text_emb
        self.age[self.index_pointer:end_idx] = 0
        self.age += 1 # new samples always have an age of 1
        if end_idx == self.size-1:
            self.index_pointer = 0
        else:
            self.index_pointer = end_idx

    def _get_decay(self) -> float:
        if not self.schedule_decay_rate:
            return self.decay_rate
        return self.initial_decay_rate + (self.final_decay_rate - self.initial_decay_rate) * (self.current_step / self.total_steps)

    def compute_loss(
            self,
            logit_scale:torch.Tensor,
            img_emb:torch.Tensor,
            text_emb:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert img_emb.size(0) == text_emb.size(0) == self.batch_size

        # current samples always have an age of 0
        weights = torch.concat([torch.zeros(self.batch_size, device=self.age.device), self.age], dim=0)
        weights = torch.pow(self.decay_rate, weights.float())
        weights = weights / weights.sum()
        
        scale = logit_scale.exp()
        logits_per_image = scale * img_emb @ torch.concat([text_emb, self.text_memory_bank], dim=0).t()
        logits_per_text = scale * text_emb @ torch.concat([img_emb, self.image_memory_bank], dim=0).t()

        target = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()

        itc_loss = (
            F.cross_entropy(logits_per_image.float()*weights, target, reduction='none')
            + F.cross_entropy(logits_per_text.float()*weights, target, reduction='none')
        ) / 2

        self._update(img_emb, text_emb)
        return itc_loss, img_itc_acc, text_itc_acc
