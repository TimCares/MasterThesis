import logging
from pytorch_lightning import Callback
from typing import *
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

class CodebookUsageCallback(Callback):
    def __init__(self, n_codebook_embed:int):
        super().__init__()
        self.n_codebook_embed = n_codebook_embed

    def on_validation_epoch_start(self, trainer, pl_module):
        self.codebook_cnt = torch.zeros(self.n_codebook_embed, dtype=torch.float64).to(pl_module.device)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.calculate_codebook_usage(pl_module)
        del self.codebook_cnt

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        embed_ind = outputs['embed_ind']
        if trainer.world_size > 1:
            outputs_gather_list = [torch.zeros_like(embed_ind) for _ in range(trainer.world_size)]
            dist.all_gather(outputs_gather_list, embed_ind)
            embed_ind = torch.cat(outputs_gather_list, dim=0).view(-1) # [B * N * Ngpu, ]

        self.codebook_cnt += torch.bincount(embed_ind, minlength=self.n_codebook_embed)

    @torch.no_grad()
    def calculate_codebook_usage(self, pl_module):
        zero_cnt = (self.codebook_cnt == 0).sum()
        
        pl_module.log(
            f"zero_cnt",
            zero_cnt,
            logger=True,
            on_epoch=True,
        )
        pl_module.log(
            f"zero_cnt_percentage",
            (zero_cnt / self.n_codebook_embed) * 100,
            logger=True,
            on_epoch=True,
        )
