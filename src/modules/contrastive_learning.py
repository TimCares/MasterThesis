import torch
import torch.nn.functional as F
from typing import Tuple


class ContrastiveLearningModule:
    def __init__(self):
        super().__init__()
        self.acc_img_emb = []
        self.acc_text_emb = []

    def compute_loss(self, logit_scale:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_emb = torch.cat(self.acc_img_emb, 0)
        text_emb = torch.cat(self.acc_text_emb, 0)

        self.acc_img_emb = []
        self.acc_text_emb = []

        logits_per_image = logit_scale.exp() * img_emb @ text_emb.t()
        logits_per_text = logits_per_image.t()

        target = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

        img_itc_acc = (logits_per_image.argmax(dim=1) == target).float().mean()
        text_itc_acc = (logits_per_text.argmax(dim=1) == target).float().mean()

        itc_loss = (
            F.cross_entropy(logits_per_image.float(), target)
            + F.cross_entropy(logits_per_text.float(), target)
        ) / 2
        return itc_loss, img_itc_acc, text_itc_acc
    
    def add_emb_pair(self, img_emb:torch.Tensor, text_emb:torch.Tensor) -> None:
        self.acc_img_emb.append(img_emb)
        self.acc_text_emb.append(text_emb)