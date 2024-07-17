import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import gather_features


# The implementation code is modified from open_clip (https://github.com/mlfoundations/open_clip.git)
class ClipLoss(nn.Module):
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

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

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
