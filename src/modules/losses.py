import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


# all following lines pasted 1 to 1 from BEiT-3 -> https://github.com/microsoft/unilm/blob/master/beit3/utils.py
class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)
    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def gather_features(
        image_features,
        text_features,
):
    gathered_image_features = GatherLayer.apply(image_features)
    gathered_text_features = GatherLayer.apply(text_features)
    all_image_features = torch.cat(gathered_image_features)
    all_text_features = torch.cat(gathered_text_features)

    return all_image_features, all_text_features


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

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss, logits_per_image, logits_per_text, labels

class ITMSimilarityLoss(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def forward(self,
                all_image_features,
                all_text_features,
                logits_per_image,
                logits_per_text,
                proj_head,):
        device = logits_per_image.device
        itm_labels = torch.cat([
            torch.ones(self.batch_size), 
            torch.zeros(self.batch_size), 
            torch.zeros(self.batch_size)]).to(device)

        with torch.no_grad():       
            weights_i2t = F.softmax(logits_per_image[:self.batch_size].float(), dim=1)
            weights_t2i = F.softmax(logits_per_text[:self.batch_size].float(), dim=1)
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        neg_text_idx = torch.multinomial(weights_i2t, 1)
        neg_image_idx = torch.multinomial(weights_t2i, 1)

        pos_image_text_pairs = torch.concat([all_image_features[:self.batch_size], all_text_features[:self.batch_size]], dim=1)
        neg_image_text_pairs = torch.concat([all_image_features[:self.batch_size], all_text_features[neg_text_idx]], dim=1)
        neg_text_image_samples = torch.concat([all_image_features[neg_image_idx], all_text_features[:self.batch_size]], dim=1)

        examples = torch.concat([pos_image_text_pairs, neg_image_text_pairs, neg_text_image_samples], dim=0)

        logits = proj_head(examples)

        return F.cross_entropy(logits, itm_labels)

    # def forward(self,
    #             logits_per_image,
    #             logits_per_text,):
    #     device = logits_per_image.device
    #     itm_labels = torch.zeros(self.batch_size).to(device)

    #     with torch.no_grad():       
    #         weights_i2t = F.softmax(logits_per_image[:self.batch_size].float(), dim=1)
    #         weights_t2i = F.softmax(logits_per_text[:self.batch_size].float(), dim=1)
    #         weights_i2t.fill_diagonal_(0)
    #         weights_t2i.fill_diagonal_(0)

    #     neg_text_idx = torch.multinomial(weights_i2t, 1)
    #     neg_image_idx = torch.multinomial(weights_t2i, 1)

    #     pos_idx = torch.arange(self.batch_size, device=device, dtype=torch.long).unsqueeze(1)
    #     image_indices = torch.cat([pos_idx, neg_text_idx], dim=1)
    #     text_indices = torch.cat([pos_idx, neg_image_idx], dim=1)

    #     binary_logits_image = logits_per_image[image_indices]
    #     binary_logits_text = logits_per_text[text_indices]

    #     total_loss = (
    #         F.cross_entropy(binary_logits_image, itm_labels) +
    #         F.cross_entropy(binary_logits_text, itm_labels)
    #         ) / 2
    #     return total_loss
