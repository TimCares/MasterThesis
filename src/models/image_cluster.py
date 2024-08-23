import torch
from torch import nn
import os
import logging
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from omegaconf import OmegaConf
from utils import freeze_module, load_beit2_teacher


logger = logging.getLogger(__name__)

class ImageCluster(nn.Module):
    def __init__(self,
                 cfg,
                 ):
        super(ImageCluster, self).__init__()
        self.cfg = cfg
        self.initted = False
        
        beit2_kwargs = OmegaConf.to_container(self.cfg.beitv2, resolve=True)
        sd_path = beit2_kwargs.pop("model_path")
        sd_name = beit2_kwargs.pop("model_name")
        beit_path = os.path.join(sd_path, sd_name)

        self.beitv2:VisionTransformerForMaskedImageModeling = load_beit2_teacher(
            sd_path=beit_path,
            **beit2_kwargs,
        )
        freeze_module(self.beitv2)

        self.cluster_prototypes_ = torch.empty(self.cfg.num_clusters, self.beitv2.embed_dim)
        self.register_buffer('cluster_prototypes', self.cluster_prototypes_)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        for k in list(sd.keys()):
            if k.startswith('beitv2.'):
                del sd[k]
        return sd


    def forward(
        self,
        image:torch.Tensor,
    ):
        bool_masked_pos = torch.zeros((image.shape[0], self.beitv2.patch_embed.num_patches),
                                      dtype=torch.bool).to(image.device)
        
        with torch.no_grad():
            cls_token = self.beitv2.forward_features(
                x=image,
                bool_masked_pos=bool_masked_pos,
            )[:, 0]

        cls_token = cls_token / cls_token.norm(dim=-1, keepdim=True)
        
        return cls_token
        
    
    def get_cluster(
        self,
        image:torch.Tensor,
    ):
        cls_token = self(image)

        cluster_scores = cls_token @ self.cluster_prototypes.t()

        out_dict = {
            'cluster_idx': cluster_scores.argmax(dim=-1),
            'cluster_dist': cluster_scores,
            'cls_token': cls_token,
        }
        return out_dict
