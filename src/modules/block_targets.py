from typing import *
import torch.nn as nn
import torch
from timm.models.vision_transformer import Attention, LayerScale
from timm.layers import DropPath, Mlp

# after "Block" from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class LayerResultBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            layer_norm_first: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_norm_first = layer_norm_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_norm_first:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = self.ls2(self.mlp(self.norm2(x)))
            t = x
            x = x + self.drop_path2(x)
            return x, t
        else:
            x = x + self.drop_path1(self.ls1(self.attn(x)))
            r = x = self.norm1(x)
            x = self.ls2(self.mlp(x))
            t = x
            x = self.norm2(r + self.drop_path2(x))
            return x, t
