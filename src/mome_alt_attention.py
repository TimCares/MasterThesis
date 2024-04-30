import torch.nn as nn
from data2vec_fairseq.models.modalities.modules import AltAttention, AltBlock
import logging

logger = logging.getLogger(__name__)

# adapted from: https://arxiv.org/pdf/2212.07525.pdf
# source: https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/modalities/modules.py
class MOMEAltBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
        cosine_attention=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.ModuleDict({
            'audio': norm_layer(dim),
            'image': norm_layer(dim),
            'text': norm_layer(dim),
            # 'all': norm_layer(dim),
        })

        mlp_hidden_dim = int(dim * mlp_ratio)
        def make_mlp():
            return Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=mlp_drop,
            )
        self.mlp = nn.ModuleDict({
            'audio': make_mlp(),
            'image': make_mlp(),
            'text': make_mlp(),
            # 'all': make_mlp(),
        })

        self.post_mlp_dropout = nn.ModuleDict({
            'audio': nn.Dropout(post_mlp_drop, inplace=False),
            'image': nn.Dropout(post_mlp_drop, inplace=False),
            'text': nn.Dropout(post_mlp_drop, inplace=False),
            # 'all': nn.Dropout(post_mlp_drop, inplace=False),
        })

        self.attention_pretrained = False

    def forward(self, x, mode:str, padding_mask=None, alibi_bias=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            r = x = self.mlp[mode](self.norm2[mode](x))
            t = x
            x = r + self.drop_path(self.post_mlp_dropout[mode](x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp[mode](x)
            t = x
            x = self.norm2[mode](r + self.drop_path(self.post_mlp_dropout[mode](x)))
            if not self.ffn_targets:
                t = x

        return x, t
    
    def init_from_pretrained(self, pretained_block:AltBlock, mode:str, init_attention:bool) -> None:
        self.norm2[mode] = pretained_block.norm2
        self.mlp[mode] = pretained_block.mlp
        self.post_mlp_dropout[mode] = pretained_block.post_mlp_dropout

        if init_attention and not self.attention_pretrained:
            self.attn = pretained_block.attn
            self.norm1 = pretained_block.norm1
            self.attention_pretrained = True
        elif init_attention and self.attention_pretrained:
            logger.warning("Attention already initialized from pretrained block, not reinitializing")
