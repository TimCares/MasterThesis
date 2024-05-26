import torch.nn as nn
from data2vec_fairseq.models.modalities.modules import AltAttention, AltBlock
import logging
from timm.models.vision_transformer import DropPath, Mlp
from data2vec_fairseq.data.modality import Modality

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
        cosine_attention=False,
        with_fuzed=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.with_fuzed = with_fuzed

        if self.with_fuzed:
            self.experts = ['vl']
        else:
            self.experts = ['image', 'text']

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
        self.norm1 = nn.ModuleDict({modality: norm_layer(dim) for modality in self.experts})
        self.norm2 = nn.ModuleDict({modality: norm_layer(dim) for modality in self.experts})

        mlp_hidden_dim = int(dim * mlp_ratio)
        def make_mlp():
            return Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=mlp_drop,
            )
        self.mlp = nn.ModuleDict({modality: make_mlp() for modality in self.experts})

        self.post_mlp_dropout = nn.ModuleDict({modality: nn.Dropout(post_mlp_drop) for modality in self.experts})

        self.attention_pretrained = False

    def forward(self, x, modality:Modality, padding_mask=None, alibi_bias=None):
        if self.with_fuzed:
            modality = Modality.VL.name.lower()
        else:
            modality = modality.name.lower()
        
        x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
        r = x = self.norm1[modality](x)
        x = self.mlp[modality](x)
        t = x
        x = self.norm2[modality](r + self.drop_path(self.post_mlp_dropout[modality](x)))

        return x, t
    
    def init_from_pretrained(self, pretained_block:AltBlock, modality:Modality, init_attention:bool) -> None:
        if self.with_fuzed:
            modality = Modality.VL.name.lower()
        else:
            modality = modality.name.lower()
        self.norm1[modality] = pretained_block.norm1
        self.norm2[modality] = pretained_block.norm2
        self.mlp[modality] = pretained_block.mlp
        self.post_mlp_dropout[modality] = pretained_block.post_mlp_dropout

        if init_attention and not self.attention_pretrained:
            self.attn = pretained_block.attn
            self.attention_pretrained = True
        elif init_attention and self.attention_pretrained:
            logger.warning("Attention already initialized from pretrained block, not reinitializing")
