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
        multimodal=False,
        with_fuzed=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.multimodal = multimodal
        self.with_fuzed = with_fuzed

        if self.multimodal:
            if self.with_fuzed:
                self.experts = ['vl']
            else:
                self.experts = ['image', 'text']
        else:
            self.experts = ['default'] # applies to whatever modality is used for this block

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

    def _check_modality(self, modality:Modality) -> str:
        if self.multimodal:
            if self.with_fuzed:
                return Modality.VL.name.lower()
            else:
                modality_str = modality.name.lower()
                assert modality_str in self.experts
                return modality_str
        else:
            return 'default'

    def forward(self, x, modality:Modality, padding_mask=None, alibi_bias=None):
        modality = self._check_modality(modality)
        
        x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
        r = x = self.norm1[modality](x)
        x = self.mlp[modality](x)
        t = x
        x = self.norm2[modality](r + self.drop_path(self.post_mlp_dropout[modality](x)))

        return x, t
    
    def init_from_pretrained(self, pretained_block:AltBlock, modality:Modality, init_attention:bool) -> None:
        modality_str = self._check_modality(modality)

        self.norm1[modality_str] = pretained_block.norm1
        self.norm2[modality_str] = pretained_block.norm2
        self.mlp[modality_str] = pretained_block.mlp
        self.post_mlp_dropout[modality_str] = pretained_block.post_mlp_dropout

        if init_attention:
            self.init_attention_from_pretrained(pretained_block, modality)

    def init_attention_from_pretrained(self, pretained_block:AltBlock, modality:Modality) -> None:
        if self.attention_pretrained:
            logger.warning("Attention already initialized from pretrained block, not reinitializing")
            return
        
        modality = self._check_modality(modality)

        self.attn = pretained_block.attn
        self.attention_pretrained = True