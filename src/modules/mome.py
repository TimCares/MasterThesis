import torch.nn as nn
from data2vec_fairseq.models.modalities.modules import AltAttention, AltBlock
import logging
from timm.models.vision_transformer import DropPath, Mlp
from data2vec_fairseq.data.modality import Modality

logger = logging.getLogger(__name__)

# adapted from: https://arxiv.org/pdf/2212.07525.pdf
# source: https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/modalities/modules.py
class MOMEBlock(nn.Module):
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
        shared_attn=True,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.multimodal = multimodal
        self.with_fuzed = with_fuzed
        self.shared_attn = shared_attn

        if self.multimodal:
            if self.with_fuzed:
                self.experts = ['vl']
                self.shared_attn = True
            else:
                self.experts = ['image', 'text']
        else:
            self.experts = ['default'] # applies to whatever modality is used for this block
            self.shared_attn = True

        def make_attn():
            return AltAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                cosine_attention=cosine_attention,
            )

        if self.shared_attn:
            self.attn = nn.ModuleDict({'default': make_attn()})
        else:
            self.attn = nn.ModuleDict({modality: make_attn() for modality in self.experts})

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

    def forward(self, x, modality:Modality, padding_mask=None, return_ffn:bool=True):
        modality = self._check_modality(modality)
        attn_key = 'default' if self.shared_attn else modality
        
        x = x + self.drop_path(self.attn[attn_key](x, padding_mask, alibi_bias, return_attn_scores=False))
        r = x = self.norm1[modality](x)
        x = self.mlp[modality](x)
        if return_ffn:
            t = x
        x = self.norm2[modality](r + self.drop_path(self.post_mlp_dropout[modality](x)))

        if return_ffn:
            return x, t
        return x
    
    def init_from_pretrained(self, pretained_block:AltBlock, modality:Modality, init_attention:bool) -> None:
        modality_str = self._check_modality(modality)

        self.norm1[modality_str] = pretained_block.norm1
        self.norm2[modality_str] = pretained_block.norm2
        self.mlp[modality_str] = pretained_block.mlp
        self.post_mlp_dropout[modality_str] = pretained_block.post_mlp_dropout

        if init_attention:
            self.init_attention_from_pretrained(pretained_block, modality)

    def init_attention_from_pretrained(self, pretained_block:AltBlock, modality:Modality) -> None:
        if self.shared_attn and self.attention_pretrained:
            logger.warning("Attention already initialized from pretrained block, not reinitializing")
            return
        
        attn_key = 'default' if self.shared_attn else modality.name.lower()

        self.attn[attn_key] = pretained_block.attn
        self.attention_pretrained = True

    def remove_modality(self, modality:Modality) -> None:
        if not self.multimodal:
            return # only one modality, removing it would remove the entire block
        
        modality_str = modality.name.lower()

        self.norm1.pop(modality_str)
        self.norm2.pop(modality_str)
        self.mlp.pop(modality_str)
        self.post_mlp_dropout.pop(modality_str)

        if not self.shared_attn:
            self.attn.pop(modality_str)