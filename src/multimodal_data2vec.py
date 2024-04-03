import torch
from torch import nn
from functools import partial
from typing import Dict, Any, List
import numpy as np

from examples.data2vec.models.modalities.modules import AltBlock
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from examples.data2vec.models.modalities.base import MaskSeed

class SimpleKDMultiModalData2Vec(nn.Module):
    def __init__(self,
                 image_encoder:nn.Module,
                 text_encoder:nn.Module,
                 audio_encoder:nn.Module,
                 encoders_embed_dim:int,
                 embed_dim:int,
                 num_heads:int,
                 mlp_ratio:float,
                 qkv_bias:bool,
                 encoder_dropout:float,
                 attention_dropout:float,
                 activation_dropout:float,
                 post_mlp_drop:float,
                 norm_eps:float,
                 norm_affine:bool,
                 layer_norm_first:bool,
                 ffn_targets:bool,
                 average_top_k_layers:int,
                 loss_beta:float,
                 loss_scale:float,
                 dropout_input:float,
                 depth:int,
                 start_drop_path_rate:float,
                 end_drop_path_rate:float,
                 layerdrop:float,
                 seed:int,
                 ):
        super().__init__()
        self.modality_encoders = nn.ModuleDict({
            "image": image_encoder,
            "text": text_encoder,
            "audio": audio_encoder
        })
        self.modality_encoders.eval()
        self.modality_encoders.is_pretrained = True

        if encoders_embed_dim != embed_dim:
            self.proj = nn.Linear(encoders_embed_dim, embed_dim)

        make_layer_norm = partial(
            nn.LayerNorm, eps=norm_eps, elementwise_affine=norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                dim=embed_dim if dim is None else dim,
                num_heads=num_heads if heads is None else heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=encoder_dropout,
                attn_drop=attention_dropout,
                mlp_drop=activation_dropout,
                post_mlp_drop=post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=layer_norm_first,
                ffn_targets=ffn_targets,
            )
        
        self.alibi_biases = {}

        self.average_top_k_layers = average_top_k_layers
        self.loss_beta = loss_beta
        self.loss_scale = loss_scale

        self.dropout_input = nn.Dropout(dropout_input)

        dpr = np.linspace(start_drop_path_rate, end_drop_path_rate, depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(depth)])

        self.norm = None
        if layer_norm_first:
            self.norm = make_layer_norm(embed_dim)

        self.layerdrop = layerdrop
        self.seed = seed

        self.apply(self._init_except_pretrained)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}

        self.num_updates = 0

    def _init_except_pretrained(self, module:nn.Module):
        if hasattr(module, "is_pretrained") and module.is_pretrained:
            return
        else:
            module.apply(init_bert_params)

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def forward(
        self,
        image,
        text,
        audio,
        target=None,
        id=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
    ):
        
        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.seed, update=self.num_updates, ids=id)
        
        extractor_outputs = []
        for mode, input in zip(["image", "text", "audio"], [image, text, audio]):
            if input is not None:
                feature_extractor = self.modality_encoders[mode]
                with torch.no_grad():
                    extractor_out = feature_extractor(
                        input,
                        padding_mask,
                        mask,
                        remove_masked=not features_only or force_remove_masked,
                        clone_batch=self.cfg.clone_batch if not features_only else 1,
                        mask_seeds=mask_seeds,
                        precomputed_mask=precomputed_mask,
                    )

                if self.proj is not None:
                    extractor_out = self.proj(extractor_out)
                extractor_outputs.append(extractor_out)

        x = extractor_outputs[0]

        # if len(extractor_outputs) == 1:        
        #     x = extractor_outputs[0]
        # else:
        #     x = torch.cat(extractor_outputs, dim=1) # TODO: look into it later when input is multimodal...

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)

        if self.norm is not None:
            x = self.norm(x)

        if remove_extra_tokens:
            x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
            if masked_padding_mask is not None:
                masked_padding_mask = masked_padding_mask[
                    :, feature_extractor.modality_cfg.num_extra_tokens :
                ]

        return {
            "x": x,
            "padding_mask": masked_padding_mask,
            "layer_results": layer_results,
            "mask": encoder_mask,
        }