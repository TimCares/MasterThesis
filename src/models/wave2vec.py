from typing import *
import torch
import torch.nn as nn

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super(ConvFeatureExtractionModel).__init__()

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):

            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(n_out, elementwise_affine=True), # "n_out" was "dim" in the original code
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        # x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

channels = [512, 512, 512, 512, 512, 512, 512]
strides = [5,2,2,2,2,2,2]
kernel_widths = [10,3,3,3,3,2,2]

feature_extractor_args = zip(channels, kernel_widths, strides)
class MMData2Vec(nn.Module):
    def __init__(self, embed_dim:int, audio_feature_encoder_args: List[Tuple[int, int, int]]):
        super(MMData2Vec, self).__init__()
        
        self.audio_feature_extractor = ConvFeatureExtractionModel(audio_feature_encoder_args, mode='layer_norm')

        feature_embed_dim = audio_feature_encoder_args[-1][0] # channels of last conv layer

        self.feature_projector = nn.Sequential(
            TransposeLast(),
            nn.LayerNorm(feature_embed_dim),
            nn.Linear(feature_embed_dim, embed_dim),
        ) if embed_dim != feature_embed_dim else None


    def forward(self, x):
        pass