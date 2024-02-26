import torch
import torch.nn as nn

class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.channels = 512
        self.strides = [5,2,2,2,2,2,2]
        self.kernel_widths = [10,3,3,3,3,2,2]
        list_conv = []
        list_ln = []
        for i, (kw, s) in enumerate(zip(self.kernel_widths, self.strides)):
            if i == 0:
                list_conv.append(nn.Conv1d(1, self.channels, kernel_size=kw, stride=s, padding=kw//2)) # TODO check padding
            else:
                list_conv.append(nn.Conv1d(self.channels, self.channels, kernel_size=kw, stride=s, padding=kw//2))
            
            list_ln.append(nn.LayerNorm(self.channels))

        self.convs = nn.ModuleList(list_conv)
        self.layer_norms = nn.ModuleList(list_ln)
        self.gelu = nn.GELU()

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = x.transpose(1, 2) # so that normalization is done over the channels
            x = self.layer_norms[i](x)
            x = x.transpose(1, 2) # back to the original shape
            x = self.gelu(x)
        return x