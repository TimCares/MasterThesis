import torch.nn as nn

class TransposeLast(nn.Module):
    def __init__(self, tranpose_dim=-2):
        super().__init__()
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        return x.transpose(self.tranpose_dim, -1)