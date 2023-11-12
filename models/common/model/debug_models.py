import torch
import torch.nn as nn

class InterceptOnlyModel(nn.Module):
    def __init__(self, d_in, d_out):
        super(InterceptOnlyModel, self).__init__()
        self.bias = nn.Parameter(torch.zeros(d_out))

    def forward(self, x):
        return self.bias.unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), -1)