import torch
import torch.nn as nn

from tfold.modules.xfold.utils import default, exists
from tfold.modules.xfold.network.modules import DirectMultiheadAttention, FeedForwardLayer
from tfold.modules.xfold.network.attention import Attention, TriangleMultiplicativeModule

# Initialization
def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def init_zero_mlp(module):
    if isinstance(module, FeedForwardLayer):
        init_zero_(module.linear2)
    elif isinstance(module, Attention):
        init_zero_(module.to_out)
    elif isinstance(module, TriangleMultiplicativeModule):
        init_zero_(module.to_out)
    elif isinstance(module, DirectMultiheadAttention):
        init_zero_(module.proj_out)
