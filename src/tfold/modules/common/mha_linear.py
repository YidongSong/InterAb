"""Multi-head linear layer."""

import numpy as np
import torch
from torch import nn


class MhaLinear(nn.Module):
    """Multi-head linear layer.

    This module performs the following operation:
      BS x (H x D_i) => BS x (H x D_o)
    which is parameterized by <H> weighting matrices, each of size <D_i x D_o>.

    It is not required for the input tensor to match the BS x (H x D_i) shape.
    As long as the total size matches, we will do the reshaping automatically.
    """

    def __init__(self, n_heads, n_dims_in, n_dims_out, enbl_split=False):
        """Constructor function."""

        super().__init__()

        self.n_heads = n_heads
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.enbl_split = enbl_split

        stdev = np.sqrt(2.0 / (n_dims_in + n_dims_out))
        self.weights = nn.Parameter(stdev * torch.randn(n_heads, n_dims_in, n_dims_out))


    def forward(self, inputs):
        """Perform the forward pass.

        Args:
        * inputs: input tensor of size BS x (H x D_i) / BS x H x D_i

        Returns:
        * outputs: output tensor of size BS x (H x D_o) / BS x H x D_o
        """

        assert inputs.numel() % (self.n_heads * self.n_dims_in) == 0, \
            f'invalid input tensor shape: {inputs.shape}'

        inputs_perm = inputs.view(-1, self.n_heads, self.n_dims_in).permute(1, 0, 2)
        outputs = torch.bmm(inputs_perm, self.weights).permute(1, 0, 2)
        if not self.enbl_split:
            outputs = torch.reshape(outputs, [-1, self.n_heads * self.n_dims_out])

        return outputs


    def __repr__(self):
        """Get the string representation."""

        config_str = ', '.join([
            f'n_heads={self.n_heads}',
            f'n_dims_in={self.n_dims_in}',
            f'n_dims_out={self.n_dims_out}',
            f'enbl_split={self.enbl_split}',
        ])
        repr_str = f'MhaLinear({config_str})'

        return repr_str
