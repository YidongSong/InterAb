"""The 3D coordinate normalization layer w/ rotational equivariance preserved.

Notes:
* Translational equivariance is no longer preserved after 3D coordinate normalization. Thus, ensure
    that input 3D coordinates fall into one of following conditions:
    > relative 3D coordinates (e.g., z_ij = x_i - x_j)
    > absolute 3D coordinates centralized by subtracting the centroid coordinate
"""

import torch
from torch import nn


class CordNorm(nn.Module):
    """The 3D coordinate normalization layer w/ SE(3)-equivariance preserved."""

    def __init__(self, eps=1e-8, scale_init=1.0):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.eps = eps
        self.scale_init = scale_init

        # setup the learnable scaling factor
        self.scale = nn.Parameter(data=torch.tensor([self.scale_init], dtype=torch.float32))


    def forward(self, cord_tns):
        """Perform the forward pass.

        Args:
        * cord_tns: 3D coordinates of size N1 x N2 x ... Nm x 3

        Returns:
        * cord_tns: normalized 3D coordinates of size N1 x N2 x ... Nm x 3
        """

        cord_tns = self.scale * cord_tns / (torch.norm(cord_tns, dim=-1, keepdim=True) + self.eps)

        return cord_tns
