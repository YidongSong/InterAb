"""Coordinate projection (CP) networks."""

import torch
from torch import nn
import numpy as np

from tfold.utils import cdist
from tfold.tools import DistEncoder


class CPNetV1(nn.Module):
    """Coordinate projection (CP) network - v1."""

    def __init__(self, n_dims_in, n_dims_out):
        """Constructor function."""

        super().__init__()

        self.linear = nn.Linear(n_dims_in, n_dims_out, bias=False)
        self.actv = nn.Sigmoid()


    def forward(self, cord_tns):
        """Perform the forward pass.

        Args:
        * cord_tns: 3D coordinates of size Nv x Gi x 3

        Returns:
        * cord_tns: 3D coordinates of size Nv x Go x 3
        """

        # sub-tract the mean vector from input 3D coordinates
        cord_vec_avg = torch.mean(cord_tns, dim=(0, 1))
        dcrd_tns = cord_tns - cord_vec_avg.view(1, 1, 3)

        # apply the linear mapping & non-linear activation
        dcrd_tns = self.linear(dcrd_tns.permute(0, 2, 1)).permute(0, 2, 1)
        dcrd_tns = self.actv(torch.norm(dcrd_tns, dim=2, keepdim=True)) * dcrd_tns

        # add the mean vector back to output 3D coordinates
        cord_tns = dcrd_tns + cord_vec_avg.view(1, 1, 3)

        return cord_tns


class CPNetV2(nn.Module):
    """Coordinate projection (CP) network - v2.

    Note:
    * It is guaranteed that all the projected points lies within the convex hull of input points.
    """

    def __init__(self, n_dims_in, n_dims_out):
        """Constructor function."""

        super().__init__()

        # initialization
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out

        # sub-networks
        self.dist_encoder = DistEncoder()
        self.n_dims_encd = self.dist_encoder.n_dims
        self.linear = nn.Linear(self.n_dims_encd, self.n_dims_out)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, cord_tns):
        """Perform the forward pass.

        Args:
        * cord_tns: 3D coordinates of size Nv x Gi x 3

        Returns:
        * cord_tns: 3D coordinates of size Nv x Go x 3
        """

        # initialization
        n_nodes = cord_tns.shape[0]

        # calculate the pairwise distance matrix
        dist_tns = cdist(cord_tns)  # Nv x Gi x Gi
        encd_tns = self.dist_encoder.run(dist_tns.view(-1)).view(
            n_nodes, self.n_dims_in, self.n_dims_in, self.n_dims_encd)  # Nv x Gi x Gi x De
        feat_tns = torch.mean(encd_tns, dim=2)  # Nv x Gi x De

        # calculate the coefficient matrix
        coef_tns = self.softmax(self.linear(feat_tns))  # Nv x Gi x Go

        # calculate output 3D coordinates
        cord_tns = torch.sum(coef_tns.unsqueeze(dim=3) * cord_tns.unsqueeze(dim=2), dim=1)

        return cord_tns


class CPNetV3(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Coordinate projection (CP) network - v3.

    Note:
    * It is guaranteed that all the projected points lies within the convex hull of input points.
    """

    def __init__(self, n_dims_in, n_dims_out, n_dims_attn=16):
        """Constructor function."""

        super().__init__()

        # initialization
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out
        self.n_dims_attn = n_dims_attn

        # sub-networks
        self.dist_encoder = DistEncoder()
        self.n_dims_encd = self.dist_encoder.n_dims
        self.linear_q = nn.Linear(self.n_dims_encd, self.n_dims_out * self.n_dims_attn)
        self.linear_k = nn.Linear(self.n_dims_encd, self.n_dims_out * self.n_dims_attn)
        self.div_fctr = np.sqrt(self.n_dims_attn)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, cord_tns):
        """Perform the forward pass.

        Args:
        * cord_tns: 3D coordinates of size Nv x Gi x 3

        Returns:
        * cord_tns: 3D coordinates of size Nv x Go x 3
        """

        # initialization
        n_nodes = cord_tns.shape[0]

        # calculate the pairwise distance matrix
        dist_tns = cdist(cord_tns)  # Nv x Gi x Gi
        encd_tns = self.dist_encoder.run(dist_tns.view(-1)).view(
            n_nodes, self.n_dims_in, self.n_dims_in, self.n_dims_encd)  # Nv x Gi x Gi x De
        feat_tns = torch.mean(encd_tns, dim=2)  # Nv x Gi x De

        # calculate query & key embeddings
        q_tns = self.linear_q(feat_tns).view(
            n_nodes, self.n_dims_in, self.n_dims_out, self.n_dims_attn)
        k_tns = self.linear_k(feat_tns).view(
            n_nodes, self.n_dims_in, self.n_dims_out, self.n_dims_attn)
        coef_tns = self.softmax(torch.sum(q_tns * k_tns, dim=-1) / self.div_fctr)  # Nv x Gi x Go

        # calculate output 3D coordinates
        cord_tns = torch.sum(coef_tns.unsqueeze(dim=3) * cord_tns.unsqueeze(dim=2), dim=1)

        return cord_tns
