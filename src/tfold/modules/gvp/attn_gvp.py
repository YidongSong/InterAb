"""Attention-based geometric vector perceptron."""

import torch
from torch import nn
import numpy as np

from tfold.utils import cdist
from tfold.modules.gvp.gvp import GVP


class AttnGVP(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Attention-based geometric vector perceptron.

    Workflow:
    1. Update scalar & vector features in a per-node manner.
    2. Calculate attention weights from input (not updated) scalar & vector features.
    3. Update scalar & vector features with attention weights.

    Note:
    * Since scalar & vector features are updated with residual connections, input/output dimensions
        are required to be the same. Otherwise, use <GVP> instead.
    * Hidden scalar & output features, which are outputs from <GVP>, may have different numbers of
        dimensions from input/output scalar & output features.
    """

    def __init__(
            self,
            n_dims_sfea,  # number of dimensions in input/output scalar features
            n_dims_vfea,  # number of dimensions in input/output vector features
            n_dims_shid=None,  # number of dimensions in hidden scalar features (outputs from GVP)
            n_dims_vhid=None,  # number of dimensions in hidden vector features (outputs from GVP)
            n_heads=4,  # number of attention heads
            n_dims_attn=16,  # number of dimensions of query/key embeddings
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # basic configurations
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_vfea = n_dims_vfea
        self.n_dims_shid = n_dims_shid if n_dims_shid is not None else n_dims_sfea
        self.n_dims_vhid = n_dims_vhid if n_dims_vhid is not None else n_dims_vfea
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn

        # additional configurations
        self.eps = 1e-6

        # geometric vector perceptron
        self.gvp = GVP(self.n_dims_sfea, self.n_dims_vfea, self.n_dims_shid, self.n_dims_vhid)

        # attention embeddings
        self.linear_q = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn)
        self.linear_k = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn)
        self.linear_v = nn.Linear(self.n_dims_vfea * self.n_dims_vfea, self.n_heads)
        self.softmax_s2a = nn.Softmax(dim=1)
        self.softmax_v2a = nn.Softmax(dim=1)

        # feed-forward updates
        self.linear_os = nn.Linear(self.n_dims_shid, self.n_dims_sfea)
        self.linear_ov = nn.Linear(self.n_dims_vhid, self.n_dims_vfea, bias=False)


    def forward(self, sfea_mat_in, vfea_tns_in, vmsk_mat=None):  # pylint: disable=too-many-locals
        """Perform the forward pass.

        Args:
        * sfea_mat_in: input scalar features of size L x D_s
        * vfea_tns_in: input vector features of size L x D_v x 3
        * vmsk_mat: (optional) input vector features' validness masks of size L x D_v

        Returns:
        * sfea_mat_out: output scalar features of size L x D_s
        * vfea_tns_out: output vector features of size L x D_v x 3
        """

        # initialization
        device = sfea_mat_in.device
        n_nodes = sfea_mat_in.shape[0]

        # initialize validness masks, if not provided
        if vmsk_mat is None:
            vmsk_mat = torch.ones((n_nodes, self.n_dims_vfea), dtype=torch.int8, device=device)

        # geometric vector perceptron
        sfea_mat_hid, vfea_tns_hid = self.gvp(sfea_mat_in, vfea_tns_in, vmsk_mat)

        # sub-tract the mean 3D coordinate from hidden vector features
        vfea_vec_avg = torch.sum(vmsk_mat.unsqueeze(dim=2) * vfea_tns_in, dim=(0, 1)) \
            / (torch.sum(vmsk_mat) + self.eps)
        vfea_tns_hid = vfea_tns_hid - vfea_vec_avg.view(1, 1, 3)  # L x D_hv x 3

        # calculate attention weights from scalar & vector features
        a_tns_scl = self.__scalar_to_attn(sfea_mat_in)
        a_tns_vec = self.__vector_to_attn(vfea_tns_in, vmsk_mat)
        a_tns = (a_tns_scl + a_tns_vec) / 2.0

        # update scalar & vector features
        sfea_mat_upd = torch.sum(
            a_tns.view(n_nodes, n_nodes, self.n_heads, 1) *
            sfea_mat_hid.view(1, n_nodes, 1, self.n_dims_shid)
        , dim=(1, 2))  # L x D_hs
        vfea_tns_upd = torch.sum(
            a_tns.view(n_nodes, n_nodes, self.n_heads, 1, 1) *
            vfea_tns_hid.view(1, n_nodes, 1, self.n_dims_vhid, 3)
        , dim=(1, 2))  # L x D_hv x 3
        sfea_mat_out = sfea_mat_in + self.linear_os(sfea_mat_upd)
        vfea_tns_out = vfea_tns_in + self.linear_ov(vfea_tns_upd.transpose(1, 2)).transpose(1, 2)

        return sfea_mat_out, vfea_tns_out


    def __scalar_to_attn(self, sfea_mat):
        """Convert scalar features into attention weights."""

        # initialization
        n_nodes = sfea_mat.shape[0]

        # calculate query & key embeddings
        q_tns = self.linear_q(sfea_mat).view(n_nodes, 1, self.n_heads, self.n_dims_attn)
        k_tns = self.linear_k(sfea_mat).view(1, n_nodes, self.n_heads, self.n_dims_attn)

        # calculate attention weights
        a_tns = self.softmax_s2a(torch.sum(q_tns * k_tns, dim=-1) / np.sqrt(self.n_dims_attn))

        return a_tns


    def __vector_to_attn(self, vfea_tns, vmsk_mat):
        """Convert vector features into attention weights."""

        # initialization
        dtype = vfea_tns.dtype
        n_nodes, n_atoms, _ = vfea_tns.shape

        # calculate pairwise distance & validness masks
        dist_mat = cdist(vfea_tns.reshape(-1, 3).to(torch.float32)).to(dtype)
        dmsk_mat = vmsk_mat.view(-1, 1) * vmsk_mat.view(1, -1)
        dist_tns = torch.reshape(dist_mat.view(
            n_nodes, n_atoms, n_nodes, n_atoms).transpose(1, 2), [n_nodes, n_nodes, -1])
        dmsk_tns = torch.reshape(dmsk_mat.view(
            n_nodes, n_atoms, n_nodes, n_atoms).transpose(1, 2), [n_nodes, n_nodes, -1])

        # calculate attention weights
        a_tns = self.softmax_v2a(self.linear_v(dmsk_tns * dist_tns))

        return a_tns
