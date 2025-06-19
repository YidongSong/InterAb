"""Geometric vector perceptron.

Reference:
* Jing et al., Learning from Protein Structure with Geometric Vector Perceptrons. ICLR 2021.
Official implementation (TensorFlow):
* https://github.com/drorlab/gvp
"""

import torch
from torch import nn


class GVP(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Geometric vector perceptron."""

    def __init__(
            self,
            n_dims_is,  # number of dimensions in input scalar features
            n_dims_iv,  # number of dimensions in input vector features
            n_dims_os,  # number of dimensions in output scalar features
            n_dims_ov,  # number of dimensions in output vector features
            n_dims_hv=None,  # number of dimensions in hidden vector features
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # basic configurations
        self.n_dims_is = n_dims_is
        self.n_dims_iv = n_dims_iv
        self.n_dims_os = n_dims_os
        self.n_dims_ov = n_dims_ov
        self.n_dims_hv = n_dims_hv if n_dims_hv is not None else n_dims_iv
        self.n_dims_hs = self.n_dims_is + self.n_dims_hv

        # additional configurations
        self.eps = 1e-6

        # linear layers & non-linear activations
        self.linear_h = nn.Linear(self.n_dims_iv, self.n_dims_hv, bias=False)
        self.linear_s = nn.Linear(self.n_dims_hs, self.n_dims_os)
        self.linear_v = nn.Linear(self.n_dims_hv, self.n_dims_ov, bias=False)
        self.actv_s = nn.Sigmoid()
        self.actv_v = nn.ReLU()


    def forward(self, sfea_mat_in, vfea_tns_in, vmsk_mat=None):
        """Perform the forward pass.

        Args:
        * sfea_mat_in: input scalar features of size L x D_is
        * vfea_tns_in: input vector features of size L x D_iv x 3
        * vmsk_mat: (optional) input vector features' validness masks of size L x D_iv

        Returns:
        * sfea_mat_out: output scalar features of size L x D_os
        * vfea_tns_out: output vector features of size L x D_ov x 3

        Note:
        * For residue-level inputs, both <D_iv> and <D_ov> equal to the number of atoms per residue.
        * For atom-level inputs, both <D_iv> and <D_ov> equal to 1 (one 3D coordinate per atom).
        * We sub-tract the mean 3D coordinate from input vector features, and add it back to output
            vector features, so as to preserve the translational equivariance.
        """

        # initialization
        device = sfea_mat_in.device
        n_nodes = sfea_mat_in.shape[0]

        # initialize validness masks, if not provided
        if vmsk_mat is None:
            vmsk_mat = torch.ones((n_nodes, self.n_dims_iv), dtype=torch.int8, device=device)

        # swap the last two dimensions of input vector features
        vfea_tns_in = torch.transpose(vfea_tns_in, 1, 2)  # L x 3 x D_iv

        # sub-tract the mean 3D coordinate from input vector features
        vfea_vec_avg = torch.sum(vmsk_mat.unsqueeze(dim=1) * vfea_tns_in, dim=(0, 2)) \
            / (torch.sum(vmsk_mat) + self.eps)
        vfea_tns_inc = vfea_tns_in - vfea_vec_avg.view(1, 3, 1)  # L x 3 x D_iv

        # apply linear mappings on vector features
        vfea_tns_h1 = self.linear_h(vmsk_mat.unsqueeze(dim=1) * vfea_tns_inc)  # L x 3 x D_hv
        vfea_tns_h2 = self.linear_v(vfea_tns_h1)  # L x 3 x D_ov
        vnrm_mat_h1 = torch.norm(vfea_tns_h1, dim=1)  # L x D_hv
        vnrm_tns_h2 = torch.norm(vfea_tns_h2, dim=1, keepdim=True)  # L x 1 x D_ov

        # calculcate output scalar & vector features
        sfea_mat_hid = torch.cat([sfea_mat_in, vnrm_mat_h1], dim=1)  # L x D_hs
        sfea_mat_out = self.actv_s(self.linear_s(sfea_mat_hid))  # L x D_os
        vfea_tns_out = self.actv_v(vnrm_tns_h2) * vfea_tns_h2  # L x 3 x D_ov

        # add the mean 3D coordinate back to output vector features
        vfea_tns_out = vfea_tns_out + vfea_vec_avg.view(1, 3, 1)  # L x 3 x D_ov

        # swap the last two dimensions of output vector features
        vfea_tns_out = torch.transpose(vfea_tns_out, 1, 2)  # L x D_ov x 3

        return sfea_mat_out, vfea_tns_out
