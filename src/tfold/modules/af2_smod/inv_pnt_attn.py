"""The invariant point attention (IPA) module."""

import numpy as np
import torch
from torch import nn

from tfold.utils import quat2rot
from tfold.utils import apply_trans


class InvPntAttn(nn.Module):  # pylint: disable=too-many-instance-attributes
    """The invariant point attention (IPA) module."""

    def __init__(
            self,
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_attn=16,   # number of dimensions in query/key/value embeddings
            n_heads=12,       # number of attention heads
            n_qpnts=4,        # number of points for query embeddings
            n_vpnts=8,        # number of points for value embeddings
            drop_prob=0.1,    # probability of an element to be zeroed (set to zero for DEQ models)
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # setup hyper-parameters
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_attn = n_dims_attn
        self.n_heads = n_heads
        self.n_qpnts = n_qpnts
        self.n_vpnts = n_vpnts
        self.drop_prob = drop_prob

        # setup additional configurations
        self.n_dims_cord = 3  # DO NOT MODIFY!
        self.n_dims_shid = self.n_heads * \
            (self.n_dims_pfea + self.n_dims_attn + self.n_vpnts * 3 + self.n_vpnts)
        self.wc = np.sqrt(2.0 / (9.0 * self.n_qpnts))  # pylint: disable=invalid-name
        self.wl = np.sqrt(1.0 / 3.0)  # pylint: disable=invalid-name
        self.ws = np.log(np.exp(1.0) - 1.0)  # pylint: disable=invalid-name

        # sub-networks - Invariant Point Attention
        self.linear_q = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_k = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_v = nn.Linear(self.n_dims_sfea, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_qp = nn.Linear(
            self.n_dims_sfea, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
        self.linear_kp = nn.Linear(
            self.n_dims_sfea, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
        self.linear_vp = nn.Linear(
            self.n_dims_sfea, self.n_heads * self.n_vpnts * self.n_dims_cord, bias=False)
        self.linear_b = nn.Linear(self.n_dims_pfea, self.n_heads, bias=False)
        self.linear_s = nn.Linear(self.n_dims_shid, self.n_dims_sfea)
        self.register_parameter(
            name='scale', param=nn.Parameter(self.ws * torch.ones((self.n_heads))))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=2)

        # sub-networks - Feed-Forward Network
        self.drop_1 = nn.Dropout(p=self.drop_prob)
        self.norm_1 = nn.LayerNorm(self.n_dims_sfea)
        self.mlp = nn.Sequential(
            nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
            nn.ReLU(),
            nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
            nn.ReLU(),
            nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
        )
        self.drop_2 = nn.Dropout(p=self.drop_prob)
        self.norm_2 = nn.LayerNorm(self.n_dims_sfea)


    def forward(self, sfea_tns, pfea_tns, quat_tns, trsl_tns):  # pylint: disable=too-many-locals,too-many-statements
        """Perform the forward pass.

        Args:
        * sfea_tns: single features of size N x L x D_s
        * pfea_tns: pair features of size N x L x L x D_p
        * quat_tns: quaternion vectors of size N x L x 4
        * trsl_tns: translation vectors of size N x L x 3

        Returns:
        * sfea_tns: single features of size N x L x D_s
        """

        # initialization
        n_smpls, n_resds, _ = sfea_tns.shape
        assert n_smpls == 1, f'batch size must be 1 in <InvPntAttn>; {n_smpls} detected'

        # calculate query/key/value embeddings
        q_tns = self.linear_q(sfea_tns).view(n_smpls, n_resds, 1, self.n_heads, self.n_dims_attn)
        k_tns = self.linear_k(sfea_tns).view(n_smpls, 1, n_resds, self.n_heads, self.n_dims_attn)
        v_tns = self.linear_v(sfea_tns).view(n_smpls, n_resds, self.n_heads, self.n_dims_attn)
        qp_tns = self.linear_qp(sfea_tns).view(
            n_smpls, n_resds, self.n_heads, self.n_qpnts, self.n_dims_cord)
        kp_tns = self.linear_kp(sfea_tns).view(
            n_smpls, n_resds, self.n_heads, self.n_qpnts, self.n_dims_cord)
        vp_tns = self.linear_vp(sfea_tns).view(
            n_smpls, n_resds, self.n_heads, self.n_vpnts, self.n_dims_cord)
        b_tns = self.linear_b(pfea_tns).view(n_smpls, n_resds, n_resds, self.n_heads)

        # apply global transformation on Q/K/V points
        rota_tns = quat2rot(quat_tns[0]).unsqueeze(dim=0)
        qp_tns_proj = apply_trans(qp_tns, rota_tns, trsl_tns, grouped=True).view(
            n_smpls, n_resds, 1, self.n_heads, self.n_qpnts, 3)
        kp_tns_proj = apply_trans(kp_tns, rota_tns, trsl_tns, grouped=True).view(
            n_smpls, 1, n_resds, self.n_heads, self.n_qpnts, 3)
        vp_tns_proj = apply_trans(vp_tns, rota_tns, trsl_tns, grouped=True).view(
            n_smpls, n_resds, self.n_heads, self.n_vpnts, 3)

        # calculate the distance between query/key points
        dist_tns = torch.norm(qp_tns_proj - kp_tns_proj, dim=-1)  # N x L x L x H x P_q

        # compute attention weights
        qk_tns = torch.sum(q_tns * k_tns, dim=-1) / np.sqrt(self.n_dims_attn)  # N x L x L x H
        qkp_tns = 0.5 * self.wc * \
            self.softplus(self.scale).view(1, 1, 1, -1) * torch.sum(dist_tns.square(), dim=-1)
        a_tns = self.softmax(self.wl * (qk_tns + b_tns - qkp_tns))  # N x L x L x H

        # update single features
        op_tns = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            pfea_tns.view(n_smpls, n_resds, n_resds, 1, self.n_dims_pfea)
        , dim=2)  # N x L x H x D_p
        ov_tns = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            v_tns.view(n_smpls, 1, n_resds, self.n_heads, self.n_dims_attn)
        , dim=2)  # N x L x H x D_a
        ovp_tns_proj = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            vp_tns_proj.view(n_smpls, 1, n_resds, self.n_heads, self.n_vpnts * 3)
        , dim=2)  # N x L x H x (P_v x 3)
        ovp_tns = apply_trans(
            ovp_tns_proj, rota_tns, trsl_tns, grouped=True, reverse=True,
        ).view(n_smpls, n_resds, self.n_heads, self.n_vpnts * 3)  # N x L x H x (P_v x 3)
        ovp_tns_norm = torch.norm(
            ovp_tns.view(n_smpls, n_resds, self.n_heads, self.n_vpnts, 3)
        , dim=4)  # N x L x H x P_v
        shid_tns = torch.cat([op_tns, ov_tns, ovp_tns, ovp_tns_norm], dim=3)  # N x L x (H x D_h')
        sfea_tns = sfea_tns + self.linear_s(shid_tns.view(n_smpls, n_resds, self.n_dims_shid))

        # pass single features through a feed-forward network
        sfea_tns = self.norm_1(self.drop_1(sfea_tns))
        sfea_tns = sfea_tns + self.mlp(sfea_tns)
        sfea_tns = self.norm_2(self.drop_2(sfea_tns))

        return sfea_tns
