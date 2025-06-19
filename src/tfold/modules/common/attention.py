# pylint: disable=invalid-name
"""Multi-head attention - original & memory efficient implementation.

Symbol notations for <Attention>:
* B: batch dimensions (all the remaining dimensions are packed here)
* N: attention dimension (also denoted as I/J)
* D_f: feature dimension
* H: number of attention heads
* D: embedding dimension per attention head
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import einops


class DConv_project(nn.Module):
    """Linear projection + 1-D depthwise convolution."""

    def __init__(self, d_model_in, d_model_out):
        """Constructor function."""

        super().__init__()

        self.linear = nn.Linear(d_model_in, d_model_out, bias=False)
        self.d_conv = nn.Conv1d(
            d_model_out, d_model_out, kernel_size=3, padding=1, groups=d_model_out)


    def forward(self, x):
        """Perform the forward pass.

        Args:
        * x: input tensor of size B x N x D_i

        Returns:
        * x: output tensor of size B x N x D_o
        """

        x = self.linear(x)  # B x N x D_o
        x = x.permute(0, 2, 1).contiguous()  # B x D_o x N
        x = self.d_conv(x)  # B x D_o x N (1-D depthwise conv. is applied along the <N> dimension)
        x = x.permute(0, 2, 1).contiguous()  # B x N x D_o

        return x


class Attention(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-head attention - original & memory efficient implementation."""

    def __init__(
            self,
            n_dims,
            n_heads=8,
            n_dim_heads=64,
            is_gating=True,
            is_head_scale=False,
            p_attn_drop=0.0,
            is_stable_softmax=True,
            rpe_type=None,
            proj_type='DConv_project',
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # setup hyper-parameters
        inner_dim = n_dim_heads * n_heads
        self.heads = n_heads
        self.is_gating = is_gating
        self.is_head_scale = is_head_scale
        self.is_stable_softmax = is_stable_softmax
        self.rpe_type = rpe_type
        self.proj_type = proj_type

        # validate hyper-parameters
        assert self.rpe_type is None, f'unsupported <rpe_type>: {self.rpe_type}'
        assert self.proj_type == 'DConv_project', f'unsupported <proj_type>: {self.proj_type}'

        # build sub-networks
        self.scale = n_dim_heads ** -0.5
        if self.is_head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, n_heads, 1, 1))
        if self.is_gating:
            self.gating = nn.Linear(n_dims, inner_dim)
            nn.init.constant_(self.gating.weight, 0.0)
            nn.init.constant_(self.gating.bias, 1.0)
        self.to_q = DConv_project(n_dims, inner_dim)
        self.to_k = DConv_project(n_dims, inner_dim)
        self.to_v = DConv_project(n_dims, inner_dim)
        self.to_out = nn.Linear(inner_dim, n_dims)
        self.attn_drop = nn.Dropout(p_attn_drop)


    def forward(self, x, mask=None, attn_bias=None, tie_dim=None):
        """Perform the forward pass.

        Args:
        * x: input tensor of size B x N x D_f
        * mask: input tensors' validness masks of size B x N
        * attn_bias: attention weights' biases of size 1 x H x I x J
        * tie_dim: length of dimension for query embedding averaging (as in ExtraMsaStack)

        Returns:
        * out: output tensor of size B x N x D_f

        Notes:
        * <mask> must be None. It is reserved here only for compatibility.
        * <B> must be divisible by <tie_dim>, if provided.
        """

        # initialization
        h = self.heads
        assert mask is None, '<mask> must be None. It is reserved here only for compatibility.'

        # obtain query, key, and value embeddings
        q = self.scale * self.to_q(x)  # B x N x (H x D)
        k = self.to_k(x)  # B x N x (H x D)
        v = self.to_v(x)  # B x N x (H x D)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  # B x H x N x D

        # calculate the averaged query embedding
        if tie_dim:
            q = torch.mean(einops.rearrange(q, '(b r) ... -> b r ...', r=tie_dim), dim=1)

        # calculate inner products of query & key embeddings
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)  # B x H x N x N
        if attn_bias is not None:
            dots = dots + attn_bias  # B x H x N x N
        if self.is_stable_softmax:
            dots = dots - dots.max(dim=-1, keepdims=True).values  # B x H x N x N

        # calculate attention weights
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)  # B x H x N x N

        # aggregate value embeddings w/ attention weights
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # B x H x N x D

        # additional scaling
        if self.is_head_scale:
            out = out * self.head_scale_params
        out = einops.rearrange(out, 'b h n d -> b n (h d)')  # B x N x (H x D)

        # apply the gating mechanism
        if self.is_gating:
            gates = self.gating(x)
            out = out * gates.sigmoid()  # B x N x (H x D)

        # apply the final output projection
        out = self.to_out(out)  # B x N x D_f

        return out


    def forward_me(self, x, mask=None, attn_bias=None, tie_dim=None):  # pylint: disable=too-many-locals
        """Perform the memory-efficient forward pass."""

        # configurations
        split_size = 32

        # initialization
        h = self.heads
        assert mask is None, '<mask> must be None. It is reserved here only for compatibility.'

        # obtain query, key, and value embeddings
        q = self.scale * self.to_q(x)  # B x N x (H x D)
        k = self.to_k(x)  # B x N x (H x D)
        v = self.to_v(x)  # B x N x (H x D)
        q, k, v = map(
            lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))  # B x H x N x D

        # calculate the averaged query embedding
        if tie_dim:
            q = torch.mean(einops.rearrange(q, '(b r) ... -> b r ...', r=tie_dim), dim=1)

        # split key and value embeddings into chunks
        k_list = torch.split(k, split_size, dim=2)
        v_list = torch.split(v, split_size, dim=2)
        b_list = [None] * len(k_list) \
            if attn_bias is None else torch.split(attn_bias, split_size, dim=3)
        o_tns_list = []
        e_sum_list = []
        s_max_list = []
        for k_chk, v_chk, b_chk in zip(k_list, v_list, b_list):
            if not self.training:
                o_tns, e_sum, s_max = self.__summarize_chunk(q, k_chk, v_chk, b_chk)
            else:
                o_tns, e_sum, s_max = checkpoint(self.__summarize_chunk, q, k_chk, v_chk, b_chk)
            o_tns_list.append(o_tns)  # B x H x N x V
            e_sum_list.append(e_sum)  # B x H x N x 1
            s_max_list.append(s_max)  # B x H x N x 1
        o_tns = torch.stack(o_tns_list, dim=3)  # B x H x N x K x V
        e_sum = torch.cat(e_sum_list, dim=3)  # B x H x N x K
        s_max = torch.cat(s_max_list, dim=3)  # B x H x N x K

        s_max_gbl = torch.max(s_max, dim=3, keepdim=True)[0]  # B x H x N x 1
        s_max_dff = torch.exp(s_max - s_max_gbl)  # B x H x N x K
        o_tns = o_tns * s_max_dff.unsqueeze(dim=-1)  # B x H x N x K x V
        e_sum = e_sum * s_max_dff  # B x H x N x K
        out = torch.sum(o_tns, dim=3) / torch.sum(e_sum, dim=3, keepdim=True)

        # additional scaling
        if self.is_head_scale:
            out = out * self.head_scale_params
        out = einops.rearrange(out, 'b h n d -> b n (h d)')  # B x N x (H x D)

        # apply the gating mechanism
        if self.is_gating:
            gates = self.gating(x)
            out = out * gates.sigmoid()  # B x N x (H x D)

        # apply the final output projection
        out = self.to_out(out)  # B x N x D_f

        return out


    def __summarize_chunk(self, q_tns, k_tns, v_tns, b_tns=None):  # pylint: disable=too-many-locals
        """Summarize the current Q/K/V/B chunk.

        Notes:
        * q_tns: 1/B x H x N x D
        * k_tns: B x H x M x D
        * v_tns: B x H x M x V
        * b_tns: 1 x H x N x M
        """

        s_tns = torch.einsum('b h n d, b h m d -> b h n m', q_tns, k_tns)  # B x H x N x M
        if b_tns is not None:
            s_tns = s_tns + b_tns  # B x H x N x M
        s_max = torch.max(s_tns, dim=-1, keepdims=True)[0].detach()  # B x H x N x 1
        e_tns = torch.exp(s_tns - s_max)  # B x H x N x M
        e_sum = torch.sum(e_tns, dim=3, keepdim=True)  # B x H x N x 1
        o_tns = torch.einsum('b h n m, b h m v -> b h n v', e_tns, v_tns)  # B x H x N x V

        return o_tns, e_sum, s_max
