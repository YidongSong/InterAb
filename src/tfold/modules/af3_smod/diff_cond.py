"""The diffusion conditioning module. Algorithm 21 of AF3"""

import torch
from torch import nn
from math import pi

from einops import rearrange
from tfold.modules.af3_smod.utils import SwiGLU, log


class FourierEmbedding(nn.Module):
    """ Algorithm 22 """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = rearrange(times, 'b -> b 1')
        rand_proj = self.proj(times)

        return torch.cos(2*pi*rand_proj)


class Transition(nn.Module):
    """Transition module for both single & pair features."""

    def __init__(
        self,
        dim,
        expansion_factor=4,
        pre_norm=True,
    ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.dim = dim
        self.expansion_factor = expansion_factor
        self.pre_norm = pre_norm

        # build the module
        if self.pre_norm:
            self.layer_norm = nn.LayerNorm(self.dim)
        self.linear_1 = nn.Linear(self.dim, self.dim * self.expansion_factor * 2, bias=False)
        self.activation = SwiGLU()
        self.linear_2 = nn.Linear(self.dim * self.expansion_factor, self.dim, bias=False)

    def forward(self, xfea_tns):
        """Perform the forward pass.

        Args:
        * xfea_tns: single/pair features of size N x L x c / N x L x L x c

        Returns:
        * xfea_tns: updated single/pair features of size N x L x c / N x L x L x c
        """

        if self.pre_norm:
            xfea_tns = self.layer_norm(xfea_tns)
        xfea_tns = self.linear_1(xfea_tns)
        xfea_tns = self.activation(xfea_tns)
        xfea_tns = self.linear_2(xfea_tns)

        return xfea_tns


class SingleConditioning(nn.Module):
    """Algorithm 21 update sfea_tns"""

    def __init__(
        self,
        sigma_data,
        n_dims_sfea=384,
        n_dims_fourier=256,
        transition_expansion_factor=2,
        activation_checkpoint_fn=None,
    ):
        super().__init__()
        self.eps = 1e-10

        self.n_dims_sfea = n_dims_sfea
        self.sigma_data = sigma_data
        self.num_transitions = 2

        self.sfea_init_proj = nn.Sequential(
            nn.LayerNorm(2 * n_dims_sfea),
            nn.Linear(2 * n_dims_sfea, n_dims_sfea, bias=False)
        )

        self.fourier_embed = FourierEmbedding(n_dims_fourier)
        self.fourier_proj = nn.Sequential(
            nn.LayerNorm(n_dims_fourier),
            nn.Linear(n_dims_fourier, n_dims_sfea, bias=False)
        )

        transitions = nn.ModuleList()
        for _ in range(self.num_transitions):
            transition = Transition(dim=n_dims_sfea, expansion_factor=transition_expansion_factor, pre_norm=True)
            transitions.append(transition)

        self.transitions = transitions

        if activation_checkpoint_fn is None:
            self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.activation_checkpoint = False

    def enable_activation_checkpoint(self, enabled=True):
        """Enable the activation_checkpoint."""

        self.activation_checkpoint = enabled

    def forward(
        self,
        times,
        inpt,
        sfea_tns_trunk,
    ):

        sfea_tns = torch.cat((sfea_tns_trunk, inpt), dim=-1)
        assert sfea_tns.shape[-1] == 2 * self.n_dims_sfea
        sfea_tns = self.sfea_init_proj(sfea_tns)

        fourier_embed = self.fourier_embed(0.25 * log(times/self.sigma_data, eps=self.eps))
        fourier_tns = self.fourier_proj(fourier_embed)
        sfea_tns = sfea_tns + rearrange(fourier_tns, 'b d -> b 1 d')

        for transition in self.transitions:
            if self.activation_checkpoint:
                sfea_tns = sfea_tns + self.activation_checkpoint_fn(transition, sfea_tns)
            else:
                sfea_tns = sfea_tns + transition(sfea_tns)

        return sfea_tns


class PairwiseConditioning(nn.Module):
    """Algorithm 21 update pfea_tns"""

    def __init__(
        self,
        n_dims_pfea_trunk,
        n_dims_penc=128,
        n_dims_pfea=128,
        transition_expansion_factor=2,
        activation_checkpoint_fn=None,
    ):
        super().__init__()

        self.num_transitions = 2
        self.pfea_init_proj = nn.Sequential(
            nn.Linear(n_dims_pfea_trunk + n_dims_penc, n_dims_pfea, bias=False),
            nn.LayerNorm(n_dims_pfea)
        )

        transitions = nn.ModuleList()
        for _ in range(self.num_transitions):
            transition = Transition(dim=n_dims_pfea, expansion_factor=transition_expansion_factor, pre_norm=True)
            transitions.append(transition)

        self.transitions = transitions

        if activation_checkpoint_fn is None:
            self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.activation_checkpoint = False

    def enable_activation_checkpoint(self, enabled=True):
        """Enable the activation_checkpoint."""

        self.activation_checkpoint = enabled

    def forward(
        self,
        pfea_tns_trunk,
        penc_tns,
    ):

        pfea_tns = torch.cat((pfea_tns_trunk, penc_tns), dim=-1)
        pfea_tns = self.pfea_init_proj(pfea_tns)

        for transition in self.transitions:
            if self.activation_checkpoint:
                pfea_tns = self.activation_checkpoint_fn(transition, pfea_tns) + pfea_tns
            else:
                pfea_tns = transition(pfea_tns) + pfea_tns

        return pfea_tns
