"""The diffusion transformer module. Algorithm 23 of AF3"""

import torch
from torch import nn

from tfold.modules.af3_smod.attention import AttentionPairBias
from tfold.modules.af3_smod.utils import AdaptiveLayerNorm
from tfold.modules.af3_smod.utils import swish


class ConditionTransition(nn.Module):
    """ Algorithm 25 """
    def __init__(
        self,
        dim,
        dim_cond,
        expansion_factor=2
    ):
        super().__init__()

        self.adaptive_norm = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.linear_1 = nn.Linear(dim, dim * expansion_factor, bias=False)
        self.linear_2 = nn.Linear(dim, dim * expansion_factor, bias=False)
        self.linear_3 = nn.Linear(dim * expansion_factor, dim, bias=False)

        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)

        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, -2)

        self.to_adaln_zero_gamma = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    def forward(
        self,
        a,
        cond,
        **kwargs
    ):
        a = self.adaptive_norm(a, cond=cond)
        b = swish(self.linear_1(a)) * self.linear_2(a)
        a = self.to_adaln_zero_gamma(cond) * self.linear_3(b)

        return a


class DiffusionTransformerBlock(nn.Module):
    """The DiffusionTransformer block."""

    def __init__(
        self,
        n_heads,
        dim=384,
        n_dims_cond=None,
        n_dims_pfea=128,
        attn_window_size=None,
        attn_pair_bias_kwargs: dict = dict(),
    ):
        super().__init__()

        # AttentionPairBias
        self.pair_bias_attn = AttentionPairBias(
            dim=dim,
            n_dims_sfea=n_dims_cond,
            n_dims_pfea=n_dims_pfea,
            n_heads=n_heads,
            window_size=attn_window_size,
            **attn_pair_bias_kwargs
        )

        # ConditionTransition
        self.condition_trans = ConditionTransition(
            dim=dim,
            dim_cond=n_dims_cond,
        )

    def forward(self, nfea_tns, sfea_tns, pfea_tns):
        attn_out = self.pair_bias_attn(nfea_tns, sfea_tns=sfea_tns, pfea_tns=pfea_tns)
        ff_out = self.condition_trans(nfea_tns, cond=sfea_tns)
        nfea_tns = attn_out + ff_out + nfea_tns

        return nfea_tns


class DiffusionTransformer(nn.Module):
    """ Algorithm 23 """

    def __init__(
        self,
        n_lyrs,
        n_heads,
        dim=384,
        n_dims_cond=None,
        n_dims_pfea=128,
        attn_window_size=None,
        attn_pair_bias_kwargs: dict = dict(),
        activation_checkpoint_fn=None
    ):
        super().__init__()

        n_dims_cond = n_dims_cond if n_dims_cond is not None else dim
        self.n_lyrs = n_lyrs

        if activation_checkpoint_fn is None:
            self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.activation_checkpoint = False

        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(
                n_heads,
                dim,
                n_dims_cond,
                n_dims_pfea,
                attn_window_size,
                attn_pair_bias_kwargs,
            )
            for _ in range(self.n_lyrs)
        ])

    def enable_activation_checkpoint(self, enabled=True):
        """Enable the activation_checkpoint."""

        self.activation_checkpoint = enabled

    def forward(
        self,
        nfea_tns,
        sfea_tns,
        pfea_tns,
    ):

        # diffusion transformer stack
        for block in self.blocks:
            if self.activation_checkpoint:
                nfea_tns = self.activation_checkpoint_fn(block, nfea_tns, sfea_tns, pfea_tns)
            else:
                nfea_tns = block(nfea_tns, sfea_tns, pfea_tns)
        return nfea_tns
