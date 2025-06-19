from torch import nn

from einops.layers.torch import Rearrange
from einops import rearrange
from tfold.modules.attention import WindowAttention
from tfold.modules.af3_smod.utils import AdaptiveLayerNorm


class AttentionPairBias(nn.Module):
    """ Algorithm 24  DiffusionAttention with pair bias and mask"""
    def __init__(
        self,
        dim,
        n_heads,
        n_dims_sfea,
        n_dims_pfea,
        window_size=None,
        num_memory_kv=0,
        **attn_kwargs
    ):
        super().__init__()

        self.window_size = window_size

        self.attn = WindowAttention(
            c_q=dim,
            c_k=dim,
            c_v=dim,
            window_size=window_size,
            num_heads=n_heads,
            **attn_kwargs
        )
        self.adaptive_norm = AdaptiveLayerNorm(dim=dim, dim_cond=n_dims_sfea)

        # line 8 of Algorithm 24
        to_attn_bias_linear = nn.Linear(n_dims_pfea, n_heads, bias=False)
        nn.init.zeros_(to_attn_bias_linear.weight)
        self.to_attn_bias = nn.Sequential(
            nn.LayerNorm(n_dims_pfea),
            to_attn_bias_linear,
            Rearrange('b ... h -> b h ...')
        )

        # line 13 of Algorithm 24
        adaln_zero_gamma_linear = nn.Linear(n_dims_sfea, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, -2)
        self.to_out = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    def forward(
        self,
        nfea_tns,
        sfea_tns,
        pfea_tns,
        attn_bias=None,
        **kwargs
    ):

        # attention bias preparation with further addition from pairwise repr
        if attn_bias is not None:
            attn_bias = rearrange(attn_bias, 'b ... -> b 1 ...')
        else:
            attn_bias = 0.
        # line 2
        nfea_tns = self.adaptive_norm(nfea_tns, sfea_tns)

        # prepare attention bias
        attn_bias = self.to_attn_bias(pfea_tns) + attn_bias
        # Attention
        nfea_tns = self.attn(nfea_tns, attn_bias=attn_bias, **kwargs)

        # line 13
        nfea_tns = self.to_out(sfea_tns) * nfea_tns

        return nfea_tns
