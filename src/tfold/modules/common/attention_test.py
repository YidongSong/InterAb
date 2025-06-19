# pylint: disable=invalid-name
"""Unit-tests for <Attention>."""

import logging
from timeit import default_timer as timer

import torch
from torch import nn
import einops
from einops.layers.torch import Rearrange

from tfold.utils import tfold_init
from tfold.modules.common.attention import Attention as AttentionTst
from tfold.modules.xfold.network.attention import Attention as AttentionRef


def get_config(n_dims, n_heads):
    """Get configurations for the attention module."""

    config = {
        'n_dims': n_dims,
        'n_heads': n_heads,
        'n_dim_heads': n_dims // n_heads,
        'is_gating': True,
        'is_head_scale': False,
        'p_attn_drop': 0.0,
        'is_stable_softmax': True,
        'rpe_type': None,
        'proj_type': 'DConv_project',
    }

    return config


def build_attn_modules(config, device):
    """Build two attention modules (reference & under-test)."""

    # build two attention modules (reference & under-test)
    attn_tst = AttentionTst(**config).to(device)
    attn_ref = AttentionRef(**config).to(device)

    # synchronize model parameters between attention modules
    for param_tst, param_ref in zip(attn_tst.parameters(), attn_ref.parameters()):
        param_ref.data = param_tst.data.detach().clone()

    return attn_tst, attn_ref


def run_benchmark(pfea_tns):
    """Run benchmark tests w/ TriangleAttentionStartingNode."""

    def _get_max_memory():
        return torch.cuda.max_memory_allocated() / 1024.0 ** 2

    # configurations
    n_repts = 16
    device = pfea_tns.device
    n_dims_pfea = pfea_tns.shape[-1]
    n_heads = 8
    config = get_config(n_dims_pfea, n_heads)

    # build the attention module
    attn = AttentionTst(**config).to(device)

    # build additional mapping for attention biases
    pfea_to_attn_bias = nn.Sequential(
        nn.Linear(n_dims_pfea, n_heads, bias=False),
        Rearrange('b i j h -> b h i j'),
    ).to(device)

    # prepare inputs for attention modules
    x = einops.rearrange(pfea_tns, 'b h w d -> (b h) w d')
    attn_bias = pfea_to_attn_bias.forward(pfea_tns)

    # measure the GPU memory consumption
    logging.info('[pre-forward] %.2f MB', _get_max_memory())
    z = attn.forward(x, attn_bias=attn_bias)
    logging.info('[post-forward] %.2f MB', _get_max_memory())
    loss = torch.mean(z - x)
    loss.backward()
    logging.info('[post-backward] %.2f MB', _get_max_memory())

    # measure the run-time speed
    for _ in range(n_repts):  # warm-up
        z = attn.forward(x, attn_bias=attn_bias)
    time_beg = timer()
    for _ in range(n_repts):  # warm-up
        z = attn.forward(x, attn_bias=attn_bias)
    time_avg = (timer() - time_beg) / n_repts
    logging.info('elapsed time: %.2f (ms)', 1000.0 * time_avg)


def test_msa_row_attn(mfea_tns, pfea_tns):  # pylint: disable=too-many-locals
    """Test row-wise attention on MSA features (MSARowAttentionWithPairBias)."""

    # configurations
    device = mfea_tns.device
    n_dims_mfea = mfea_tns.shape[-1]
    n_dims_pfea = pfea_tns.shape[-1]
    n_heads = 12
    config = get_config(n_dims_mfea, n_heads)

    # build two attention modules (reference & under-test)
    attn_tst, attn_ref = build_attn_modules(config, device)

    # build additional mapping for attention biases
    pfea_to_attn_bias = nn.Sequential(
        nn.Linear(n_dims_pfea, n_heads, bias=False),
        Rearrange('b i j h -> b h i j'),
    ).to(device)

    # prepare inputs for attention modules
    h = mfea_tns.shape[1]
    x = einops.rearrange(mfea_tns, 'b h w d -> (b h) w d')
    attn_bias_tst = pfea_to_attn_bias.forward(pfea_tns)
    attn_bias_ref = einops.repeat(attn_bias_tst, 'b h i j -> (b x) h i j', x=h)

    # perform the forward pass w/ each attention module
    x_out_tst = attn_tst.forward(x, attn_bias=attn_bias_tst)
    x_out_ref = attn_ref.forward(x, attn_bias=attn_bias_ref)

    # compare outputs from two attention modules
    x_err = torch.norm(x_out_tst - x_out_ref).item()
    logging.info('[MSARowAttentionWithPairBias] x_err: %.4f', x_err)


def test_msa_col_attn(mfea_tns, is_global):
    """Test column-wise attention on MSA features (MSAColumnAttention/MSAColumnGlobalAttention)."""

    # configurations
    device = mfea_tns.device
    n_dims_mfea = mfea_tns.shape[-1]
    n_heads = 12
    config = get_config(n_dims_mfea, n_heads)

    # build two attention modules (reference & under-test)
    attn_tst, attn_ref = build_attn_modules(config, device)

    # prepare inputs for attention modules
    w = mfea_tns.shape[2]
    x = einops.rearrange(mfea_tns, 'b h w d -> (b w) h d')
    tie_dim = w if is_global else None

    # perform the forward pass w/ each attention module
    x_out_tst = attn_tst.forward(x, tie_dim=tie_dim)
    x_out_ref = attn_ref.forward(x, tie_dim=tie_dim)

    # compare outputs from two attention modules
    x_err = torch.norm(x_out_tst - x_out_ref).item()
    name = 'MSAColumnAttention' if not is_global else 'MSAColumnGlobalAttention'
    logging.info('[%s] x_err: %.4f', name, x_err)


def test_pair_row_attn(pfea_tns):
    """Test row-wise attention on pair features (TriangleAttentionStartingNode)."""

    # configurations
    device = pfea_tns.device
    n_dims_pfea = pfea_tns.shape[-1]
    n_heads = 8
    config = get_config(n_dims_pfea, n_heads)

    # build two attention modules (reference & under-test)
    attn_tst, attn_ref = build_attn_modules(config, device)

    # build additional mapping for attention biases
    pfea_to_attn_bias = nn.Sequential(
        nn.Linear(n_dims_pfea, n_heads, bias=False),
        Rearrange('b i j h -> b h i j'),
    ).to(device)

    # prepare inputs for attention modules
    h = pfea_tns.shape[1]
    x = einops.rearrange(pfea_tns, 'b h w d -> (b h) w d')
    attn_bias_tst = pfea_to_attn_bias.forward(pfea_tns)
    attn_bias_ref = einops.repeat(attn_bias_tst, 'b h i j -> (b x) h i j', x=h)

    # perform the forward pass w/ each attention module
    x_out_tst = attn_tst.forward(x, attn_bias=attn_bias_tst)
    x_out_ref = attn_ref.forward(x, attn_bias=attn_bias_ref)

    # compare outputs from two attention modules
    x_err = torch.norm(x_out_tst - x_out_ref).item()
    logging.info('[TriangleAttentionStartingNode] x_err: %.4f', x_err)


def test_pair_col_attn(pfea_tns):
    """Test column-wise attention on pair features (TriangleAttentionEndingNode)."""

    # configurations
    device = pfea_tns.device
    n_dims_pfea = pfea_tns.shape[-1]
    n_heads = 8
    config = get_config(n_dims_pfea, n_heads)

    # build two attention modules (reference & under-test)
    attn_tst, attn_ref = build_attn_modules(config, device)

    # build additional mapping for attention biases
    pfea_to_attn_bias = nn.Sequential(
        nn.Linear(n_dims_pfea, n_heads, bias=False),
        Rearrange('b i j h -> b h i j'),
    ).to(device)

    # prepare inputs for attention modules
    w = pfea_tns.shape[2]
    x = einops.rearrange(pfea_tns, 'b h w d -> (b w) h d')
    attn_bias_tst = pfea_to_attn_bias.forward(pfea_tns)
    attn_bias_ref = einops.repeat(attn_bias_tst, 'b h i j -> (b x) h i j', x=w)

    # perform the forward pass w/ each attention module
    x_out_tst = attn_tst.forward(x, attn_bias=attn_bias_tst)
    x_out_ref = attn_ref.forward(x, attn_bias=attn_bias_ref)

    # compare outputs from two attention modules
    x_err = torch.norm(x_out_tst - x_out_ref).item()
    logging.info('[TriangleAttentionEndingNode] x_err: %.4f', x_err)


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_algns = 128
    n_resds = 128
    n_dims_mfea = 384
    n_dims_pfea = 256
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # randomly initialize MSA & pair features
    mfea_tns = torch.randn((n_smpls, n_algns, n_resds, n_dims_mfea), device=device)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, n_dims_pfea), device=device)

    # run benchmark tests w/ TriangleAttentionStartingNode
    run_benchmark(pfea_tns)

    # test case 1: row-wise attention on MSA features (MSARowAttentionWithPairBias)
    test_msa_row_attn(mfea_tns, pfea_tns)

    # test case 2: column-wise attention on MSA features (MSAColumnAttention)
    test_msa_col_attn(mfea_tns, is_global=False)

    # test case 3: column-wise attention on MSA features (MSAColumnGlobalAttention)
    test_msa_col_attn(mfea_tns, is_global=True)

    # test case 4: row-wise attention on pair features (TriangleAttentionStartingNode)
    test_pair_row_attn(pfea_tns)

    # test case 5: column-wise attention on pair features (TriangleAttentionEndingNode)
    test_pair_col_attn(pfea_tns)


if __name__ == '__main__':
    main()
