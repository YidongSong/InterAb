"""Mathematics-related utility functions."""
# pylint: disable=invalid-name

import numpy as np
import torch
from torch import nn


def cdist(x1, x2=None):
    """Calculate the pairwise distance matrix.

    Args:
    * x1: input tensor of size N x D or B x N x D
    * x2: (optional) input tensor of size M x D or B x M x D

    Returns:
    * dist_tns: pairwise distance of size N x M or B x N x M

    Note:
    * If <x2> is not provided, then pairwise distance will be computed within <x1>.
    * The matrix multiplication approach will not be used to avoid the numerical stability issue.
    """

    # initialization
    x2 = x1 if x2 is None else x2

    # recursively call if the batch dimension is missing
    if (x1.ndim == 2) and (x2.ndim == 2):
        return cdist(x1.unsqueeze(dim=0), x2.unsqueeze(dim=0))[0]

    # validate inputs
    assert (x1.ndim == 3) and (x2.ndim == 3)
    assert (x1.shape[0] == x2.shape[0]) and (x1.shape[2] == x2.shape[2])

    # calculate the pairwise distance matrix
    with torch.cuda.amp.autocast():
        dist_tns = torch.cdist(x1, x2, compute_mode='donot_use_mm_for_euclid_dist')

    return dist_tns


def cvt_to_one_hot(tensor, depth):
    """Convert an integer array into one-hot encodings.

    Args:
    * tensor: integer array of size D1 x D2 x ... x Dk
    * depth: one-hot encodings's depth - C

    Returns:
    * onht_tns: one-hot encodings of size D1 x D2 x ... x Dk x C
    """

    if isinstance(tensor, np.ndarray):
        assert np.min(tensor) >= 0 and np.max(tensor) < depth
        onht_tns = np.reshape(
            np.eye(depth)[tensor.ravel()], list(tensor.shape) + [depth]).astype(np.float32)
    elif isinstance(tensor, torch.Tensor):
        onht_tns = nn.functional.one_hot(tensor, depth)
    else:
        raise TypeError(f'invalid tensor type: {type(tensor)}')

    return onht_tns


def split_by_head(tensor, n_heads):
    """Split the k-dimensional tensor by number of heads.

    Args:
    * tensor: input tensor of size D1 x D2 x ... x Dk
    * n_heads: number of heads - H

    Returns:
    * mhead_tns: multi-head tensor of size D1 x D2 x ... x H x Dk' (where Dk = H * Dk')
    """

    assert tensor.shape[-1] % n_heads == 0, \
        f'the last dimension ({tensor.shape[-1]}) is not divisiable by # of heads ({n_heads})'

    mhead_tns_shape = list(tensor.shape)[:-1] + [n_heads, tensor.shape[-1] // n_heads]
    if isinstance(tensor, np.ndarray):
        mhead_tns = np.reshape(tensor, mhead_tns_shape)
    elif isinstance(tensor, torch.Tensor):
        mhead_tns = torch.reshape(tensor, mhead_tns_shape)
    else:
        raise TypeError(f'invalid tensor type: {type(tensor)}')

    return mhead_tns


def check_tensor_shape(tensor, shape):
    """Check the k-dimensional tensor's shape.

    Args:
    * tensor: input tensor of size D1 x D2 x ... x Dk
    * shape: tensor shape (-1: no restraint)

    Returns: n/a
    """

    assert tensor.ndim == len(shape), \
        f'mismatched number of tensor dimensions: {tensor.ndim} vs. {len(shape)}'
    for idx, dim_len in enumerate(shape):
        assert dim_len in [tensor.shape[idx], -1], \
            f'mismatched dimension length: {tensor.shape[idx]} vs. {dim_len}'
