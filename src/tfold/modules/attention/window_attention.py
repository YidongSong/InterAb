# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/19 14:24
# Author: chenchenqin
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.modules.common import Linear

from tfold.utils import pad_at_dim
from tfold.utils import permute_final_dims
from tfold.utils import flatten_final_dims


def repeat_kv(x: torch.Tensor, repeats: int, dim=-2) -> torch.Tensor:
    if repeats == 1:
        return x

    return torch.repeat_interleave(x, dim=dim, repeats=repeats)


def padding_same(seq_len, kernel, stride):
    if seq_len % stride == 0:
        pad = max(kernel - stride, 0)
    else:
        pad = max(kernel - (seq_len % stride), 0)
    return pad


class WindowAttention(nn.Module):
    """sequence-local atom attention is equivalent to self attention within rectangular blocks along the diagonal.
    """

    def __init__(self,
                 c_q: int,
                 *,
                 c_k: int = None,
                 c_v: int = None,
                 dim: int = None,
                 window_size=None,
                 num_heads,
                 num_kv_heads=None,
                 bias=False,
                 dropout=0.0,
                 pack_qkv=True,
                 gating=False):
        super(WindowAttention, self).__init__()
        dim = dim or c_q
        assert dim % num_heads == 0, f"number of heads({num_heads}) must be divisible by hiddens({dim})"
        self.dim = dim
        self.c_q = c_q
        self.c_k = c_k or c_q
        self.c_v = c_v or c_q
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.bias = bias
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.repeats = self.dim // self.kv_dim
        self.attn_dropout = dropout
        self.pack_qkv = pack_qkv
        if self.pack_qkv:
            self.linear_qkv = Linear(self.dim, self.dim + 2 * self.kv_dim, bias=self.bias)
        else:
            self.linear_q = Linear(self.dim, self.dim, bias=self.bias)
            self.linear_k = Linear(self.dim, self.kv_dim, bias=self.bias)
            self.linear_v = Linear(self.dim, self.kv_dim, bias=self.bias)

        self.linear_o = Linear(self.dim, self.dim, bias=self.bias, init="final")
        self.gating = gating
        if self.gating:
            self.linear_g = Linear(self.dim, self.dim, bias=self.bias, init="gating")
        self.window = (window_size, window_size) if isinstance(window_size, int) else window_size
        if self.window is not None:
            assert self.window[0] <= self.window[1], f"query window must be smaller than value window"

    def _project_qkv(self,
                     q: torch.Tensor,
                     k: torch.Tensor = None,
                     v: torch.Tensor = None,
                     ):
        """
        Args:
            q_x: [*, Lq, C] query data, is also sequence feature s

        Returns:
            q, k, v: tensor[*, seq_len, num_heads, head_dim]
        """
        k = q if k is None else k
        v = k if v is None else v
        if self.pack_qkv:
            q, k, v = self.linear_qkv(q).split([self.dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            q = self.linear_q(q)
            k = self.linear_k(k)
            v = self.linear_v(v)

        q = q.view(q.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_kv_heads, head_dim]
        k = k.view(k.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_q_head, head_dim]
        v = v.view(v.shape[:-1] + (-1, self.head_dim))  # [*, seq_len, num_kv_heads, head_dim]

        k = repeat_kv(k, self.repeats)  # [*, seq_len, num_heads, head_dim]
        v = repeat_kv(v, self.repeats)

        return q, k, v

    def unfold_diag_block_scan(self, attn_mask, window_size):
        """deprecated unfold take too much memory and slow"""
        local_q_len, local_kv_len = window_size
        q_len = attn_mask.shape[-2]
        num_boxes = q_len // local_q_len
        # unfold only support 4d tensor and dtype not be bool
        print(f"1 gpu used {torch.cuda.max_memory_allocated(device=None) / 1024 / 1024 / 1024} memory")
        local_attn_mask = F.unfold(attn_mask.reshape(-1, *attn_mask.shape[-3:]).half(),
                                   kernel_size=window_size,
                                   stride=(local_q_len, local_q_len)
                                   ).to(attn_mask.dtype)  # [*, num_heads * w1 * w2, nq * nk]
        attn_mask = local_attn_mask.reshape(*attn_mask.shape[:-2], *window_size, num_boxes, num_boxes)
        attn_mask = torch.diagonal(attn_mask, dim1=-1, dim2=-2)  # [*, num_heads, w1, w2, n]
        attn_mask = permute_final_dims(attn_mask, [3, 0, 1, 2])  # [*, n, num_heads, w1, w2]
        return attn_mask

    def diag_block_scan(self, attn_mask, window_size):
        local_q_len, local_kv_len = window_size
        q_len, kv_len = attn_mask.shape[-2:]
        num_boxes = q_len // local_q_len
        masks = []
        for i in range(num_boxes):
            offset = i * local_q_len
            masks.append(
                attn_mask[..., offset: offset + local_q_len, offset:offset + local_kv_len]
            )
        attn_mask = torch.stack(masks, dim=-4)
        return attn_mask

    def _local_scaled_dot_product_attention(self,
                                            query,
                                            key,
                                            value,
                                            window_size,
                                            attn_mask=None,
                                            dropout_p=0.0):
        """
        Args:
            q, k, v: tensor[*, seq_len, num_heads, head_dim]
            attn_mask: [*, num_heads, seq_len, seq_len], attention bool mask or float bias
        """
        *batch_dims, seq_len, num_heads, head_dim = query.shape
        local_q_len, local_kv_len = window_size
        pad_q_len = padding_same(seq_len, kernel=local_q_len, stride=local_q_len)
        pad_kv_len = padding_same(seq_len, kernel=local_kv_len, stride=local_q_len)
        if pad_q_len > 0:
            query = pad_at_dim(query, (0, pad_q_len), dim=-3)

        if pad_kv_len > 0:
            key = pad_at_dim(key, (0, pad_kv_len), dim=-3)
            value = pad_at_dim(value, (0, pad_kv_len), dim=-3)

        if attn_mask is not None:
            if pad_q_len > 0 or pad_kv_len > 0:
                # pad from rignt to left
                attn_mask = F.pad(attn_mask, (0, pad_kv_len, 0, pad_q_len), value=0)
            attn_mask = self.diag_block_scan(attn_mask, window_size)  # [*, n, num_head, local_q_len, local_kv_len]
        # note that unfold size is [*, n, num_heads, head_dim, local_len]
        q = query.unfold(dimension=-3, size=local_q_len, step=local_q_len).transpose(
            -1, -2)  # [*, n, num_head, local_q_len, head_dim]
        k = key.unfold(dimension=-3, size=local_kv_len, step=local_q_len).transpose(
            -1, -2)  # [*, n, num_head, local_kv_len, head_dim]
        v = value.unfold(dimension=-3, size=local_kv_len, step=local_q_len).transpose(
            -1, -2)  # [*, n, num_head, local_q_len, head_dim]
        # lazy expand batch dims for saving memory
        if attn_mask is not None and attn_mask.shape[0] != q.shape[0]:
            n = q.shape[0] // attn_mask.shape[0]
            attn_mask = attn_mask.repeat_interleave(n, dim=0)
        y = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=attn_mask,
                                           dropout_p=dropout_p)  # [*, n, num_heads, w1, head_dim]
        y = y.transpose(-2, -3)  # [*, n, w1, num_heads, head_dim]
        return y.reshape(*batch_dims, -1, num_heads * head_dim)[..., :seq_len, :]

    def _scaled_dot_product_attention(self,
                                      q, k, v,
                                      attn_mask=None,
                                      dropout_p=0.0):
        """
        Args:
            q, k, v: tensor[*, q_len or kv_len, num_heads, head_dim]
            attn_mask: [*, num_heads, q_len, kv_len], attention bool mask or float bias

        Returns:
            out: [*, seq_len, dim]
        """
        q = q.transpose(-2, -3)  # [*, num_heads, seq_len, dim]
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        y = F.scaled_dot_product_attention(q, k, v,
                                           attn_mask=attn_mask,
                                           dropout_p=dropout_p,
                                           )  # [*, num_heads, seq_len, head_dim]
        y = y.transpose(-2, -3)

        return flatten_final_dims(y, 2)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor = None,
                value: torch.Tensor = None,
                attn_mask: Optional[torch.Tensor] = None,
                attn_bias: Optional[torch.Tensor] = None):
        """
        Args:
            x: [*, seq_len, dim]
            attn_mask: [*, 1, seq_len, seq_len]
            attn_bias: [*, 1, seq_len, seq_len], tensor of list of attention biases

        Returns:
            y: [*, seq_len, dim], ouptut hiddens
        """
        q, k, v = self._project_qkv(query, key, value)

        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        if attn_bias is not None:
            if isinstance(attn_bias, (list, tuple)):
                attn_bias = sum(attn_bias)

            if attn_mask is not None:
                if attn_mask.shape != attn_bias.shape:
                    attn_mask = attn_mask.expand_as(attn_bias)
                attn_bias.masked_fill_(~attn_mask, torch.finfo(attn_bias.dtype).min)

            attn_mask = attn_bias.to(query.dtype)

        dropout_p = self.attn_dropout if self.training else 0.0
        seq_len = query.shape[-2]
        if self.window is None or seq_len <= max(self.window):
            y = self._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        else:
            if len(query.shape) > 3:
                # force q, k, v 4d tensor
                q, k, v = [xs.reshape(-1, *xs.shape[-3:]) for xs in [q, k, v]]  # [bs, seq_len, num_heads, head_dim]
                attn_mask = attn_mask.reshape(-1, *attn_mask.shape[-3:])

            y = self._local_scaled_dot_product_attention(q, k, v,
                                                         window_size=self.window,
                                                         attn_mask=attn_mask,
                                                         dropout_p=dropout_p)
            if len(query.shape) > 3:
                batch_dims = query.shape[:-2]
                y = y.reshape(*batch_dims, *y.shape[-2:])

        if self.gating:
            y = y * self.linear_g(query).sigmoid()

        y = self.linear_o(y)

        return y
