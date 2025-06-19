"""Basic layers."""
# pylint: skip-file

import torch
from torch import nn


class Mish(nn.Module):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=None):
        super().__init__()

    def forward(self, input):
        return Func.mish(input)


@torch.jit.script
def mish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    """
    return input * torch.tanh(nn.functional.softplus(input))


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def bn_conv1d(in_planes, out_planes, kernel_size, dilated, bias):
    """1-D convolution followed by batch normalization."""
    return nn.Sequential(
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.BatchNorm1d(out_planes),
    )


def in_conv1d(in_planes, out_planes, kernel_size, dilated, bias):
    """1-D convolution followed by instance normalization."""
    return nn.Sequential(
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.InstanceNorm1d(out_planes),
    )


def bn_conv2d(in_planes, out_planes, kernel_size, dilated, bias):
    """2-D convolution followed by batch normalization."""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            dilation=dilated,
            padding=(dilated * (kernel_size - 1) + 1) // 2,
            bias=bias,
        ),
        nn.BatchNorm2d(out_planes),
    )
