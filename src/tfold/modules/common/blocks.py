"""Basic blocks."""
# pylint: skip-file

from tfold.modules.common.basics import *


class BottleNeck(nn.Module):
    """Residual block w/ bottleneck."""

    def __init__(self, inplanes, planes, stride=1,  dilation=1, norm_layer=None):
        super(BottleNeck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes//2, planes//2, stride, dilation=dilation)

        self.bn3 = norm_layer(inplanes//2)
        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(inplanes//2, planes, stride)

        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out += identity
        return out


class SEModule(nn.Module):
    """Squeeze-n-excitation module."""

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEBottleneck(nn.Module):
    """Squeeze-n-excitation block w/ bottleneck."""

    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, norm_layer=None):
        super(SEBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.bn2 = norm_layer(planes // 2)
        self.bn3 = norm_layer(planes // 2)

        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes//2, planes//2, stride, groups=groups, dilation=dilation)

        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(planes//2, planes, stride)
        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.se_module(out) + identity
        return out


class SEBottleneckInvolution(nn.Module):
    """Squeeze-n-excitation block w/ bottleneck and in-convolution."""

    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, norm_layer=None):
        super(SEBottleneckInvolution, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = involution(channels=planes//2, kernel_size=11, stride=1)

        self.bn3 = norm_layer(inplanes//2)
        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(inplanes//2, planes, stride)
        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.se_module(out) + identity
        return out


class SEBottleneckInvolutionFix(nn.Module):
    """Squeeze-n-excitation block w/ bottleneck and in-convolution (bug fixed?)."""

    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, norm_layer=None):
        super(SEBottleneckInvolutionFix, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes, planes//2, stride)
        self.bn1 = norm_layer(inplanes//2)
        self.relu1 = nn.ELU(inplace=True)

        self.conv2 = involution(channels=planes//2, kernel_size=9, stride=1)
        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)

        self.conv3 = conv1x1(inplanes//2, planes, stride)
        self.bn3 = norm_layer(planes)
        self.relu3 = nn.ELU(inplace=True)

        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.se_module(out) + identity
        return out


class DeformSEBottleneck(nn.Module):
    """Deformed squeeze-n-excitation block w/ bottleneck."""

    def __init__(self, inplanes, planes, stride=1,  dilation=1, norm_layer=None):
        super(DeformSEBottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv1 = conv1x1(inplanes, planes//2, stride)

        self.bn2 = norm_layer(planes//2)
        self.relu2 = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes//2, planes//2, stride, dilation=dilation, is_deform=True)

        self.bn3 = norm_layer(inplanes//2)
        self.relu3 = nn.ELU(inplace=True)
        self.conv3 = conv1x1(inplanes//2, planes, stride)
        self.se_module = SEModule(planes, reduction=16)
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        out = self.se_module(out) + identity
        return out


class SEDilatedBottleneck(nn.Module):
    """Squeeze-n-excitation block w/ bottleneck and dilated convolution."""

    def __init__(self, inplanes, planes, stride=1,  dilation=1, norm_layer=None):
        super(SEDilatedBottleneck, self).__init__()

        dplanes = planes // 4
        self.b1 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=1)
        self.b2 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=2)
        self.b4 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=4)
        self.b8 = SEBottleneck(inplanes, dplanes, stride=1,  dilation=8)


    def forward(self, x):

        b1 = self.b1(x)
        b2 = self.b2(x)
        b4 = self.b4(x)
        b8 = self.b8(x)

        out = torch.cat([b1, b2, b4, b8], dim=1)
        return out
