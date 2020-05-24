import math
import torch
from torch.nn import init
from .core import WeightNormal
from .container import BayesianModule
from ..utils import _single, _pair, _triple


class BayesianConvNd(BayesianModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, groups, bias, weight, prior, bias_prior=None):
        super(BayesianConvNd, self).__init__(
            in_channels, out_channels, prior, bias_prior)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.groups = groups

        if transposed:
            self.weight = weight(
                in_channels, out_channels // groups, *kernel_size)
        else:
            self.weight = weight(
                out_channels, in_channels // groups, *kernel_size)

        if bias:
            self.bias = weight(out_channels)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        pass


class NormalConvNd(BayesianConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, groups, bias, prior):
        kernel_size = _single(kernel_size)
        super(NormalConvNd, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, groups, bias, WeightNormal,
            prior)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight.mean, a=math.sqrt(5))
        init.normal_(self.weight.scale, -2.0, 0.15)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight.mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias.mean, -bound, bound)
            init.normal_(self.bias.scale, -2.0, 0.15)

        self.sample()

    def sample(self):
        self.weight.sample()
        self.sampled = (self.weight.sampled,)

        if self.bias is not None:
            self.bias.sample()
            self.sampled += (self.bias.sampled,)
        else:
            self.sampled += (None,)


class NormalConv1d(NormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 prior=torch.distributions.normal.Normal(0, .1)):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(NormalConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, groups, bias, prior)

    def forward(self, x, sample=True):
        if sample:
            self.sample()

        return torch.nn.functional.conv1d(
            x, *self.sampled,
            self.stride, self.padding,
            self.dilation, self.groups)


class NormalConv2d(NormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 prior=torch.distributions.normal.Normal(0, .1)):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(NormalConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, groups, bias, prior)

    def forward(self, x, sample=True):
        if sample:
            self.sample()

        return torch.nn.functional.conv2d(
            x, *self.sampled,
            self.stride, self.padding,
            self.dilation, self.groups)


class NormalConv3d(NormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 prior=torch.distributions.normal.Normal(0, .1)):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(NormalConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, groups, bias, prior)

    def forward(self, x, sample=True):
        if sample:
            self.sample()

        return torch.nn.functional.conv3d(
            x, *self.sampled,
            self.stride, self.padding,
            self.dilation, self.groups)


class FlipOutNormalConvNd(NormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, groups, prior):  # todo: no bias
        kernel_size = _single(kernel_size)
        super(FlipOutNormalConvNd, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, groups, False, prior)

    def sample(self, batch_size=1, additional_dims=()):
        self.R = (torch.rand(batch_size,
                             self.weight.size(0), *additional_dims,
                             device=self.weight.device) - .5).sign()
        self.S = (torch.rand(batch_size,
                             self.weight.size(1), *additional_dims,
                             device=self.weight.device) - .5).sign()
        self.sampled = (self.R, self.S)


class FlipOutNormalConv1d(FlipOutNormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 prior=torch.distributions.normal.Normal(0, .1)):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(FlipOutNormalConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, groups, prior)

    def forward(self, x, sample=True):
        if sample:
            self.sample(x.size(0), (1,))

        output = torch.nn.functional.conv1d(
            x, self.weight.mean, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

        output += torch.nn.functional.conv1d(
            x * self.S.expand_as(x),
            self.weight.stddev, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups) * self.R.expand_as(output)

        return output


class FlipOutNormalConv2d(FlipOutNormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 prior=torch.distributions.normal.Normal(0, .1)):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(FlipOutNormalConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, groups, prior)

    def forward(self, x, sample=True):
        if sample:
            self.sample(x.size(0), (1, 1))

        output = torch.nn.functional.conv2d(
            x, self.weight.mean, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

        output += torch.nn.functional.conv2d(
            x * self.S.expand_as(x),
            self.weight.stddev, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups) * self.R.expand_as(output)

        return output


class FlipOutNormalConv3d(FlipOutNormalConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 prior=torch.distributions.normal.Normal(0, .1)):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(FlipOutNormalConv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, groups, prior)

    def forward(self, x, sample=True):
        if sample:
            self.sample(x.size(0), (1, 1, 1))

        output = torch.nn.functional.conv3d(
            x, self.weight.mean, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

        output += torch.nn.functional.conv3d(
            x * self.S.expand_as(x),
            self.weight.stddev, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups) * self.R.expand_as(output)

        return output


class MCDropoutConvNd(BayesianModule):

    def __init__(self, in_channels, out_channels, drop_prob):
        super(MCDropoutConvNd, self).__init__(
            in_channels, out_channels, None)

        self.drop_prob = drop_prob


class MCDropoutConv1d(MCDropoutConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, drop_prob=0.5):
        super(MCDropoutConv1d, self).__init__(
            in_channels, out_channels, drop_prob)

        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

    def forward(self, x, sample=True):
        return torch.nn.functional.dropout(
            self.conv(x),
            self.drop_prob,
            sample, False
        )


class MCDropoutConv2d(MCDropoutConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, drop_prob=0.5):
        super(MCDropoutConv2d, self).__init__(
            in_channels, out_channels, drop_prob)

        self.conv = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

    def forward(self, x, sample=True):
        return torch.nn.functional.dropout(
            self.conv(x),
            self.drop_prob,
            sample, False
        )


class MCDropoutConv3d(MCDropoutConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, drop_prob=0.5):
        super(MCDropoutConv3d, self).__init__(
            in_channels, out_channels, drop_prob)

        self.conv = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias)

    def forward(self, x, sample=True):
        return torch.nn.functional.dropout(
            self.conv(x),
            self.drop_prob,
            sample, False
        )
