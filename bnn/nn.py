import math
import torch
from torch.distributions import Normal
from torch.nn import init, Module, Parameter
from torch.distributions.kl import kl_divergence
from .utils import _item_or_list, _single, _pair, _triple, apply_wb, traverse

# Container


class BayesianModule(Module):

    def __init__(self, in_channels, out_channels, prior):
        super(BayesianModule, self).__init__()

        self.prior = prior
        self.in_channels = in_channels
        self.out_channels = out_channels


class BayesianNetworkModule(Module):

    def __init__(self, in_channels, out_channels, prior, samples=30):
        super(BayesianNetworkModule, self).__init__()

        self.prior = prior
        self.samples = samples
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _forward(self, x, *args, **kwargs):
        raise NotImplementedError('self._forward() not implemented')

    def traverse(self, fn, *args, **kwargs):
        return traverse(self, fn, *args, **kwargs)

    def forward(self, x, samples=None, *args, **kwargs):
        if samples is None:
            samples = self.samples

        return _item_or_list([self._forward(x, *args, **kwargs)
                              for _ in range(samples)])


# Weight


class WeightNormal(Module):

    def __init__(self, *channels):
        super(WeightNormal, self).__init__()

        self.mean = Parameter(torch.Tensor(*channels))
        self.scale = Parameter(torch.Tensor(*channels))

        self.shape = self.size()
        self.device = self.mean.device
        self.requires_grad = self.mean.requires_grad

        self.sample()

    @property
    def stddev(self):
        return 1e-6 + torch.nn.functional.softplus(self.scale)

    @property
    def variance(self):
        return self.stddev.pow(2)

    @property
    def dist(self):
        return Normal(self.mean, self.stddev)

    def size(self, *dims):
        return self.mean.size(*dims)

    def sample(self):
        self.sampled = self.mean + self.stddev * torch.randn_like(self.mean)

    def log_prob(self, value):
        log_term = self.stddev.log() + math.log(math.sqrt(2 * math.pi))
        return -((value - self.mean) ** 2) / (2 * self.variance) - log_term


# Linear


class BayesianLinear(BayesianModule):

    def __init__(self, in_channels, out_channels, bias, weight, prior):
        super(BayesianLinear, self).__init__(
            in_channels, out_channels, prior)

        self.weight = weight(out_channels, in_channels)
        if bias:
            self.bias = weight(out_channels)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        pass


class NormalLinear(BayesianLinear):

    def __init__(self, in_channels, out_channels, bias=True,
                 prior=torch.distributions.normal.Normal(0, .1)):
        super(NormalLinear, self).__init__(
            in_channels, out_channels, bias, WeightNormal, prior)

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

    def forward(self, x, sample=True):
        if sample:
            self.sample()

        return torch.nn.functional.linear(x, *self.sampled)


class FlipoutNormalLinear(NormalLinear):

    def __init__(self, in_channels, out_channels,  # todo: no bias
                 prior=torch.distributions.normal.Normal(0, .1)):
        super(FlipoutNormalLinear, self).__init__(
            in_channels, out_channels, False, prior)

    def sample(self):
        self.R = (torch.rand(self.weight.size(0),
                             device=self.weight.device) - .5).sign()
        self.S = (torch.rand(self.weight.size(1),
                             device=self.weight.device) - .5).sign()
        self.sampled = (self.R, self.S)

    def forward(self, x, sample=True):
        if sample:
            self.sample()

        b_part = (x * self.S).matmul(self.weight.stddev.t()) * self.R

        return torch.nn.functional.linear(x, self.weight.mean, b_part)


# Conv


class BayesianConvNd(BayesianModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, groups, bias, weight, prior):
        super(BayesianConvNd, self).__init__(
            in_channels, out_channels, prior)
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


# Recurrent


pass


# Loss


class KLDivergence(torch.nn.Module):
    def __init__(self, number_of_batches=1):
        super(KLDivergence, self).__init__()
        self.n_batches = number_of_batches

    def compute_kl(self, param, module):
        return kl_divergence(
            param.dist, module.prior
        ).mean()

    def forward(self, model):
        result = model.traverse(
            lambda m: apply_wb(m, self.compute_kl))

        if result is None:
            raise ValueError(
                'KLDivergence was not able to find BayasianModules')

        return torch.mean(
            torch.stack(result)) / self.n_batches
