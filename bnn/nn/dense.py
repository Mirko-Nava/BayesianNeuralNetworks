import math
import torch
from torch.nn import init
from .core import WeightNormal
from .container import BayesianModule


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
