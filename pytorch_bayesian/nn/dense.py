import math
import torch
from torch.nn import init
from .container import BayesianModule
from torch.distributions import Normal
from .core import WeightNormal, WeightMultivariateNormal


class BayesianLinear(BayesianModule):

    def __init__(self, in_features, out_features, bias, weight, prior, bias_prior=None):
        super(BayesianLinear, self).__init__(
            in_features, out_features, prior, bias_prior)

        self.weight = weight(out_features, in_features)
        if bias:
            self.bias = weight(out_features)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        pass


class NormalLinear(BayesianLinear):

    def __init__(self, in_features, out_features, bias=True,
                 prior=torch.distributions.normal.Normal(0, .1)):
        super(NormalLinear, self).__init__(
            in_features, out_features, bias, WeightNormal, prior)

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

    def __init__(self, in_features, out_features,  # todo: no bias
                 prior=torch.distributions.normal.Normal(0, .1)):
        super(FlipoutNormalLinear, self).__init__(
            in_features, out_features, False, prior)

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


class MultivariateNormalLinear(BayesianLinear):

    def __init__(self, in_features, out_features, bias=True, weight_prior=None, bias_prior=None):
        if not weight_prior:
            weight_prior = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(out_features, in_features),
                torch.eye(in_features).repeat(out_features, 1, 1))

        if bias and not bias_prior:
            bias_prior = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(out_features),
                torch.eye(out_features))

        super(MultivariateNormalLinear, self).__init__(
            in_features, out_features, bias, WeightMultivariateNormal, weight_prior, bias_prior)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight.mean, a=math.sqrt(5))
        init.normal_(self.weight.scale, -2.0, 0.15)

        with torch.no_grad():
            triu = torch.triu(torch.ones_like(
                self.weight.scale), 1).to(torch.bool)
            self.weight.scale[triu] = -100

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight.mean)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias.mean, -bound, bound)
            init.normal_(self.bias.scale, -2.0, 0.15)

            with torch.no_grad():
                triu = torch.triu(torch.ones_like(
                    self.bias.scale), 1).to(torch.bool)
                self.bias.scale[triu] = -100

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


class NormalInverseGammaLinear(BayesianModule):

    def __init__(self, in_features, out_features, bias=True):
        super(NormalInverseGammaLinear, self).__init__(
            in_features, out_features, None)

        self.linear = torch.nn.Linear(in_features, 4 * out_features, bias)

    def forward(self, x, sample=False):
        gamma, upsilon, alpha, beta = torch.split(
            self.linear(x), self.out_channels, dim=-1)

        upsilon = 1e-10 + torch.nn.functional.softplus(upsilon)
        alpha = 1 + 1e-10 + torch.nn.functional.softplus(alpha)
        beta = 1e-10 + torch.nn.functional.softplus(beta)

        if sample:
            mean = gamma.clone()
            stddev = torch.sqrt(beta / (upsilon * (alpha - 1)))
            return Normal(mean, stddev)
        else:
            return (gamma, upsilon, alpha, beta)


class MCDropoutLinear(BayesianModule):

    def __init__(self, in_features, out_features, bias=True, drop_prob=0.5):
        super(MCDropoutLinear, self).__init__(
            in_features, out_features, None)

        self.drop_prob = drop_prob
        self.linear = torch.nn.Linear(in_features, out_features, bias)

    def forward(self, x, sample=True):
        return torch.nn.functional.dropout(
            self.linear(x),
            self.drop_prob,
            sample, False
        )
