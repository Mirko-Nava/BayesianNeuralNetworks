import math
import torch
from torchsummary import summary
from torch.distributions import constraints
from torch.distributions.exp_family import ExponentialFamily


class KLDivergence(torch.nn.Module):
    def __init__(self, number_of_batches):
        super(KLDivergence, self).__init__()
        self.number_of_batches = number_of_batches

    def _kl_normal_normal(self, p, q):
        var_ratio = (p.stddev / q.stddev).pow(2)
        t1 = ((p.mean - q.mean) / q.stddev).pow(2)
        return (var_ratio + t1 - 1 - var_ratio.log()).mean() / self.number_of_batches

    def forward(self, model):
        result = 0
        for layer in model.layers:
            if isinstance(layer, BayesianLinear):
                W = layer.W
                b = layer.b
                prior = layer.prior

                result += self._kl_normal_normal(prior, W)
                result += self._kl_normal_normal(prior, b)

        return result / len(model.layers)


class LinearNormal(torch.nn.Module, ExponentialFamily):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return 1e-5 + torch.nn.functional.softplus(self.scale)

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, *channels):
        super(LinearNormal, self).__init__()

        self.loc = torch.nn.Parameter(
            torch.empty(*channels, requires_grad=True))
        torch.nn.init.uniform_(self.loc, -.2, 0.2)

        self.scale = torch.nn.Parameter(
            torch.empty(*channels, requires_grad=True))
        torch.nn.init.uniform_(self.scale, -2, 0)

        self.sample()

    def sample(self):
        return self.rsample()

    def rsample(self):
        self.sampled = self.mean + self.stddev * torch.randn_like(self.mean)
        return self.sampled

    # def expand(self, batch_shape, _instance=None):
    #     new = self._get_checked_instance(LinearNormal, _instance)
    #     batch_shape = torch.Size(batch_shape)
    #     new.loc = self.loc.expand(batch_shape)
    #     new.scale = self.scale.expand(batch_shape)
    #     super(LinearNormal, new).__init__(batch_shape, validate_args=False)
    #     new._validate_args = self._validate_args
    #     return new

    # def log_prob(self, value):
    #     log_stddev = self.stddev.log()
    #     return -((value - self.loc) ** 2) / (2 * self.variance) - log_stddev - math.log(math.sqrt(2 * math.pi))

    # def cdf(self, value):
    #     return 0.5 * (1 + torch.erf((value - self.mean) * self.stddev.reciprocal() / math.sqrt(2)))

    # def icdf(self, value):
    #     return self.mean + self.stddev * torch.erfinv(2 * value - 1) * math.sqrt(2)

    # def entropy(self):
    #     return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.stddev)

    # @property
    # def _natural_params(self):
    #     return (self.mean / self.stddev.pow(2), -0.5 * self.stddev.pow(2).reciprocal())

    # def _log_normalizer(self, x, y):
    #     return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)


class BayesianLinear(torch.nn.Module):

    def __init__(self, in_channels, out_channels, mean=0, stddev=.1):
        super(BayesianLinear, self).__init__()

        self.W = LinearNormal(out_channels, in_channels)
        self.b = LinearNormal(out_channels)
        self.prior = torch.distributions.normal.Normal(mean, stddev)

    @property
    def sampled(self):
        return self.W.sampled, self.b.sampled

    def sample(self):
        self.W.sample()
        self.b.sample()

    def forward(self, x):
        self.sample()
        return torch.nn.functional.linear(x, *self.sampled)


class BayesianNeuralNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BayesianNeuralNetwork, self).__init__()

        self.layers = torch.nn.Sequential(
            BayesianLinear(in_channels, 512),
            torch.nn.ReLU(),
            BayesianLinear(512, out_channels),
            torch.nn.Softmax(dim=-1)
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = BayesianNeuralNetwork(784, 10)
