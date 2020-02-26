import math
import torch
from torchsummary import summary


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batchsize = x.shape[0]
        return x.view(batchsize, -1)


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
            if isinstance(layer, BayesianLinearNormal):
                W = layer.weight
                b = layer.bias
                prior = layer.prior

                result += self._kl_normal_normal(prior, W)
                result += self._kl_normal_normal(prior, b)

        return result / len(model.layers)


class LinearNormal(torch.nn.Module):

    def __init__(self, *channels):
        super(LinearNormal, self).__init__()

        self.mean = torch.nn.Parameter(
            torch.empty(*channels, requires_grad=True))
        torch.nn.init.uniform_(self.mean, -1, 1)

        self.scale = torch.nn.Parameter(
            torch.empty(*channels, requires_grad=True))
        torch.nn.init.normal_(self.scale, -2, 1)

        self.sample()

    def size(self):
        return self.mean.size()

    @property
    def requires_grad(self):
        return self.mean.requires_grad

    @property
    def stddev(self):
        return 1e-6 + torch.nn.functional.softplus(self.scale)

    @property
    def variance(self):
        return self.stddev.pow(2)

    def sample(self):
        self.sampled = self.mean + self.stddev * torch.randn_like(self.mean)
        return self.sampled

    def log_prob(self, value):
        log_stddev = self.stddev.log()
        return -((value - self.mean) ** 2) / (2 * self.variance) - log_stddev - math.log(math.sqrt(2 * math.pi))

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


class BayesianLinearNormal(torch.nn.Module):

    def __init__(self, in_channels, out_channels, prior=torch.distributions.normal.Normal(0, .1)):
        super(BayesianLinearNormal, self).__init__()

        self.weight = LinearNormal(out_channels, in_channels)
        self.bias = LinearNormal(out_channels)
        self.prior = prior

    @property
    def sampled(self):
        return self.weight.sampled, self.bias.sampled

    def sample(self):
        self.weight.sample()
        self.bias.sample()

    def forward(self, x):
        self.sample()
        return torch.nn.functional.linear(x, *self.sampled)


class BayesianNeuralNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_channels, prior=torch.distributions.normal.Normal(0, .1)):
        super(BayesianNeuralNetwork, self).__init__()

        self.layers = torch.nn.Sequential(
            BayesianLinearNormal(in_channels, 512, prior=prior),
            torch.nn.ReLU(),
            BayesianLinearNormal(512, out_channels, prior=prior),
            torch.nn.Softmax(dim=-1)
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = BayesianNeuralNetwork(784, 10).to('cuda')
    model.summary((1, 784))
