import math
import torch
from torchsummary import summary
from types import SimpleNamespace


def loglike(x, mu, sigma):
    # isotropicc_gauss_loglike
    return -(torch.log(sigma) + .5 * (((x - mu) / sigma) ** 2)).mean()


class NormalWeight(torch.nn.Module):
    def __init__(self, *channels):
        super(NormalWeight, self).__init__()

        self.mu = torch.nn.Parameter(
            torch.Tensor(*channels).uniform_(-0.1, 0.1))

        self.rho = torch.nn.Parameter(torch.Tensor(*channels).uniform_(-3, 0))

    @property
    def std(self):
        # return 1 + torch.nn.functional.elu(self.rho)
        return 1e-6 + torch.nn.functional.softplus(self.rho)

    def forward(self):
        eps = torch.randn_like(self.mu)
        return self.mu + self.std * eps


class BayesianLinear(torch.nn.Module):
    def __init__(self, in_channels, out_channels, prior=torch.distributions.normal.Normal(0, 0.1)):
        super(BayesianLinear, self).__init__()

        self.W = NormalWeight(out_channels, in_channels)
        self.b = NormalWeight(out_channels)
        self.prior = prior

    def forward(self, x, lqw=0, lpw=0):
        W = self.W()
        b = self.b()

        lqw += loglike(W, self.W.mu, self.W.std)
        lqw += loglike(b, self.b.mu, self.b.std)
        lpw += loglike(W, self.prior.mean, self.prior.stddev)
        lpw += loglike(b, self.prior.mean, self.prior.stddev)
        result = torch.nn.functional.linear(x, W, b)

        return result, lqw, lpw


class BayesianNeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BayesianNeuralNetwork, self).__init__()

        self.layers = torch.nn.ModuleList([
            BayesianLinear(in_channels, 256),
            BayesianLinear(256, 256),
            BayesianLinear(256, out_channels)
        ])

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        out = (x,)
        for layer in self.layers:
            out = layer(*out)

        return out


if __name__ == "__main__":
    model = BayesianNeuralNetwork(784, 10)
    # summary(model, (784,), device='cpu')
    # todo summary not working
