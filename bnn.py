import math
import torch
from torchsummary import summary
from types import SimpleNamespace


def log_gaussians(x, mu, sigma):
    return -math.log(math.sqrt(2 * math.pi)) - torch.log(sigma) - ((x - mu) ** 2) / (2 * sigma ** 2)


class ELBOLoss(torch.nn.Module):
    def __init__(self, number_of_batches):
        super(ELBOLoss, self).__init__()
        self.number_of_batches = number_of_batches

    def _elbo(self, prediction, target, w, prior):
        log_likelihood = torch.nn.functional.cross_entropy(prediction, target)
        log_prior = log_gaussians(w.sampled, prior.mean, prior.stddev).mean()
        log_posterior = log_gaussians(w.sampled, w.mu, w.sigma).mean()

        return (1 / self.number_of_batches) * (log_posterior - log_prior) + log_likelihood

    def forward(self, prediction, target, model):
        result = 0
        num_layers = 0
        for layer in model.layers:
            if isinstance(layer, BayesianLinear):
                W = layer.W
                b = layer.b
                prior = layer.prior

                result += self._elbo(prediction, target, W, prior)
                result += self._elbo(prediction, target, b, prior)

                num_layers += 1

        return result / num_layers


class NormalWeight(torch.nn.Module):

    def __init__(self, *channels):
        super(NormalWeight, self).__init__()

        self.mu = torch.nn.Parameter(
            torch.empty(*channels, requires_grad=True))
        torch.nn.init.uniform_(self.mu, -0.2, 0.2)

        self.rho = torch.nn.Parameter(
            torch.empty(*channels, requires_grad=True))
        torch.nn.init.uniform_(self.rho, -3, 0)

        self.sample()

    @property
    def sigma(self):
        # note perhaphs this one is better
        # return 1 + torch.nn.functional.elu(self.rho)
        return 1e-6 + torch.nn.functional.softplus(self.rho)

    def sample(self):
        self.sampled = self.mu + self.sigma * torch.randn_like(self.mu)


class BayesianLinear(torch.nn.Module):
    prior_types = ['gaussian']

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BayesianLinear, self).__init__()

        self.W = NormalWeight(out_channels, in_channels)
        self.b = NormalWeight(out_channels)

        if 'prior' not in kwargs:
            self.prior = torch.distributions.normal.Normal(0, 0.1)
        elif kwargs['prior'] == 'gaussian':
            self.prior = torch.distributions.normal.Normal(
                kwargs['mean'], kwargs['std'])
        else:
            raise ValueError(
                f'Invalid prior type passed ({prior}), expected one of {prior_types}')

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
            BayesianLinear(in_channels, 1024),
            torch.nn.ReLU(),
            BayesianLinear(1024, out_channels),
            torch.nn.Softmax()
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = BayesianNeuralNetwork(784, 10)
    summary(model, (784,), device='cpu')
