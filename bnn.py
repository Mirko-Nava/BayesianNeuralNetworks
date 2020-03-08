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


class NormalWeight(torch.nn.Module):

    def __init__(self, *channels):
        super(NormalWeight, self).__init__()

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
    def shape(self):
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


class BayesianLinearNormal(torch.nn.Module):

    def __init__(self, in_channels, out_channels, prior=torch.distributions.normal.Normal(0, .1)):
        super(BayesianLinearNormal, self).__init__()

        self.weight = NormalWeight(out_channels, in_channels)
        self.bias = NormalWeight(out_channels)
        self.prior = prior

    @property
    def sampled(self):
        return self.weight.sampled, self.bias.sampled

    def sample(self):
        self.weight.sample()
        self.bias.sample()

    def forward(self, x, sample=True):
        if sample:
            self.sample()

        return torch.nn.functional.linear(x, *self.sampled)


class BayesianModelBase(torch.nn.Module):

    def __init__(self, in_channels, out_channels, prior, samples):
        super(BayesianModelBase, self).__init__()

        self.prior = prior
        self.samples = samples
        self.in_channels = in_channels
        self.out_channels = out_channels


class BayesianNeuralNetwork(BayesianModelBase):

    def __init__(self, in_channels, out_channels,
                 prior=torch.distributions.normal.Normal(0, .1), samples=30):
        super(BayesianNeuralNetwork, self).__init__(
            in_channels, out_channels, prior, samples
        )

        self.layers = torch.nn.Sequential(
            BayesianLinearNormal(in_channels, 512, prior=prior),
            torch.nn.ReLU(),
            BayesianLinearNormal(512, out_channels, prior=prior),
            torch.nn.Softmax(dim=-1)
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x, samples=None):
        if samples is None:
            samples = self.samples

        return [self.layers(x) for _ in range(samples)]


class BayesianConvolutionalNeuralNetwork(BayesianModelBase):

    def __init__(self, in_channels, out_channels,
                 prior=torch.distributions.normal.Normal(0, .1), samples=30):
        super(BayesianConvolutionalNeuralNetwork, self).__init__(
            in_channels, out_channels, prior, samples
        )

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 8, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(8, 8, 3),
            torch.nn.ReLU(),
            Flatten(),
            BayesianLinearNormal(968, 512, prior=prior),
            torch.nn.ReLU(),
            BayesianLinearNormal(512, out_channels, prior=prior),
            torch.nn.Softmax(dim=-1)
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x, samples=None):
        if samples is None:
            samples = self.samples

        return [self.layers(x) for _ in range(samples)]


class PruneBayesianNormal():

    def _prune_linear_normal(self, weight, percentage):
        shape = weight.shape
        log_prob = weight.log_prob(0)
        log_prob = (log_prob - log_prob.min()) / \
            (log_prob.max() - log_prob.min())
        mask = log_prob < percentage
        weight.mean[mask] = 0
        weight.scale[mask] = -20

    def prune(self, model, percentage=0.5):
        for layer in model.layers:
            if type(layer) == BayesianLinearNormal:
                self._prune_linear_normal(layer.weight,
                                          percentage)


if __name__ == "__main__":
    model = BayesianNeuralNetwork(784, 10, samples=1).to('cuda')
    model.summary([(1, 784)], device='cuda')
