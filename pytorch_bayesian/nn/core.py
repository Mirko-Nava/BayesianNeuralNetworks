import math
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.distributions import Normal, MultivariateNormal


class WeightNormal(Module):

    def __init__(self, *channels):
        super(WeightNormal, self).__init__()

        self.mean = Parameter(torch.Tensor(*channels))
        self.scale = Parameter(torch.Tensor(*channels))

        self.sample()

    @property
    def device(self):
        return self.mean.device

    @property
    def requires_grad(self):
        return self.mean.requires_grad

    @property
    def stddev(self):
        return 1e-10 + torch.nn.functional.softplus(self.scale)

    @property
    def variance(self):
        return self.stddev.pow(2)

    @property
    def dist(self):
        return Normal(self.mean, self.stddev)

    @property
    def shape(self):
        return self.size()

    def size(self, *dims):
        return self.mean.size(*dims)

    def sample(self):
        self.sampled = self.mean + self.stddev * torch.randn_like(self.mean)


class WeightMultivariateNormal(Module):

    def __init__(self, *channels):
        super(WeightMultivariateNormal, self).__init__()

        self.mean = Parameter(torch.Tensor(*channels))
        self.scale = Parameter(
            torch.eye(channels[-1]).repeat(*channels[:-1], 1, 1))

        self.sample()

    @property
    def device(self):
        return self.mean.device

    @property
    def requires_grad(self):
        return self.mean.requires_grad

    @property
    def stddev(self):
        return self.variance.sqrt()

    @property
    def variance(self):
        return torch.eye(self.size(-1), device=self.device) * 1e-10 + \
            torch.nn.functional.softplus(self.scale)

    @property
    def dist(self):
        return MultivariateNormal(
            self.mean,
            scale_tril=self.variance)

    @property
    def shape(self):
        return self.size()

    def size(self, *dims):
        return self.mean.size(*dims)

    def sample(self):
        self.sampled = self.mean + \
            torch.matmul(self.stddev, torch.rand_like(
                self.mean).unsqueeze(-1)).squeeze(-1)
