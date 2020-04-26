import math
import torch
from torch.nn import Module
from torch.distributions import Normal
from torch.nn.parameter import Parameter


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
