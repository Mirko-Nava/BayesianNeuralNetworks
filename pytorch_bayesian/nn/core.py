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
        tril = torch.tril(torch.nn.functional.softplus(self.scale))
        return torch.eye(self.size(-1), device=self.device) * 1e-10 + tril

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


class WeightNormalInverseGamma(Module):

    def __init__(self, *channels):
        super(WeightNormalInverseGamma, self).__init__()

        self.gamma = Parameter(torch.Tensor(*channels))
        self.log_upsilon = Parameter(torch.Tensor(*channels))
        self.log_alpha = Parameter(torch.Tensor(*channels))
        self.log_beta = Parameter(torch.Tensor(*channels))

        self.sampled = WeightNormal(*channels)

        self.sample()

    @property
    def device(self):
        return self.gamma.device

    @property
    def requires_grad(self):
        return self.gamma.requires_grad

    @property
    def upsilon(self):
        return 1e-10 + torch.nn.functional.softplus(self.log_upsilon)

    @property
    def alpha(self):
        return 1 + 1e-10 + torch.nn.functional.softplus(self.log_alpha)

    @property
    def beta(self):
        return 1e-10 + torch.nn.functional.softplus(self.log_beta)

    @property
    def dist(self):
        raise NotImplementedError(
            'Normal-Inverse-Gamma distribution not implemented')

    @property
    def shape(self):
        return self.size()

    def size(self, *dims):
        return self.gamma.size(*dims)

    def sample(self):
        stddev = torch.sqrt(self.beta / (self.upsilon * (self.alpha - 1)))
        scale = stddev + torch.log(1 - torch.exp(-stddev))  # softplus inverse

        with torch.no_grad():
            self.sampled.mean.copy_(self.gamma.clone())
            self.sampled.scale.copy_(scale.clone())

        self.sampled.sample()
