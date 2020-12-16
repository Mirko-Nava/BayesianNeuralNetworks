import math
import torch
import warnings
from torch.nn import Module
from ..utils import apply_wb
from .dense import MultivariateNormalLinear
from torch.distributions.kl import kl_divergence
from torch.distributions import MultivariateNormal


class KLDivergence(Module):
    def __init__(self, number_of_batches=1):
        super(KLDivergence, self).__init__()
        self.n_batches = number_of_batches

    def compute_kl(self, param, module, type):
        if type == 'w':
            prior = module.weight_prior
        else:
            prior = module.bias_prior

        # note: we must convert to cpu as distributions containing tensors are not
        # parameters and do not get converted when calling .to(device)
        if isinstance(module, MultivariateNormalLinear):
            prior = MultivariateNormal(prior.mean.to(param.device),
                                       scale_tril=prior.scale_tril.to(param.device))

        return kl_divergence(param.dist, prior).mean()

    def forward(self, model):
        result = model.traverse(
            lambda m: apply_wb(m, self.compute_kl, pass_module=True, pass_type=True))

        if result is None:
            raise ValueError(
                'KLDivergence was not able to find BayasianModules')

        return torch.stack(result).mean() / self.n_batches


class Entropy(Module):
    def __init__(self, dim=0):
        super(Entropy, self).__init__()
        self.dim = dim

    def forward(self, x):
        if (x == 0).all():
            warnings.warn(
                'Entropy received a tensor containing all zeros',
                RuntimeWarning)
        return (-x * torch.log(x + 1e-10)).sum(dim=self.dim).mean()


class NormalInverseGaussianLoss(Module):
    def __init__(self, reg_lambda=1e-2):
        super(NormalInverseGaussianLoss, self).__init__()
        self.reg_lambda = reg_lambda

    def nll(self, y, gamma, upsilon, alpha, beta):
        omega = 2 * beta * (1 + upsilon)
        return 0.5 * torch.log(math.pi / upsilon)  \
            - alpha * torch.log(omega)  \
            + (alpha + 0.5) * torch.log(upsilon * (y - gamma) ** 2 + omega)  \
            + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    def forward(self, y, gamma, upsilon, alpha, beta):
        regularizer = torch.mean(torch.abs(y - gamma) * (2 * upsilon + alpha))
        nll = self.nll(y, gamma, upsilon, alpha, beta).mean()
        return nll + self.reg_lambda * regularizer
