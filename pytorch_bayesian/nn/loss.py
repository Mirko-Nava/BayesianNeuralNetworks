import torch
from torch.nn import Module
from ..utils import apply_wb
from torch.distributions.kl import kl_divergence


class KLDivergence(Module):
    def __init__(self, number_of_batches=1):
        super(KLDivergence, self).__init__()
        self.n_batches = number_of_batches

    def compute_kl(self, param, module):
        return kl_divergence(
            param.dist, module.prior
        ).mean()

    def forward(self, model):
        result = model.traverse(
            lambda m: apply_wb(m, self.compute_kl, pass_module=True))

        if result is None:
            raise ValueError(
                'KLDivergence was not able to find BayasianModules')

        return torch.stack(result).mean() / self.n_batches


class Entropy(Module):
    def __init__(self, dim=0):
        super(Entropy, self).__init__()
        self.dim = dim

    def forward(self, x):
        return (-x * torch.log(x + 1e-10)).sum(dim=self.dim).mean()
