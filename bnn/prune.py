import torch
from .utils import apply_wb


class PruneNormal():

    def __call__(self, module, percentage=0.5):
        self.prune(module, percentage)

    def prune_param(self, param, percentage, module):
        log_prob = param.dist.log_prob(0)
        log_prob = (log_prob - log_prob.min()) / \
            (log_prob.max() - log_prob.min())
        mask = log_prob < percentage
        param.mean[mask] = 0
        param.scale[mask] = -20

    def prune(self, module, percentage=0.5):
        with torch.no_grad():
            module.traverse(
                lambda m: apply_wb(m, self.prune_param, percentage))
