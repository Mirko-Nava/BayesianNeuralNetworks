import torch
from ..utils import apply_wb


class PruneNormal():

    def __call__(self, module, percentage=0.5):
        self.prune(module, percentage)

    def prune_param(self, param, percentage, module):
        log_prob = param.dist.log_prob(0)
        flattened = log_prob.flatten()
        _, indices = torch.topk(flattened, int(percentage * flattened.size(0)))
        mask = torch.zeros_like(flattened)
        mask = mask.scatter(0, indices, 1).bool().view(log_prob.shape)
        param.mean[mask] = 0
        param.scale[mask] = -20

    def prune(self, module, percentage=0.5):
        with torch.no_grad():
            module.traverse(
                lambda m: apply_wb(m, self.prune_param, percentage))
