import torch
import pytest
from pytorch_bayesian.prune import *
from pytorch_bayesian.utils import *


def test_PruneNormal(get_PruneNormal):
    pruner = PruneNormal()
    for example in get_PruneNormal:
        module, p = example

        bw = module.traverse(
            lambda m: apply_wb(m, lambda x, module: x.mean.clone(),
                               pass_module=True))

        pruner(module, p)

        aw = module.traverse(
            lambda m: apply_wb(m, lambda x, module: x.mean.clone(),
                               pass_module=True))
        dw = [(a != b).float().mean() for b, a in zip(bw, aw)]

        print(dw, p)
        assert torch.stack(dw).mean() - p < 1e-5
