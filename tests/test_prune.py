import torch
import pytest
from bnn.prune import *
from bnn.utils import *


def test_PruneNormal(get_PruneNormal):
    pruner = PruneNormal()
    for example in get_PruneNormal:
        module, p = example

        bw = module.traverse(
            lambda m: apply_wb(m, lambda x, module: x.mean.clone()))

        pruner(module, p)

        aw = module.traverse(
            lambda m: apply_wb(m, lambda x, module: x.mean.clone()))
        dw = [(a != b).float().mean() for b, a in zip(bw, aw)]

        print(dw, p)
        assert torch.stack(dw).mean() - p < 1e-5
