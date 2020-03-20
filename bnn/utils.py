from itertools import repeat
from collections.abc import Iterable
from torch.nn.modules.container import Sequential, ModuleList, ModuleDict


# Misc


def _item_or_list(n):
    return n[0] if len(n) == 1 else n


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)


# Traversal


def apply_wb(module, fn, *args, **kwargs):
    result = []

    for param in [module.weight, module.bias]:
        if param is not None:
            kwargs['module'] = module
            res = fn(param, *args, **kwargs)

            if res is not None:
                result.append(res)

    if result:
        return result


def traverse(module, fn, *args, **kwargs):
    from .nn import BayesianModule, BayesianNetworkModule

    if isinstance(module, (ModuleList, ModuleDict, Sequential, BayesianNetworkModule)):
        result = []
        for m in module.children():
            res = traverse(m, fn, *args, **kwargs)
            if res is not None:
                result += res

        if result:
            return result
    elif isinstance(module, BayesianModule):
        result = fn(module, *args, **kwargs)
    else:
        result = None

    if isinstance(result, list) and result:
        return result
