import torch
import pytest
from torch.nn import init
from pytorch_bayesian.nn import *
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.distributions.kl import kl_divergence
from torch import zeros_like, ones_like, full_like, Size


def allclose(x, y, tol=1e-5):
    return torch.allclose(x, y, atol=tol, rtol=tol)


def eq_dist(x, y):
    if type(x) != type(y):
        return False
    if x.batch_shape != y.batch_shape:
        return False
    return kl_divergence(x, y) == 0


def test_KLDivergence(get_KLDivergence):
    kldiv = KLDivergence()
    for example in get_KLDivergence:
        module, r = example

        if r == 'error':
            with pytest.raises(ValueError):
                kldiv(module)
        else:
            result = kldiv(module)
            assert isinstance(result, torch.Tensor)
            assert result > 0


def test_Entropy(get_Entropy):
    for example in get_Entropy:
        dim, x = example

        entropy = Entropy(dim)

        result = entropy(x)
        assert isinstance(result, torch.Tensor)
        assert result >= 0
