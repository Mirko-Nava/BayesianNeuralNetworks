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

        if (x == 0).all():
            with pytest.warns(RuntimeWarning):
                result = entropy(x)
        else:
            result = entropy(x)
        assert isinstance(result, torch.Tensor)
        assert result >= 0


def test_NormalInverseGaussianLoss(get_NormalInverseGaussianLoss):
    for example in get_NormalInverseGaussianLoss:
        l = example[0]
        g, v, a, b = (torch.tensor([[p]]) for p in example[1:])

        loss_function = NormalInverseGaussianLoss(l)
        y = zeros_like(g)

        result = loss_function(y, g, v, a, b)
        print(result)
        assert isinstance(result, torch.Tensor)
        assert result >= 0
