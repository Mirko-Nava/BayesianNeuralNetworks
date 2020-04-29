import torch
import pytest
from torch.nn import init
from pytorch_bayesian.nn import *
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.distributions.kl import kl_divergence
from torch import zeros_like, ones_like, full_like


def allclose(x, y, tol=1e-5):
    return torch.allclose(x, y, atol=tol, rtol=tol)


def eq_dist(x, y):
    if type(x) != type(y):
        return False
    if x.batch_shape != y.batch_shape:
        return False
    return kl_divergence(x, y) == 0


def test_BayesianLinear(get_BayesianLinear):
    for example in get_BayesianLinear:
        i, o, b, w, p = example
        bl = BayesianLinear(*example)

        assert isinstance(bl.weight, w)
        assert bl.weight.shape == (o, i)

        if b:
            assert isinstance(bl.bias, w)
            assert bl.bias.shape == (o,)
        else:
            assert bl.bias is None


def test_NormalLinear(get_NormalLinear):
    for example in get_NormalLinear:
        i, o, b, p = example
        nl = NormalLinear(*example)

        assert eq_dist(nl.prior, Normal(0, 1))
        assert isinstance(nl.weight, WeightNormal)
        assert nl.weight.shape == (o, i)
        assert hasattr(nl, 'sample')
        assert hasattr(nl, 'sampled')
        assert isinstance(nl.sampled, tuple)
        assert len(nl.sampled) == 2

        if b:
            assert nl.bias.shape == (o,)
        else:
            assert nl.bias is None

        init.constant_(nl.weight.mean, 1)
        init.constant_(nl.weight.scale, -100)
        if b:
            init.constant_(nl.bias.mean, 3)
            init.constant_(nl.bias.scale, -100)
        nl.sample()

        x = ones_like(nl.weight.mean)
        result = nl(x)

        if b:
            assert allclose(result, full_like(result, i + 3))
        else:
            assert allclose(result, full_like(result, i))


def test_FlipoutNormalLinear(get_FlipoutNormalLinear):
    for example in get_FlipoutNormalLinear:
        i, o, p = example
        fonl = FlipoutNormalLinear(*example)

        assert eq_dist(fonl.prior, Normal(0, 1))
        assert isinstance(fonl.weight, WeightNormal)
        assert fonl.weight.shape == (o, i)
        assert hasattr(fonl, 'sample')
        assert hasattr(fonl, 'sampled')
        assert isinstance(fonl.sampled, tuple)

        assert isinstance(fonl.weight, WeightNormal)
        assert len(fonl.sampled) == 2

        init.constant_(fonl.weight.mean, 1)
        init.constant_(fonl.weight.scale, -100)
        fonl.sample()

        assert hasattr(fonl, 'forward')
        result = fonl(ones_like(fonl.weight.mean))
        assert allclose(result, full_like(result, i))


def test_MCDropoutLinear(get_MCDropoutLinear):
    for example in get_MCDropoutLinear:
        i, o, b, p, s = example
        mcdl = MCDropoutLinear(i, o, b, p)

        assert mcdl.prior is None
        assert mcdl.linear.weight.shape == (o, i)

        if b:
            assert mcdl.linear.bias.shape == (o,)
        else:
            assert mcdl.linear.bias is None

        init.constant_(mcdl.linear.weight, 1)
        if b:
            init.constant_(mcdl.linear.bias, 3)

        x = ones_like(mcdl.linear.weight)
        result = mcdl(x, sample=s)

        assert allclose(torch.tensor(p * s),
                        (result == 0).float().mean(),
                        tol=1e-2)

        if b:
            assert allclose(result.mean(),
                            torch.tensor(i + 3, dtype=torch.float),
                            tol=1e-1)
        else:
            assert allclose(result.mean(),
                            torch.tensor(i, dtype=torch.float),
                            tol=1e-1)
