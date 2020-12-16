import torch
import pytest
from torch.nn import init
from torch import zeros_like
from pytorch_bayesian.nn import *
from torch.nn.parameter import Parameter
from torch.distributions import Normal, MultivariateNormal


def allclose(x, y, tol=1e-5):
    return torch.allclose(x, y, atol=tol, rtol=tol)


def test_WeightNormal(get_WeightNormal):
    for example in get_WeightNormal:
        wn = WeightNormal(*example)

        assert isinstance(wn.mean, Parameter)
        assert isinstance(wn.scale, Parameter)
        assert wn.mean.shape == example
        assert wn.scale.shape == example
        assert wn.shape == wn.mean.shape
        assert wn.device == wn.mean.device
        assert wn.requires_grad == wn.mean.requires_grad
        assert hasattr(wn, 'sampled')
        assert isinstance(wn.sampled, torch.Tensor)
        assert isinstance(wn.dist, Normal)
        assert wn.size() == example
        assert wn.size(0) == example[0]

        init.constant_(wn.mean, 0)
        init.constant_(wn.scale, -100)
        wn.sample()

        assert (wn.stddev > 0).all()
        assert (wn.stddev ** 2 == wn.variance).all()
        assert allclose(wn.sampled,
                        zeros_like(wn.sampled),
                        tol=1e-5)


def test_WeightMultivariateNormal(get_WeightMultivariateNormal):
    for example in get_WeightMultivariateNormal:
        wmn = WeightMultivariateNormal(*example)

        assert isinstance(wmn.mean, Parameter)
        assert isinstance(wmn.scale, Parameter)
        assert wmn.mean.shape == example
        assert wmn.scale.shape == (*example, example[-1])
        assert wmn.shape == wmn.mean.shape
        assert wmn.device == wmn.mean.device
        assert wmn.requires_grad == wmn.mean.requires_grad
        assert hasattr(wmn, 'sampled')
        assert isinstance(wmn.sampled, torch.Tensor)
        assert isinstance(wmn.dist, MultivariateNormal)
        assert wmn.size() == example
        assert wmn.size(0) == example[0]

        init.constant_(wmn.mean, 0)
        init.constant_(wmn.scale, -100)
        wmn.sample()

        assert allclose(wmn.stddev, torch.tril(wmn.stddev))
        assert ((wmn.stddev > 0) | torch.triu(
            torch.ones_like(wmn.stddev), 1).to(torch.bool)).all()
        assert (wmn.stddev == wmn.variance.sqrt()).all()
        assert allclose(wmn.sampled.mean(),
                        torch.zeros(1),
                        tol=1e-5)
