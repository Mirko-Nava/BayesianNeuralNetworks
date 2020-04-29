import torch
import pytest
from torch.nn import init
from torch import zeros_like
from pytorch_bayesian.nn import *
from torch.distributions import Normal
from torch.nn.parameter import Parameter


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
                        zeros_like(wn.sampled))
