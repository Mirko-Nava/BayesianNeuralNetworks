import torch
import pytest
from pytorch_bayesian.nn import *
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


def eq_dist(x, y):
    if type(x) != type(y):
        return False
    if x.batch_shape != y.batch_shape:
        return False
    return kl_divergence(x, y) == 0


def test_BayesianModule():
    bm = BayesianModule(3, 5, Normal(0, 1))

    assert bm.in_channels == 3
    assert bm.out_channels == 5
    assert eq_dist(bm.weight_prior, Normal(0, 1))
    assert eq_dist(bm.bias_prior, Normal(0, 1))


def test_BayesianNetworkModule():
    bnm = BayesianNetworkModule(3, 5, 10)

    assert bnm.in_channels == 3
    assert bnm.out_channels == 5
    assert bnm.samples == 10

    with pytest.raises(NotImplementedError):
        bnm.forward(torch.zeros(2, 3, 5))
