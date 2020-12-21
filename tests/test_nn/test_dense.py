import torch
import pytest
from torch.nn import init
from pytorch_bayesian.nn import *
from torch.nn.parameter import Parameter
from torch.distributions.kl import kl_divergence
from torch import zeros_like, ones_like, full_like
from torch.distributions import Normal, MultivariateNormal


def allclose(x, y, tol=1e-5):
    return torch.allclose(x, y, atol=tol, rtol=tol)


def eq_dist(x, y):
    if type(x) != type(y):
        return False
    if x.batch_shape != y.batch_shape:
        return False
    return (kl_divergence(x, y) < 1e-8).all()


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

        assert eq_dist(nl.weight_prior, Normal(0, 1))
        assert eq_dist(nl.bias_prior, Normal(0, 1))
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

        assert eq_dist(fonl.weight_prior, Normal(0, 1))
        assert eq_dist(fonl.bias_prior, Normal(0, 1))
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


def test_MultivariateNormalLinear(get_MultivariateNormalLinear):
    for example in get_MultivariateNormalLinear:
        i, o, b = example
        mnl = MultivariateNormalLinear(*example)

        wp = MultivariateNormal(torch.zeros(o, i),
                                torch.eye(i).repeat(o, 1, 1))
        bp = None if not b else MultivariateNormal(
            torch.zeros(o), torch.eye(o))

        assert eq_dist(mnl.weight_prior, wp)
        if b:
            assert eq_dist(mnl.bias_prior, bp)

        assert isinstance(mnl.weight, WeightMultivariateNormal)
        assert mnl.weight.shape == (o, i)
        assert hasattr(mnl, 'sample')
        assert hasattr(mnl, 'sampled')
        assert isinstance(mnl.sampled, tuple)
        assert len(mnl.sampled) == 2

        if b:
            assert mnl.bias.shape == (o,)
        else:
            assert mnl.bias is None

        init.constant_(mnl.weight.mean, 1)
        init.constant_(mnl.weight.scale, -100)
        with torch.no_grad():
            mnl.weight.scale *= torch.tril(torch.ones_like(mnl.weight.scale))

        if b:
            init.constant_(mnl.bias.mean, 3)
            init.constant_(mnl.bias.scale, -100)
            with torch.no_grad():
                mnl.bias.scale *= torch.tril(torch.ones_like(mnl.bias.scale))
        mnl.sample()

        x = ones_like(mnl.weight.mean)
        result = mnl(x)

        if b:
            assert allclose(result, full_like(result, i + 3))
        else:
            assert allclose(result, full_like(result, i))


def test_NormalInverseGaussianLinear(get_NormalInverseGaussianLinear):
    for example in get_NormalInverseGaussianLinear:
        i, o, b = example
        nigl = NormalInverseGaussianLinear(*example)
        assert nigl.weight_prior is None
        assert nigl.bias_prior is None
        assert nigl.linear.weight.shape == (4 * o, i)

        if b:
            assert nigl.linear.bias.shape == (4 * o,)
        else:
            assert nigl.linear.bias is None

        init.constant_(nigl.linear.weight, 1)
        if b:
            init.constant_(nigl.linear.bias, 3)

        assert hasattr(nigl, 'forward')
        x = torch.ones(1, i)
        dist = nigl(x, sample=True)
        gamma, upsilon, alpha, beta = nigl(x)

        assert gamma.shape == (1, o)
        assert upsilon.shape == (1, o)
        assert alpha.shape == (1, o)
        assert beta.shape == (1, o)

        if b:
            mean = torch.tensor(i + 3, dtype=torch.float)
        else:
            mean = torch.tensor(i, dtype=torch.float)

        other = torch.nn.functional.softplus(mean)

        assert allclose(gamma.mean(),
                        mean,
                        tol=1e-5)
        assert allclose(upsilon.mean(),
                        other,
                        tol=1e-5)
        assert allclose(alpha.mean(),
                        1 + other,
                        tol=1e-5)
        assert allclose(beta.mean(),
                        other,
                        tol=1e-5)

        assert eq_dist(dist, Normal(mean.repeat(1, o),
                                    torch.sqrt(1 / other).repeat(1, o)))


def test_MCDropoutLinear(get_MCDropoutLinear):
    for example in get_MCDropoutLinear:
        i, o, b, p, s = example
        mcdl = MCDropoutLinear(i, o, b, p)

        assert mcdl.weight_prior is None
        assert mcdl.bias_prior is None
        assert mcdl.linear.weight.shape == (o, i)

        if b:
            assert mcdl.linear.bias.shape == (o,)
        else:
            assert mcdl.linear.bias is None

        init.constant_(mcdl.linear.weight, 1)
        if b:
            init.constant_(mcdl.linear.bias, 3)

        assert hasattr(mcdl, 'forward')
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
