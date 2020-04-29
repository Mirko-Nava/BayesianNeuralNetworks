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


def test_BayesianModule():
    bm = BayesianModule(3, 5, Normal(0, 1))

    assert bm.in_channels == 3
    assert bm.out_channels == 5
    assert eq_dist(bm.prior, Normal(0, 1))


def test_BayesianNetworkModule():
    bnm = BayesianNetworkModule(3, 5, 10)

    assert bnm.in_channels == 3
    assert bnm.out_channels == 5
    assert bnm.samples == 10

    with pytest.raises(NotImplementedError):
        bnm.forward(torch.zeros(2, 3, 5))


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


def test_BayesianConvNd(get_BayesianConvNd):
    for example in get_BayesianConvNd:
        i, o, k, s, pad, d, t, g, b, w, p = example
        bcnd = BayesianConvNd(*example)

        assert bcnd.kernel_size == k
        assert bcnd.stride == s
        assert bcnd.padding == pad
        assert bcnd.dilation == d
        assert bcnd.transposed == t
        assert bcnd.groups == g
        assert isinstance(bcnd.weight, w)

        if t:
            assert bcnd.weight.shape == (i, o // g, *k)
        else:
            assert bcnd.weight.shape == (o, i // g, *k)

        if b:
            assert bcnd.bias.shape == (o,)
        else:
            assert bcnd.bias is None


def test_NormalConvNd(get_NormalConvNd):
    for example in get_NormalConvNd:
        i, o, k, s, pad, d, t, g, b, p = example
        ncnd = NormalConvNd(*example)

        assert isinstance(ncnd.weight, WeightNormal)
        assert hasattr(ncnd, 'sample')
        assert hasattr(ncnd, 'sampled')
        assert isinstance(ncnd.sampled, tuple)
        assert len(ncnd.sampled) == 2

        if t:
            assert ncnd.weight.shape == (i, o // g, *k)
        else:
            assert ncnd.weight.shape == (o, i // g, *k)

        if b:
            assert isinstance(ncnd.bias, WeightNormal)
            assert ncnd.bias.shape == (o,)
            assert isinstance(ncnd.sampled[1], torch.Tensor)
        else:
            assert ncnd.bias is None
            assert ncnd.sampled[1] is None


def test_NormalConv1d(get_NormalConv1d):
    for example in get_NormalConv1d:
        i, o, k, s, pad, d, g, b, p = example
        nc1d = NormalConv1d(*example)

        assert eq_dist(nc1d.prior, Normal(0, 1))
        init.constant_(nc1d.weight.mean, 1)
        init.constant_(nc1d.weight.scale, -100)
        if b:
            init.constant_(nc1d.bias.mean, 3)
            init.constant_(nc1d.bias.scale, -100)
        nc1d.sample()

        x = torch.ones(1, i, 10)
        result = nc1d(x)
        expected = torch.nn.functional.conv1d(
            x, ones_like(nc1d.weight.mean), None,
            s, pad, d, g)

        if b:
            assert allclose(result, expected + 3)
        else:
            assert allclose(result, expected)


def test_NormalConv2d(get_NormalConv2d):
    for example in get_NormalConv2d:
        i, o, k, s, pad, d, g, b, p = example
        nc2d = NormalConv2d(*example)

        assert eq_dist(nc2d.prior, Normal(0, 1))
        init.constant_(nc2d.weight.mean, 1)
        init.constant_(nc2d.weight.scale, -100)
        if b:
            init.constant_(nc2d.bias.mean, 3)
            init.constant_(nc2d.bias.scale, -100)
        nc2d.sample()

        x = torch.ones(1, i, 10, 10)
        result = nc2d(x)
        expected = torch.nn.functional.conv2d(
            x, ones_like(nc2d.weight.mean), None,
            s, pad, d, g)

        if b:
            assert allclose(result, expected + 3)
        else:
            assert allclose(result, expected)


def test_NormalConv3d(get_NormalConv3d):
    for example in get_NormalConv3d:
        i, o, k, s, pad, d, g, b, p = example
        nc3d = NormalConv3d(*example)

        assert eq_dist(nc3d.prior, Normal(0, 1))
        init.constant_(nc3d.weight.mean, 1)
        init.constant_(nc3d.weight.scale, -100)
        if b:
            init.constant_(nc3d.bias.mean, 3)
            init.constant_(nc3d.bias.scale, -100)
        nc3d.sample()

        x = torch.ones(1, i, 10, 10, 10)
        result = nc3d(x)
        expected = torch.nn.functional.conv3d(
            x, ones_like(nc3d.weight.mean), None,
            s, pad, d, g)

        if b:
            assert allclose(result, expected + 3)
        else:
            assert allclose(result, expected)


def test_FlipOutNormalConvNd(get_FlipOutNormalConvNd):
    for example in get_FlipOutNormalConvNd:
        i, o, k, s, pad, d, t, g, p = example
        foncnd = FlipOutNormalConvNd(*example)

        assert isinstance(foncnd.weight, WeightNormal)
        assert hasattr(foncnd, 'sample')
        assert hasattr(foncnd, 'sampled')
        assert isinstance(foncnd.sampled, tuple)
        assert len(foncnd.sampled) == 2


def test_FlipOutNormalConv1d(get_FlipOutNormalConv1d):
    for example in get_FlipOutNormalConv1d:
        i, o, k, s, pad, d, g, p = example
        fonc1d = FlipOutNormalConv1d(*example)

        assert eq_dist(fonc1d.prior, Normal(0, 1))
        init.constant_(fonc1d.weight.mean, 1)
        init.constant_(fonc1d.weight.scale, -100)
        fonc1d.sample()

        x = torch.ones(7, i, 10)
        result = fonc1d(x)
        expected = torch.nn.functional.conv1d(
            x, ones_like(fonc1d.weight.mean), None,
            s, pad, d, g)
        assert allclose(result, expected)


def test_FlipOutNormalConv2d(get_FlipOutNormalConv2d):
    for example in get_FlipOutNormalConv2d:
        i, o, k, s, pad, d, g, p = example
        fonc2d = FlipOutNormalConv2d(*example)

        assert eq_dist(fonc2d.prior, Normal(0, 1))
        init.constant_(fonc2d.weight.mean, 1)
        init.constant_(fonc2d.weight.scale, -100)
        fonc2d.sample()

        x = torch.ones(7, i, 10, 10)
        result = fonc2d(x)
        expected = torch.nn.functional.conv2d(
            x, ones_like(fonc2d.weight.mean), None,
            s, pad, d, g)
        assert allclose(result, expected)


def test_FlipOutNormalConv3d(get_FlipOutNormalConv3d):
    for example in get_FlipOutNormalConv3d:
        i, o, k, s, pad, d, g, p = example
        fonc3d = FlipOutNormalConv3d(*example)

        assert eq_dist(fonc3d.prior, Normal(0, 1))
        init.constant_(fonc3d.weight.mean, 1)
        init.constant_(fonc3d.weight.scale, -100)
        fonc3d.sample()

        x = torch.ones(7, i, 10, 10, 10)
        result = fonc3d(x)
        expected = torch.nn.functional.conv3d(
            x, ones_like(fonc3d.weight.mean), None,
            s, pad, d, g)
        assert allclose(result, expected)


def test_MCDropoutConvNd(get_MCDropoutConvNd):
    for example in get_MCDropoutConvNd:
        i, o, p = example
        mcdcnd = MCDropoutConvNd(*example)

        assert mcdcnd.prior is None
        assert mcdcnd.in_channels == i
        assert mcdcnd.out_channels == o


def test_MCDropoutConv1d(get_MCDropoutConv1d):
    for example in get_MCDropoutConv1d:
        i, o, k, s, pad, d, g, b, p = example
        mcdc1d = MCDropoutConv1d(*example)

        assert mcdc1d.conv.weight.shape == (o, i // g, k)

        if b:
            assert mcdc1d.conv.bias.shape == (o,)
        else:
            assert mcdc1d.conv.bias is None

        init.constant_(mcdc1d.conv.weight, 1)
        if b:
            init.constant_(mcdc1d.conv.bias, 3)

        x = torch.ones(1, i, 100)
        result = mcdc1d(x)
        expected = torch.nn.functional.conv1d(
            x, ones_like(mcdc1d.conv.weight), None,
            s, pad, d, g)

        if b:
            assert allclose(result.mean(),
                            expected.mean() + 3,
                            tol=1e-1)
        else:
            assert allclose(result.mean(),
                            expected.mean(),
                            tol=1e-1)


def test_MCDropoutConv2d(get_MCDropoutConv2d):
    for example in get_MCDropoutConv2d:
        i, o, k, s, pad, d, g, b, p = example
        mcdc2d = MCDropoutConv2d(*example)

        assert mcdc2d.conv.weight.shape == (o, i // g, k, k)

        if b:
            assert mcdc2d.conv.bias.shape == (o,)
        else:
            assert mcdc2d.conv.bias is None

        init.constant_(mcdc2d.conv.weight, 1)
        if b:
            init.constant_(mcdc2d.conv.bias, 3)

        x = torch.ones(1, i, 100, 100)
        result = mcdc2d(x)
        expected = torch.nn.functional.conv2d(
            x, ones_like(mcdc2d.conv.weight), None,
            s, pad, d, g)

        if b:
            assert allclose(result.mean(),
                            expected.mean() + 3,
                            tol=1e-1)
        else:
            assert allclose(result.mean(),
                            expected.mean(),
                            tol=1e-1)


def test_MCDropoutConv3d(get_MCDropoutConv3d):
    for example in get_MCDropoutConv3d:
        i, o, k, s, pad, d, g, b, p = example
        mcdc3d = MCDropoutConv3d(*example)

        assert mcdc3d.conv.weight.shape == (o, i // g, k, k, k)

        if b:
            assert mcdc3d.conv.bias.shape == (o,)
        else:
            assert mcdc3d.conv.bias is None

        init.constant_(mcdc3d.conv.weight, 1)
        if b:
            init.constant_(mcdc3d.conv.bias, 3)

        x = torch.ones(1, i, 100, 10, 10)
        result = mcdc3d(x)
        expected = torch.nn.functional.conv3d(
            x, ones_like(mcdc3d.conv.weight), None,
            s, pad, d, g)

        if b:
            assert allclose(result.mean(),
                            expected.mean() + 3,
                            tol=1e-1)
        else:
            assert allclose(result.mean(),
                            expected.mean(),
                            tol=1e-1)


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
