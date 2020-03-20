import pytest
from bnn.nn import *
from bnn import utils
from torch import Size

# Fixtures for utils


class ComposableBNN(BayesianNetworkModule):

    def __init__(self, in_channels, out_channels, prior, arch):
        super(ComposableBNN, self).__init__(
            in_channels, out_channels, prior, samples=1)

        self.arch = arch

    def _forward(self, x, *args, **kwargs):
        return self.arch(x)


@pytest.fixture
def get_item_or_list():
    return [
        (0,),
        (1,),
        (2, 3),
        [0],
        [1],
        [2, 3]
    ]


@pytest.fixture
def get_single():
    return [
        0,
        1,
        2.3,
        (0,),
        (1,),
    ]


@pytest.fixture
def get_pair():
    return [
        0,
        1,
        2.3,
        (0, 0),
        (1, 1),
        (2, 3)
    ]


@pytest.fixture
def get_triple():
    return [
        0,
        1,
        2.3,
        (0, 0, 0),
        (1, 1, 1),
        (2, 3, 4)
    ]


@pytest.fixture
def get_apply_wb():
    return [
        (torch.nn.Linear(3, 3),
         lambda x, module: None,
         None),
        (torch.nn.Linear(3, 3),
         lambda x, module: x.shape,
         [(3, 3), (3,)]),
        (NormalLinear(3, 3, Normal(0, 1)),
         lambda x, module: x.shape,
         [(3, 3), (3,)]),
        (NormalLinear(3, 3, Normal(0, 1)),
         lambda x, module: type(x),
         [WeightNormal, WeightNormal]),
        (NormalLinear(3, 3, False, Normal(0, 1)),
         lambda x, module: type(x),
         [WeightNormal])
    ]


@pytest.fixture
def get_traverse():
    return [
        (torch.nn.Linear(3, 3),
         lambda x: [x],
         None),
        (BayesianModule(3, 3, Normal(0, 1)),
         lambda x: [type(x.prior)],
         [Normal]),
        (NormalLinear(3, 3, Normal(0, 1)),
         lambda x: [x.bias is not None],
         [True]),
        (NormalLinear(3, 3, False, Normal(0, 1)),
         lambda x: [x.bias is not None],
         [False]),
        (ComposableBNN(3, 4, Normal(0, 1),
                       NormalLinear(3, 4, Normal(0, 1))),
         lambda x: [x.bias is not None],
         [True]),
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Sequential(
                       NormalLinear(3, 4, Normal(0, 1)),
                       NormalLinear(4, 2, Normal(0, 1)),
                       NormalLinear(2, 1, Normal(0, 1)))),
         lambda x: [x.weight.shape],
         [Size([4, 3]), Size([2, 4]), Size([1, 2])])
    ]


# Fixtures for nn


@pytest.fixture
def get_WeightNormal():
    return [
        (1,),
        (3, 4),
        (5, 6, 7)
    ]


@pytest.fixture
def get_BayesianLinear():
    return [
        (1, 1, False, WeightNormal, Normal(0, 1)),
        (3, 4, False, WeightNormal, Normal(0, 1)),
        (1, 1, True, WeightNormal, Normal(0, 1)),
        (3, 4, True, WeightNormal, Normal(0, 1))
    ]


@pytest.fixture
def get_NormalLinear():
    return [
        (1, 1, False, Normal(0, 1)),
        (3, 4, False, Normal(0, 1)),
        (1, 1, True, Normal(0, 1)),
        (3, 4, True, Normal(0, 1)),
        (11, 7, True, Normal(0, 1)),
        (11, 7, False, Normal(0, 1))
    ]


@pytest.fixture
def get_FlipoutNormalLinear():
    return [
        (1, 1, Normal(0, 1)),
        (3, 4, Normal(0, 1)),
        (1, 1, Normal(0, 1)),
        (3, 4, Normal(0, 1)),
        (11, 7, Normal(0, 1)),
        (11, 7, Normal(0, 1))
    ]


@pytest.fixture
def get_BayesianConvNd():
    return [
        (1, 1, (1,), 1, 1, 1, False, 1, False, WeightNormal, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, True, 1, False, WeightNormal, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, False, 1, True, WeightNormal, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, True, 1, True, WeightNormal, Normal(0, 1)),
        (3, 4, (3, 3), 1, 1, 1, False, 1, False, WeightNormal, Normal(0, 1)),
        (3, 4, (3, 3), 1, 1, 1, True, 1, False, WeightNormal, Normal(0, 1)),
        (3, 4, (3, 3), 1, 1, 1, False, 1, True, WeightNormal, Normal(0, 1)),
        (3, 4, (3, 3), 1, 1, 1, True, 1, True, WeightNormal, Normal(0, 1))
    ]


@pytest.fixture
def get_NormalConvNd():
    return [
        (1, 1, (1,), 1, 1, 1, False, 1, False, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, True, 1, False, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, False, 1, True, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, True, 1, True, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, False, 1, False, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, True, 1, False, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, False, 1, True, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, True, 1, True, Normal(0, 1))
    ]


@pytest.fixture
def get_NormalConv1d():
    return [
        (1, 1, 1, 1, 1, 1, 1, False, Normal(0, 1)),
        (1, 1, 1, 1, 1, 1, 1, True, Normal(0, 1)),
        (3, 4, 3, 1, 1, 1, 1, False, Normal(0, 1)),
        (3, 4, 3, 1, 1, 1, 1, True, Normal(0, 1))
    ]


get_NormalConv2d = get_NormalConv1d
get_NormalConv3d = get_NormalConv1d


@pytest.fixture
def get_FlipOutNormalConvNd():
    return [
        (1, 1, (1,), 1, 1, 1, False, 1, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, True, 1, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, False, 1, Normal(0, 1)),
        (1, 1, (1,), 1, 1, 1, True, 1, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, False, 1, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, True, 1, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, False, 1, Normal(0, 1)),
        (3, 4, (3, 3, 7), 1, 1, 1, True, 1, Normal(0, 1))
    ]


@pytest.fixture
def get_FlipOutNormalConv1d():
    return [
        (1, 1, 1, 1, 1, 1, 1, Normal(0, 1)),
        (3, 4, 3, 1, 1, 1, 1, Normal(0, 1))
    ]


get_FlipOutNormalConv2d = get_FlipOutNormalConv1d
get_FlipOutNormalConv3d = get_FlipOutNormalConv1d


@pytest.fixture
def get_KLDivergence():
    return [
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Linear(3, 4)),
         'error'),
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Sequential(
                           torch.nn.Linear(3, 4),
                           torch.nn.Linear(4, 2),
                           torch.nn.Linear(2, 1))),
         'error'),
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Sequential(
                           torch.nn.Linear(3, 4),
                           NormalLinear(4, 2, Normal(0, 1)),
                           torch.nn.Linear(2, 1))),
         'no error')
    ]


# Fixtures for prune


@pytest.fixture
def get_PruneNormal():
    return [
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Sequential(
                           torch.nn.Linear(3, 100),
                           NormalLinear(100, 100, Normal(0, 1)),
                           torch.nn.Linear(100, 4))),
         0.3),
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Sequential(
                           torch.nn.Linear(3, 100),
                           NormalLinear(100, 100, Normal(0, 1)),
                           torch.nn.Linear(100, 4))),
         0.5),
        (ComposableBNN(3, 4, Normal(0, 1),
                       torch.nn.Sequential(
                           torch.nn.Linear(3, 100),
                           NormalLinear(100, 100, Normal(0, 1)),
                           torch.nn.Linear(100, 4))),
         0.8)
    ]
