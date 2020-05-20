import torch
from torch.nn import Module
from ..utils import _item_or_list, traverse


class BayesianModule(Module):

    def __init__(self, in_channels, out_channels, prior, bias_prior=None):
        super(BayesianModule, self).__init__()

        self.weight_prior = prior
        self.bias_prior = bias_prior if bias_prior else prior
        self.in_channels = in_channels
        self.out_channels = out_channels


class BayesianNetworkModule(Module):

    def __init__(self, in_channels, out_channels, samples=10):
        super(BayesianNetworkModule, self).__init__()

        self.samples = samples
        self.in_channels = in_channels
        self.out_channels = out_channels

    def _forward(self, x, *args, **kwargs):
        raise NotImplementedError('self._forward() not implemented')

    def traverse(self, fn, *args, **kwargs):
        return traverse(self, fn, *args, **kwargs)

    def forward(self, x, samples=None, *args, **kwargs):
        if samples is None:
            samples = self.samples

        return _item_or_list([self._forward(x, *args, **kwargs)
                              for _ in range(samples)])
