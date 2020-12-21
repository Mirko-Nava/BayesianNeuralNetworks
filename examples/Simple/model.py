import torch
from torch.nn import Linear, ReLU
from pytorch_bayesian.nn import BayesianNetworkModule, NormalInverseGammaLinear


class BNN(BayesianNetworkModule):

    def __init__(self, in_channels, out_channels):
        super(BNN, self).__init__(in_channels, out_channels, samples=1)

        self.layers = torch.nn.Sequential(
            Linear(in_channels, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            Linear(100, 100),
            ReLU(),
            NormalInverseGammaLinear(100, out_channels),
        )

    def _forward(self, x):
        return self.layers(x)
