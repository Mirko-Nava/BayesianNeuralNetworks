import torch
from torch.nn import ELU, Softmax
from pytorch_bayesian.nn import BayesianNetworkModule, MCDropoutLinear


class BNN(BayesianNetworkModule):

    def __init__(self, in_features, out_features, samples=100):
        super(BNN, self).__init__(in_features, out_features, samples)

        self.layers = torch.nn.Sequential(
            MCDropoutLinear(in_features, 256, drop_prob=.2),
            ELU(),
            MCDropoutLinear(256, out_features, drop_prob=.2),
            Softmax(dim=-1)
        )

    def _forward(self, x):
        return self.layers(x)
