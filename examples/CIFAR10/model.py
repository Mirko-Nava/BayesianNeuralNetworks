import torch
from torch.nn import Linear, Conv2d, MaxPool2d, BatchNorm2d, ELU, Softmax
from pytorch_bayesian.nn import BayesianNetworkModule, NormalConv2d, MultivariateNormalLinear, NormalLinear


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class BCNN(BayesianNetworkModule):

    def __init__(self, in_channels, out_channels, samples=20):
        super(BCNN, self).__init__(in_channels, out_channels, samples)

        self.layers = torch.nn.Sequential(
            Conv2d(in_channels, 64, 5, padding=2, stride=2),
            BatchNorm2d(64),
            ELU(),
            Conv2d(64, 128, 5, padding=2, stride=2),
            ELU(),
            Conv2d(128, 128, 5, padding=2, stride=2),
            ELU(),
            Conv2d(128, 128, 3, padding=1),
            ELU(),
            Conv2d(128, 128, 3, padding=1),
            ELU(),
            NormalConv2d(128, 128, 3, padding=1),
            ELU(),
            Flatten(),
            Linear(2048, 128),
            ELU(),
            MultivariateNormalLinear(128, out_channels),
            Softmax(dim=-1)
        )

    def _forward(self, x):
        return self.layers(x)
