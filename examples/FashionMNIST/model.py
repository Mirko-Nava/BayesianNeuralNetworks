import torch
from torch.nn import Conv2d, BatchNorm2d, ELU, Softmax
from bnn.nn import BayesianNetworkModule, NormalConv2d, NormalLinear


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class BCNN(BayesianNetworkModule):

    def __init__(self, in_channels, out_channels, samples=10):
        super(BCNN, self).__init__(in_channels, out_channels, samples)

        self.layers = torch.nn.Sequential(
            Conv2d(in_channels, 32, 5, padding=2, stride=2),
            BatchNorm2d(32),
            ELU(),
            NormalConv2d(32, 32, 3, padding=1, stride=1),
            ELU(),
            Conv2d(32, 64, 3, padding=0, stride=2),
            ELU(),
            Conv2d(64, 64, 3, padding=1, stride=2),
            ELU(),
            Flatten(),
            NormalLinear(576, out_channels),
            Softmax(dim=-1)
        )

    def _forward(self, x):
        return self.layers(x)
