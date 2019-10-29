import math
import torch
from torchsummary import summary


class NeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NeuralNetwork, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, out_channels),
            torch.nn.Softmax()
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = NeuralNetwork(784, 10)
    summary(model, (784,), device='cpu')
