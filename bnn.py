import torch
from torchsummary import summary
from types import SimpleNamespace


class NormalWeight(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalWeight, self).__init__()

        self.mu = torch.nn.Parameter(torch.Tensor(
            in_channels, out_channels).uniform_(-0.1, 0.1))

        self.rho = torch.nn.Parameter(torch.Tensor(
            in_channels, out_channels).uniform_(-3, 0))

    def forward(self):
        eps = torch.empty_like(self.mu).normal_()
        # std = 1 + torch.nn.functional.elu(self.rho)
        std = 1e-6 + torch.nn.functional.softplus(self.rho)
        return self.mu + std * eps


class BayesianDense(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BayesianDense, self).__init__()

        self.W = NormalWeight(in_channels, out_channels)
        self.b = NormalWeight(1, out_channels)

    def forward(self, x):
        W = self.W()
        b = self.b()
        return torch.mm(x, W) + b


class BayesianNeuralNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BayesianNeuralNetwork, self).__init__()

        self.layers = torch.nn.Sequential(
            BayesianDense(in_channels, 256),
            BayesianDense(256, 256),
            BayesianDense(256, out_channels),
            torch.nn.Softmax()
        )

    def summary(self, *args, **kwargs):
        summary(self, *args, **kwargs)

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    model = BayesianNeuralNetwork(784, 10)
    for _ in range(10):
        print(torch.argmax(model(torch.zeros(1, 784))).detach().numpy())
    # summary(model, (784,), device='cpu')
    # todo summary not working
