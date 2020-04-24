import torch
from bnn.nn import Entropy
from examples.mnist import BCNN
from torchsummary import summary
from bnn.prune import PruneNormal
from torchvision import transforms
from torchvision.datasets import MNIST


def main():

    # Hyperparameters

    batch_size = 1024
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset

    transform = transforms.ToTensor()
    test_dataset = MNIST(root='./examples/data/',
                         train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Model

    model = BCNN(1, 10).to(device)
    model.load_state_dict(torch.load('./examples/pretrained/mnist_small.pth'))
    model.eval()
    summary(model, (1, 28, 28))

    # Uncertainty measurements

    pass


if __name__ == '__main__':
    main()
