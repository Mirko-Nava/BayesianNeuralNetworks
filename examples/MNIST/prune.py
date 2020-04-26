import torch
from model import BCNN
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_bayesian.prune import PruneNormal


def main():

    # Hyperparameters

    batch_size = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset

    transform = transforms.ToTensor()
    test_dataset = MNIST(root='./examples/data/',
                         train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Model

    model = BCNN(1, 10).to(device)
    model.load_state_dict(
        torch.load('./examples/MNIST/mnist_pretrained.pth',
                   map_location=device))
    model.eval()

    # Pruning

    pruner = PruneNormal()

    for drop_percentage in torch.linspace(.75, 1, 6):
        pruner(model, drop_percentage)

        count = 0
        correct = 0
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device), test_y.to(device)

                test_preds = model(test_x)
                test_logits = torch.stack(
                    test_preds, dim=-1).prod(dim=-1).argmax(dim=-1)

                count += len(test_y)
                correct += (test_logits == test_y).to(torch.float).sum()

        test_accuracy = correct / count

        print(f'dropped {100 * drop_percentage:.2f}% of weights',
              f'accuracy: {100 * test_accuracy:.2f}%', sep=', ')

    3 == 3


if __name__ == '__main__':
    main()
