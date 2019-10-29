import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from nn import NeuralNetwork
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dist_bnn import BayesianNeuralNetwork
from torchvision import datasets, transforms


def train(args, model, device, train_loader, loss_function, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()

        loss = 0
        std = []

        for _ in range(args.samples):
            prediction = model(data)
            std += [prediction.detach().cpu().numpy().argmax(axis=1)]
            loss += loss_function(prediction, target)  # , model)

        std = np.array(std).std(axis=0).mean()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{:5.0f}/{} ({:3.0f}%)]\tLoss: {:.6f}\tStdDev: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), std))


def test(args, model, device, test_loader, loss_function):
    model.eval()
    count = 0
    test_loss = 0
    stds = []
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            std = []
            for _ in range(args.samples):
                prediction = model(data)

                std += [prediction.detach().cpu().numpy().argmax(axis=1)]
                test_loss += torch.nn.functional.cross_entropy(
                    prediction, target)

            stds.append(np.std(std, axis=0).mean())

            pred = prediction.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            count += 1

    stds = np.std(stds)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), StdDev {:.6f}\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), stds))


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: a value)')
    parser.add_argument('--samples', type=int, default=10, metavar='N',
                        help='how many samples to use for estimating the distribution')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True)

    n_batches = len(train_loader.dataset) / args.batch_size

    model = BayesianNeuralNetwork(784, 10).to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader,
              loss_function, optimizer, epoch)
        test(args, model, device, test_loader, loss_function)

    model.to('cpu')
    with torch.no_grad():
        for images, _ in test_loader:
            for im in images[:10]:
                plot_preds(im, model)
                plt.show()
            plot_preds(torch.randn(1, 28, 28), model)
            plt.show()
            break


def plot_preds(im, model):
    preds = [model(im.view(784)).detach().numpy()
             for _ in range(100)]
    plt.subplot(1, 2, 1)
    plt.imshow(np.transpose(im, [1, 2, 0])[:, :, 0], )
    plt.subplot(1, 2, 2)
    plt.hist(np.argmax(preds, axis=1), bins=10)
    plt.ylim(0, 1)


if __name__ == '__main__':
    main()
