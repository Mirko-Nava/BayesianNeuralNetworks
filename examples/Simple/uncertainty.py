import torch
import numpy as np
from model import BNN
import matplotlib.pyplot as plt
from dataset import SimpleDataset
from pytorch_bayesian.nn import NormalInverseGaussianUncertainty


def main():

    # Hyperparameters

    batch_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset

    test_dataset = SimpleDataset(root='./examples/data/',
                                 train=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Metrics

    uncertainty_function = NormalInverseGaussianUncertainty()

    # Model

    model = BNN(1, 1).to(device)
    model.load_state_dict(
        torch.load('./examples/Simple/simple_pretrained.pth',
                   map_location=device))
    model.eval()

    # Uncertainty measurements

    Xs = []
    Mean = []
    Aleatoric = []
    Epistemic = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            gamma, upsilon, alpha, beta = model(x)

            aleatoric, epistemic = uncertainty_function(upsilon, alpha, beta)

            Xs.append(x.cpu().numpy())
            Mean.append(gamma.cpu().numpy())
            Aleatoric.append(aleatoric.cpu().numpy())
            Epistemic.append(epistemic.cpu().numpy())

    Xs = np.concatenate(Xs)
    Mean = np.concatenate(Mean)
    Aleatoric = np.concatenate(Aleatoric)
    Epistemic = np.concatenate(Epistemic)

    ax1 = plt.subplot(1, 2, 1)
    plt.title('Aleatoric')
    for k in np.linspace(0, 3, 4):
        plt.fill_between(Xs[:, 0],
                         (Mean - k * np.sqrt(Aleatoric))[:, 0],
                         (Mean + k * np.sqrt(Aleatoric))[:, 0],
                         alpha=.25, facecolor='orange')
    plt.scatter(Xs, Mean, s=.2, alpha=.05, c='b')

    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    plt.title('Epistemic')
    for k in np.linspace(0, 3, 4):
        plt.fill_between(Xs[:, 0],
                         (Mean - k * np.sqrt(Epistemic))[:, 0],
                         (Mean + k * np.sqrt(Epistemic))[:, 0],
                         alpha=.25, facecolor='orange')
    plt.scatter(Xs, Mean, s=.2, alpha=.05, c='b')

    ax1.set_aspect(1 / ax1.get_data_ratio())
    ax2.set_aspect(1 / ax2.get_data_ratio())

    plt.show()


if __name__ == '__main__':
    main()
