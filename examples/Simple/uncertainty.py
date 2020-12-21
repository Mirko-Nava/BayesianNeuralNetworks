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
    Aleatoric = []
    Epistemic = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            _, upsilon, alpha, beta = model(x)

            aleatoric, epistemic = uncertainty_function(upsilon, alpha, beta)

            Xs.append(x.cpu().numpy())
            Aleatoric.append(aleatoric.cpu().numpy())
            Epistemic.append(epistemic.cpu().numpy())

    Xs = np.concatenate(Xs)
    Aleatoric = np.concatenate(Aleatoric)
    Epistemic = np.concatenate(Epistemic)

    ax1 = plt.subplot(1, 2, 1)
    plt.title('Aleatoric')
    plt.scatter(Xs, Aleatoric, s=.2, alpha=.3, c='b')
    ax1.set_yscale('log')

    plt.subplot(1, 2, 2, sharey=ax1)
    plt.title('Epistemic')
    plt.scatter(Xs, Epistemic, s=.2, alpha=.3, c='orange')

    plt.show()


if __name__ == '__main__':
    main()
