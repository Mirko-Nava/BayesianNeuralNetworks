import torch
from tqdm import tqdm
from model import BNN
from statistics import mean
from torchsummary import summary
from dataset import SimpleDataset
from pytorch_bayesian.nn import NormalInverseGaussianLoss


def main():

    # Hyperparameters

    epochs = 100
    batch_size = 128
    learning_rate = 5e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset

    train_dataset = SimpleDataset(root='./examples/data/',
                                  train=True)
    test_dataset = SimpleDataset(root='./examples/data/',
                                 train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Model

    model = BNN(1, 1).to(device)
    summary(model, (1,), device=device)

    # Loss, Metrics and Optimizer

    loss_function = NormalInverseGaussianLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training

    epochs_logger = tqdm(range(1, epochs + 1), desc='epoch')
    for epoch in epochs_logger:
        running_loss = []
        steps_logger = tqdm(train_loader, total=len(train_loader), desc='step')
        for x, y in steps_logger:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            gamma, upsilon, alpha, beta = model(x)

            loss = loss_function(gamma, upsilon, alpha, beta, y)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            log_str = f'L: {mean(running_loss):.4f}'
            steps_logger.set_postfix_str(log_str)

        count = 0
        test_loss = 0
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x, test_y = test_x.to(device), test_y.to(device)

                gamma, upsilon, alpha, beta = model(test_x)

                test_loss += loss_function(gamma, upsilon,
                                           alpha, beta, test_y).item()
                count += 1

        test_loss = test_loss / count

        log_str += f', TL: {test_loss:.4f}'
        epochs_logger.set_postfix_str(log_str)

    torch.save(model.state_dict(), './examples/Simple/simple_pretrained.pth')


if __name__ == '__main__':
    main()
