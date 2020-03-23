import torch
from tqdm import tqdm
from statistics import mean
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.nn import Conv2d, BatchNorm2d, ELU, Softmax
from bnn.nn import BayesianNetworkModule, NormalConv2d, NormalLinear, KLDivergence, Entropy


# Hyperparameters


epochs = 10
batch_size = 1024
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Dataset


transform = transforms.ToTensor()
train_dataset = MNIST(root='./examples/data/',
                      train=True, transform=transform)
test_dataset = MNIST(root='./examples/data/',
                     train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)


# Model


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


model = BCNN(1, 10).to(device)
summary(model, (1, 28, 28))


# Loss, Metrics and Optimizer


kld_function = KLDivergence(number_of_batches=len(train_loader))
loss_function = torch.nn.CrossEntropyLoss()
entropy_function = Entropy(dim=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=1, cooldown=1, threshold=0.05, verbose=True)


# Training


epochs_logger = tqdm(range(1, epochs + 1), desc='epoch')
for epoch in epochs_logger:
    running_acc = []
    running_ent = []
    running_loss = []
    steps_logger = tqdm(train_loader, total=len(train_loader), desc='step')
    for x, y in steps_logger:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        preds = model(x)
        divergence = kld_function(model)
        likelihood = torch.stack([
            loss_function(pred, y) for pred in preds
        ]).mean() / model.samples

        loss = likelihood + divergence
        loss.backward()
        optimizer.step()

        agg_preds = torch.stack(preds, dim=0).prod(dim=0)
        logits = agg_preds.argmax(dim=-1)
        accuracy = (logits == y).float().mean()
        entropy = entropy_function(agg_preds)

        running_acc.append(accuracy.item())
        running_ent.append(entropy.item())
        running_loss.append(loss.item())

        log_str = f'L: {mean(running_loss):.4f}, A: {mean(running_acc):.4f}, E: {mean(running_ent):.4f}'
        steps_logger.set_postfix_str(log_str)

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

    scheduler.step(mean(running_loss))

    log_str += f', TA: {test_accuracy:.4f}'
    epochs_logger.set_postfix_str(log_str)
