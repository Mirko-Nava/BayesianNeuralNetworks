import torch
from tqdm import tqdm
from model import BNN
from statistics import mean
from dataset import Titanic
from torchsummary import summary
from pytorch_bayesian.nn import Entropy


def main():

    # Hyperparameters

    epochs = 50
    batch_size = 128
    learning_rate = 5e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset

    train_dataset = Titanic(root='./examples/data/',
                            train=True)
    test_dataset = Titanic(root='./examples/data/',
                           train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Model

    model = BNN(9, 2).to(device)
    summary(model, (9,), device=device)

    # Loss, Metrics and Optimizer

    loss_function = torch.nn.CrossEntropyLoss()
    entropy_function = Entropy(dim=-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
            likelihood = torch.stack([
                loss_function(pred, y) for pred in preds
            ]).mean()

            loss = likelihood
            loss.backward()
            optimizer.step()

            agg_preds = torch.stack(preds, dim=0).mean(dim=0)
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
                    test_preds, dim=-1).mean(dim=-1).argmax(dim=-1)

                count += len(test_y)
                correct += (test_logits == test_y).to(torch.float).sum()

        test_accuracy = correct / count

        log_str += f', TA: {test_accuracy:.4f}'
        epochs_logger.set_postfix_str(log_str)

    torch.save(model.state_dict(), './examples/Titanic/titanic_pretrained.pth')


if __name__ == '__main__':
    main()
