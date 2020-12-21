import torch
from model import BCNN
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import transforms
from pytorch_bayesian.nn import Entropy
from torchvision.datasets import CIFAR10


def main():

    # Hyperparameters

    device = 'cpu'

    # Dataset

    transform = transforms.ToTensor()
    test_dataset = CIFAR10(root='./examples/data/',
                           train=False, transform=transform)

    # Model

    model = BCNN(3, 10).to(device)
    model.load_state_dict(
        torch.load('./examples/CIFAR10/cifar10_pretrained.pth',
                   map_location=device))
    model.eval()

    # Metrics

    entropy_function = Entropy(dim=-1)

    # Uncertainty measurements

    n_images = 5
    fig, axes = plt.subplots(n_images, 2)
    random_indices = torch.randint(0, len(test_dataset), (n_images,))

    for axs, index in zip(axes, random_indices):
        image, _ = test_dataset[index]
        image = image.to(device).unsqueeze(0)
        transformation = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ColorJitter(0, .1, .1, .3)])
        transformed = transformation(image[0])
        transformed = transforms.ToTensor()(transformed).unsqueeze(0)

        for ax, name, x in zip(axs, ['original', 'transformed'], [image, transformed]):
            preds = model(x)

            agg_preds = torch.stack(preds, dim=0).mean(dim=0)
            logits = agg_preds.argmax(dim=-1).item()
            entropy = entropy_function(agg_preds).item()

            ax.axis('off')
            ax.imshow(x.cpu().numpy()[0, ...].transpose(1, 2, 0))
            ax.set_title(
                f'{name}, E: {entropy:.3f}, C: {logits}')

    plt.show()


if __name__ == '__main__':
    main()
