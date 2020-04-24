import torch
from PIL import Image
from model import BCNN
from bnn.nn import Entropy
import matplotlib.pyplot as plt
from torchsummary import summary
from bnn.prune import PruneNormal
from torchvision import transforms
from torchvision.datasets import FashionMNIST


def main():

    # Hyperparameters

    device = 'cpu'

    # Dataset

    transform = transforms.ToTensor()
    test_dataset = FashionMNIST(root='./examples/data/',
                                train=False, transform=transform)

    # Model

    model = BCNN(1, 10).to(device)
    model.load_state_dict(
        torch.load('./examples/FashionMNIST/fmnist_pretrained.pth',
                   map_location=device))
    model.eval()
    summary(model, (1, 28, 28), device=device)

    # Metrics

    entropy_function = Entropy(dim=-1)

    # Uncertainty measurements

    n_images = 5
    fig, axes = plt.subplots(n_images, 2)
    random_indices = torch.randint(0, len(test_dataset), (n_images,))

    for axs, index in zip(axes, random_indices):
        image, _ = test_dataset[index]
        image = image.to(device).unsqueeze(0)
        rotated = transforms.ToPILImage()(image[0]).rotate(
            45, resample=Image.BILINEAR)
        rotated = transforms.ToTensor()(rotated).unsqueeze(0)

        for ax, name, x in zip(axs, ['original', 'rotated'], [image, rotated]):
            preds = model(x)

            agg_preds = torch.stack(preds, dim=0).prod(dim=0)
            logits = agg_preds.argmax(dim=-1).item()
            entropy = entropy_function(agg_preds).item()

            ax.axis('off')
            ax.imshow(x.cpu().numpy()[0, 0, ...])
            ax.set_title(
                f'{name}, E: {entropy:.3f}, C: {logits}')

    plt.show()


if __name__ == '__main__':
    main()
