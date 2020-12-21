import os
import torch
import numpy as np
import pandas as pd


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, root, train, generate=False, transform=None):
        self.root = root + 'Simple/'
        self.transform = transform

        if generate:
            self.generate()

        if not self.check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use generate=True to generate it')

        filename = 'train' if train else 'test'

        self.df = pd.read_csv(os.path.join(self.root,
                                           filename + '.csv'))

        self.x = self.df['X'].values
        self.y = self.df['Y'].values

    def generate(self):
        if self.check_exists():
            return

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        print('Generating data to ' + self.root)

        def f(x): return x ** 3 + np.random.randn(*x.shape) * 3

        for (s, e), fname in zip([(-4, 4), (-8, 8)], ['train', 'test']):
            x = np.linspace(s, e, 10001)
            filename = os.path.join(self.root, fname + '.csv')
            pd.DataFrame(dict(X=x, Y=f(x))).to_csv(filename)

    def check_exists(self):
        return os.path.exists(self.root)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = (self.x[index, None], self.y[index, None])

        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)

        return (x, y)

# ds = SimpleDataset('./examples/data/', train=True, generate=True)
# ds2 = SimpleDataset('./examples/data/', train=False)
# import matplotlib.pyplot as plt

# plt.scatter(ds2.x, ds2.y, c='r', alpha=.2, s=.2)
# plt.scatter(ds.x, ds.y, c='b', alpha=.2, s=.2)
# plt.xlim(-8.5, 8.5)
# plt.ylim(-150, 150)
# plt.show()
