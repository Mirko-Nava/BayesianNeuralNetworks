import os
import torch
import urllib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class Titanic(torch.utils.data.Dataset):

    filename = 'Titanic/titanic.csv'
    resource = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'

    def __init__(self, root, train, download=False, transform=None):
        self.root = root
        self.transform = transform

        if download:
            self.download()

        if not self.check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.df = pd.read_csv(os.path.join(self.root,
                                           self.filename))

        self.df['Sex'] = self.df['Sex'].apply(
            lambda x: {'male': [0, 1], 'female': [1, 0]}[x])
        self.df['Pclass'] = self.df['Pclass'].apply(
            lambda x: {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}[x])

        if train:
            self.df = self.df.iloc[:600]
        else:
            self.df = self.df.iloc[600:]

        self.x = self.df[['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard',
                          'Parents/Children Aboard', 'Fare']].values

        self.y = torch.tensor(self.df['Survived'].values).long()

    def download(self):
        if self.check_exists():
            return

        if not os.path.exists(self.root):
            os.mkdir(self.root)

        if not os.path.exists(os.path.join(self.root, 'Titanic')):
            os.mkdir(os.path.join(self.root, 'Titanic'))

        fpath = os.path.join(self.root, self.filename)

        print('Downloading ' + self.resource + ' to ' + fpath)
        urllib.request.urlretrieve(self.resource,
                                   fpath)

    def check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.filename))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = (self.x[index], self.y[index])

        x = np.concatenate(list(map(
            lambda x: np.array(x if isinstance(x, list) else [x]),
            x)))
        x = torch.tensor(x).float()

        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)

        return (x, y)
