import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class NIDDataset(Dataset):
    def __init__(self, info_csv_file, root_dir, shuffle, transform=None):
        info = pd.read_csv(info_csv_file, header=None)
        self.batch_size = int(info.iloc[0].values[0])
        self.n = int(info.iloc[1].values[0])
        self.features = info.iloc[2:].values.flatten().tolist()

        self.shuffle = shuffle
        self.cached_batch = -1
        self.cached_data = None

        self.root_dir = root_dir
        self.transform = transform

        self.seed = np.random.randint(0, 10, 10000)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        batch, offset = idx // self.batch_size, idx % self.batch_size
        if batch != self.cached_batch:
            fname = os.path.join(self.root_dir, str(batch) + '.csv')
            df = pd.read_csv(fname, header=None)
            if self.shuffle:
                df = df.sample(frac=1, random_state=self.seed[batch])
            self.cached_batch = batch
            self.cached_data = df.as_matrix()

        sample = self.cached_data[offset]
        if self.transform:
            sample = self.transform(sample)

        return sample[:-1], sample[-1]

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).float()

def load_data(batch_size, cuda, data_dir='../data/', shuffle=True):
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_dataset = NIDDataset(
        info_csv_file=data_dir + 'train/info.csv',
        root_dir=data_dir + 'train/',
        shuffle=shuffle,
        transform=ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False, **kwargs
    )
    test_dataset = NIDDataset(
        info_csv_file=data_dir + 'test/info.csv',
        root_dir=data_dir + 'test/',
        shuffle=shuffle,
        transform=ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False, **kwargs
    )
    n_features = len(train_dataset.features) - 1

    return n_features, train_loader, test_loader
