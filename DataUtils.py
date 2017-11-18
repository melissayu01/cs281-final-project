import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class NIDDataset(Dataset):
    def __init__(self, info_csv_file, root_dir, transform=None):
        info = pd.read_csv(info_csv_file, header=None)
        self.batch_size = int(info.iloc[0].values[0])
        self.n = int(info.iloc[1].values[0])
        self.features = info.iloc[2:].values.flatten().tolist()

        self.cached_batch = -1
        self.cached_data = None

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        batch, offset = idx // self.batch_size, idx % self.batch_size
        if batch != self.cached_batch:
            fname = os.path.join(self.root_dir, str(batch) + '.csv')
            self.cached_batch = batch
            self.cached_data = pd.read_csv(fname, header=None).as_matrix()

        sample = self.cached_data[offset]
        if self.transform:
            sample = self.transform(sample)

        return sample[:-1], sample[-1]

class ToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).type(torch.FloatTensor)
