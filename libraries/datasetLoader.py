
# pytorch imports
import torch
from torch.utils.data import Dataset

import pandas as pd

testingDataPercentage = 0.7
dtype = torch.float

class CustomDataset(Dataset):
    def __init__(self, data_path, train):
        data = pd.read_csv(data_path, header=None)
        self.train = train
        self.train_tensor = torch.tensor(data.values, dtype=dtype)
        self.len = self.train_tensor.size(dim = 0)
        self.test_idx = int(self.len * testingDataPercentage)

    def __len__(self):
        if(self.train):
            return self.test_idx
        else:
            return self.len - self.test_idx

    def __getitem__(self, idx):
        if(self.train):
            return self.train_tensor[idx,:-1], int(self.train_tensor[idx,-1])
        else:
            return self.train_tensor[idx + self.test_idx,:-1], int(self.train_tensor[idx+self.test_idx,-1])