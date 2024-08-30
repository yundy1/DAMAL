from torch.utils.data import Dataset
import torch
import numpy as np

cuda = True if torch.cuda.is_available() else False

class matDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        idx = index % len(self.Y)
        data = []
        for x in self.X:
            data.append(x[idx])
        x = data
        y = self.Y[idx]


        if cuda:
            for i in range(len(x)):
                x[i] = x[i].cuda()
            y = y.cuda()

        return x, y

    def __len__(self):
        return len(self.Y)