import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):

    def __init__(self, file_path):
        data = np.load(file_path)
        self.boards = torch.from_numpy(data['boards'])
        self.scores = torch.from_numpy(data['scores']).float()

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        return self.boards[index], self.scores[index]