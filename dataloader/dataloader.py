import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pdb

class CustomDataset(Dataset):
    def __init__(self, device=torch.device('cpu')):
        self.device = device

        list = os.listdir('checkpoint/')
        self.length = len(list) * 100

    def __getitem__(self, index):
        file_index = index // 100
        row_index = index % 100
        data = torch.load(f'checkpoint/dataset_{file_index}.pt',
                          map_location=self.device)[row_index]

        return data[index]

    def __len__(self):
        return self.length
