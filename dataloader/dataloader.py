import json
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pdb


class CustomDataset(Dataset):
    def __init__(self, device=torch.device("cpu")):
        self.device = device

        list = os.listdir("checkpoint/preprocessed")
        self.length = len(list) * 100

    def __getitem__(self, index):
        # pdb.set_trace()
        file_idx = index // 100
        row_idx = index % 100

        with open(f"checkpoint/preprocessed/dataset_{file_idx}.json") as f:
            data = json.load(f)

        # str(dict) -> dict
        data = json.loads(data[row_idx])

        state = torch.tensor(data["state"]).to(self.device)
        action = torch.tensor(data["action"]).to(self.device)
        reward = torch.tensor(data["reward"]).to(self.device)
        next_state = torch.tensor(data["next_state"]).to(self.device)
        done = torch.tensor(data["done"]).to(self.device)
        truncated = torch.tensor(data["truncated"]).to(self.device)

        rx = data["state_path_x"]
        ry = data["state_path_y"]

        return (state, action, reward, next_state, done, truncated, rx, ry)

    def __len__(self):
        return self.length
