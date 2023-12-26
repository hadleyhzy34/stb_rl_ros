import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
import pdb


class CustomDataset(Dataset):
    def __init__(self, path, device=torch.device("cpu")):
        self.device = device

        self.path = path
        # pdb.set_trace()
        self.list = os.listdir(path)
        self.length = len(self.list) * 100

    def __getitem__(self, index):
        # pdb.set_trace()
        file_idx = index // 100
        row_idx = index % 100

        with open(os.path.join(self.path, self.list[file_idx])) as f:
            data = json.load(f)

        # str(dict) -> dict
        data = json.loads(data[row_idx])

        state = torch.tensor(data["state"]).to(self.device)
        goal = torch.tensor(data["goal"]).to(self.device)
        action = torch.tensor(data["action"]).to(self.device)
        reward = torch.tensor(data["reward"]).to(self.device)
        next_state = torch.tensor(data["next_state"]).to(self.device)
        next_state_goal = torch.tensor(data["next_state_goal"]).to(self.device)
        done = torch.tensor(data["done"]).to(self.device)
        truncated = torch.tensor(data["truncated"]).to(self.device)

        rx = data["state_path_x"]
        ry = data["state_path_y"]
        path_len = len(rx)

        return (
            state,
            goal,
            action,
            reward,
            next_state,
            next_state_goal,
            done,
            truncated,
            path_len,
            # rx,
            # ry,
        )

    def __len__(self):
        return self.length
