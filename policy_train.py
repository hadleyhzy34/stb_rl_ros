import os
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from agent.por import Por
from dataloader.dataloader import CustomDataset
import pdb
from tqdm import tqdm
import statistics
from collections import deque
import time
import yaml


def train():
    with open("config/train_value.yml", "r") as f:
        args = yaml.full_load(f)
    # torch.autograd.set_detect_anomaly(True)
    data = CustomDataset(args["data_path"], device=args["device"])
    train_dataloader = DataLoader(data, batch_size=args["batch_size"], shuffle=True)

    # pdb.set_trace()
    agent = Por(
        args["state_size"],
        args["action_size"],
        batch_size=args["batch_size"],
        lr=args["lr"],
        episode_step=args["episodes"],
        device=torch.device(args["device"]),
    )
    agent.train()

    # pdb.set_trace()
    loaded_model = torch.load("weights/value/20.pt")
    agent.load_state_dict(loaded_model)

    for i in range(1, args["episodes"] + 1):
        print(f"current learning rate: {agent.policy_optimizer.param_groups[0]['lr']}")

        value_loss = []
        # value_loss = deque(maxlen=100)

        for data in (pbar := tqdm(train_dataloader)):
            (
                state,
                goal,
                action,
                reward,
                next_state,
                next_state_goal,
                done,
                truncated,
                path_len,
            ) = data

            vloss = agent.learn_act(state, goal, next_state, next_state_goal, action)

            # vloss = agent.learn_value(state, goal, path_len)

            value_loss.append(vloss)
            pbar.set_description(f"policy loss: {statistics.fmean(value_loss)}")
            pbar.refresh()

        if i % args["save_freq"] == 0:
            torch.save(
                agent.state_dict(),
                os.path.join(args["weight_policy_file"], "policy" + str(i) + ".pt"),
            )

        agent.policy_lr_schedule.step()
        # agent.value_lr_schedule.step()


if __name__ == "__main__":
    train()
