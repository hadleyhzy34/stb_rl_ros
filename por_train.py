import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from agent.por import Por
from dataloader.dataloader import CustomDataset
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
import time


def train(args):
    torch.autograd.set_detect_anomaly(True)
    data = CustomDataset(device=args.device)

    agent = Por(
        args.state_size,
        args.action_size,
        batch_size=args.batch_size,
        epsilon=0.9,
        epsilon_decay=0.95,
        epsilon_min=0.01,
        lr=1e-3,
        device=torch.device("cuda"),
    )

    # agent.load_state_dict(torch.load(Config.weight_file))
    agent.train()

    train_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True)

    for i in range(args.episodes):
        print(f"current learning rate: {agent.value_optimizer.param_groups[0]['lr']}")

        value_loss = []
        # step_loss = []

        for data in (pbar := tqdm(train_dataloader)):
            state, action, reward, next_state, done, truncated, rx, ry = data

            value = agent.get_value(state)

            vloss = agent.learn_value(value, rx, ry)

            value_loss.append(vloss)

            pbar.set_description(f"value loss: {statistics.fmean(value_loss)}")
            pbar.refresh()

    # plt.plot(np.arange(len(total_all_loss)),total_all_loss,label="total_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_dist_loss,label="dist_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_coll_loss,label="coll_loss")
    # plt.plot(np.arange(len(total_all_loss)),total_angl_loss,label="angle_loss")
    # # plt.plot(np.arange(len(total_all_loss)),total_step_loss,label="step_loss")
    # plt.legend()
    # plt.show()

    # torch.save(agent.state_dict(), Config.weight_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default="tb3")
    parser.add_argument("--state_size", type=int, default=362)
    parser.add_argument("--action_size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--replay_buffer_size", type=int, default=10_000)
    parser.add_argument("--episode_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--rank_update_interval", type=int, default=200)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    args = parser.parse_args()
    train(args)
