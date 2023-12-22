import argparse
import os
import json
import numpy as np
import torch
from agent.por import Por
from dataloader.dataloader import CustomDataset
import pdb
from matplotlib import pyplot as plt
from tqdm import tqdm
import statistics
import time
from dataloader.type import SarsaModel


def train(args):
    agent = Por(
        args.state_size,
        args.action_size,
        batch_size=args.batch_size,
        epsilon=0.9,
        epsilon_decay=0.95,
        epsilon_min=0.01,
        lr=1e-3,
        device="cpu",
    )

    # pdb.set_trace()
    file_idx = 0
    for file in sorted(os.listdir("./checkpoint/raw")):
        step_data = []

        with open(f"./checkpoint/raw/{file}") as f:
            data = json.load(f)  # list file

            for d in data:
                cur_data = json.loads(d)

                state = torch.tensor(cur_data["state"])
                next_state = torch.tensor(cur_data["next_state"])

                state_path_x, state_path_y, state_check = agent.expert_path(
                    state[None, :]
                )
                (
                    next_state_path_x,
                    next_state_path_y,
                    next_state_check,
                ) = agent.expert_path(next_state[None, :])

                if not state_check or not next_state_check:
                    continue

                sarsa_data = SarsaModel(
                    state=cur_data["state"],
                    reward=cur_data["reward"],
                    next_state=cur_data["next_state"],
                    done=cur_data["done"],
                    truncated=cur_data["truncated"],
                    action=cur_data["action"],
                    state_path_x=state_path_x,
                    state_path_y=state_path_y,
                    next_state_path_x=next_state_path_x,
                    next_state_path_y=next_state_path_y,
                )

                # pydantic model to json data
                sarsa_json = sarsa_data.model_dump_json()
                step_data.append(sarsa_json)

                if len(step_data) == 100:
                    with open(
                        f"checkpoint/preprocessed/dataset_{file_idx}.json", "w"
                    ) as f:
                        f.write(json.dumps(step_data))
                    step_data = []
                    print(f"{file_idx}th file saved")
                    file_idx += 1


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
