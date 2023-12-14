import rospy
import random
import csv
import argparse
import numpy as np
import pdb
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn

# from env.env import Env
from env.con_env import Env
import string
import pandas as pd


def collect(args):
    rospy.init_node(args.namespace)

    EPISODES = args.episodes

    env = Env(args.state_size, args.action_size)

    action_bound = np.array([0.15 / 2, 1.5])

    file_index = 0
    line_count = 0

    for _ in range(EPISODES):
        done = False
        state, _ = env.gc_reset()

        for _ in range(args.episode_step):
            action = (np.random.random((2,)) * 2 + np.array([0, -1.0])) * action_bound
            # pdb.set_trace()

            assert (
                action[0] >= 0 and action[0] <= 0.15
            ), f"linear velocity is not in range: {action[0]}"
            assert (
                action[1] >= -1.5 and action[1] <= 1.5
            ), f"angular velocity is not in range: {action[1]}"

            # execute actions and wait until next scan(state)
            next_state, reward, done, truncated, info = env.gc_collect_step(action)

            # pdb.set_trace()
            srsda = np.concatenate([state, [reward], next_state, [done], action])

            if not done:
                with open(f"checkpoint/dataset_{file_index}.csv", "a") as f:
                    np.savetxt(f, srsda, fmt="%1.4f", delimiter=",", newline="\n")
                    line_count += 1

            # set next state as next step's current state
            state = next_state

            # reset rank number randomly
            env.rank = random.randint(0, 15)

            # write data to a new file after 100 samples
            if line_count == 100:
                line_count = 0
                file_index += 1

            if done or truncated:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default="tb3")
    parser.add_argument("--state_size", type=int, default=256)
    parser.add_argument("--action_size", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--replay_buffer_size", type=int, default=5000)
    parser.add_argument("--episode_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--update_step", type=int, default=500)
    args = parser.parse_args()
    collect(args)
