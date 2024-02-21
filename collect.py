import rospy
import random
import csv
import argparse
import json
import numpy as np
import pdb
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn

# from env.env import Env
from dataloader.type import SarsaModel
from env.env_tb3 import Env
from agent.por import Por


def collect(args):
    # pdb.set_trace()
    rospy.init_node(args.namespace)

    agent = Por(
        args.state_size,
        args.action_size,
        batch_size=args.batch_size,
        epsilon=0.9,
        epsilon_decay=0.95,
        epsilon_min=0.01,
        lr=1e-3,
        device=torch.device(args.device),
    )

    EPISODES = args.episodes

    env = Env(args.state_size, args.action_size)

    file_idx = 0
    step_data = []

    for _ in range(EPISODES):
        done = False
        state, _ = env.reset()

        for _ in range(args.episode_step):
            action = np.random.random(2) * 2 - 1.0  # [-1,1)

            # execute actions and wait until next scan(state)
            next_state, reward, done, truncated, info = env.step(action)

            if not done:
                # check if goal pose too closed to obstacles
                if not agent.validate_state(state):
                    # print("state value is not correct")
                    continue

                sarsa_data = SarsaModel(
                    state=state.tolist(),
                    reward=reward,
                    next_state=next_state.tolist(),
                    done=done,
                    truncated=truncated,
                    action=action.tolist(),
                )

                # pydantic model to json data
                sarsa_json = sarsa_data.model_dump_json()
                step_data.append(sarsa_json)

            # set next state as next step's current state
            state = next_state

            # write data to a new file after 100 samples
            if len(step_data) == 100:
                with open(f"checkpoint/raw/dataset_{file_idx}.json", "w") as f:
                    f.write(json.dumps(step_data))

                print(f"file {file_idx} saved")
                # # check
                # pdb.set_trace()
                # with open(f"checkpoint/dataset_{file_idx}.json", "r") as f:
                #     json_object = json.load(f)

                file_idx += 1
                step_data = []

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
