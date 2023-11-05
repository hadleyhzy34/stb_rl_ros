import pdb
import time
import argparse
import rospy
import numpy as np
import random
import os
from std_msgs.msg import Float32MultiArray
import torch
from torch.utils import tensorboard
from env.env_v0 import Env
from util.print_model import print_model
from util.log import TensorboardCallback
from torch.utils.tensorboard import SummaryWriter
import string
from stable_baselines3 import DQN_CQL, DQN
from stable_baselines3.common.logger import configure

def train(args):
    rospy.init_node('turtlebot3_dqn_stage_1')

    #summary writer session name
    res = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    os.mkdir("/home/hadley/Developments/stb_rl_ros/log/"+res)

    print_model(args)

    env = Env(args)

    if args.cql == 'True':
        model = DQN_CQL("MlpPolicy", env,
                        learning_starts=args.learning_starts,
                        batch_size=args.batch_size,
                        tau=args.tau,
                        train_freq=1,
                        target_update_interval=args.target_update_interval,
                        buffer_size=args.replay_buffer_size,
                        stats_window_size=1,
                        exploration_final_eps=args.exploration_final_eps,
                        verbose=1)
    else:
        model = DQN("MlpPolicy", env,
                        learning_starts=args.learning_starts,
                        batch_size=args.batch_size,
                        tau=args.tau,
                        train_freq=1,
                        target_update_interval=args.target_update_interval,
                        buffer_size=args.replay_buffer_size,
                        stats_window_size=1,
                        exploration_final_eps=args.exploration_final_eps,
                        verbose=1)

    # set up logger
    new_logger = configure("log/"+res,["stdout","csv","tensorboard"])
    model.set_logger(new_logger)

    # train the agent and display a progress bar
    model.learn(total_timesteps=args.total_timesteps,
                # progress_bar=True,
                log_interval=1,
                callback=TensorboardCallback())

    model_name = ''
    if args.continuous == 'True':
        model_name = 'con_dqn_'
    else:
        model_name = 'dis_dqn_'
    if args.cql == 'True':
        model_name += 'wcql'
    else:
        model_name += 'ocql'
    model.save(f"weights/{model_name}_{args.total_timesteps}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', type=str, default='tb3')
    parser.add_argument('--state_size', type=int, default=362)
    parser.add_argument('--action_size', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--replay_buffer_size', type=int, default=10_000)
    parser.add_argument('--episode_step', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--rank_update_interval', type=int, default=200)
    parser.add_argument('--learning_starts', type=int, default=10_000)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration_final_eps', type=float, default=0.01)
    parser.add_argument('--target_update_interval', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=100_000)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--cql', type=str, default='False')
    parser.add_argument('--continuous', type = str, default = 'False')

    args = parser.parse_args()
    train(args)
