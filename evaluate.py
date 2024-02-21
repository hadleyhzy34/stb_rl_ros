import torch
from functools import total_ordering
import gymnasium as gym
import rospy
import os
import pdb
import argparse
import statistics
from env.env_eval import Env
from agent.por import Por


def test(args):
    rospy.init_node("tb3_stage")
    # pdb.set_trace()

    env = Env(args.state_size, args.action_size)

    # pdb.set_trace()
    agent = Por(
        args.state_size,
        args.action_size,
        device=torch.device(args.device),
    )
    agent.eval()

    # pdb.set_trace()
    loaded_model = torch.load("weights/policy/policy15.pt")
    agent.load_state_dict(loaded_model)

    scores, success = [], []
    obs, info = env.reset()
    score = 0.0
    episode = 0

    while True:
        action = agent.predict(obs).cpu().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)

        score += reward

        if terminated or truncated:
            if info["status"] == "goal":
                success.append(1.0)
            else:
                success.append(0.0)
            scores.append(score)
            score = 0.0
            obs, info = env.reset()

            print(
                f"episodes: {episode+1}||"
                f"score: {statistics.fmean(scores)}||"
                f"success: {statistics.fmean(success)}"
            )

            episode += 1
            if episode >= args.episodes:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", type=str, default="tb3")
    parser.add_argument("--state_size", type=int, default=366)
    parser.add_argument("--action_size", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--replay_buffer_size", type=int, default=10_000)
    parser.add_argument("--episode_step", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--rank_update_interval", type=int, default=200)
    parser.add_argument("--learning_starts", type=int, default=10_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target_update_interval", type=int, default=1)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--mode", type=str, default="test")
    parser.add_argument("--cql", type=str, default="False")
    parser.add_argument("--continuous", type=str, default="False")
    args = parser.parse_args()
    test(args)
