import os
import json
import pdb
import numpy as np
import random
import time
import sys
import math
import torch
import torch.nn as nn
from typing import Tuple, List
from models.pointmlp import Backbone
from models.policy import GaussianPolicy
from preprocess.dijkstra import Dijkstra
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR


def wloss(
    act_distri, state_value, next_state_value, actions, device=torch.device("cpu")
):
    # pdb.set_trace()
    # actions = (actions + torch.tensor([1.0, 0.0], device=device)) * torch.tensor(
    #     [0.15 / 2, 1.5], device=device
    # )
    # pdb.set_trace()
    assert actions.max() <= 1.0 and actions.min() >= -1.0, "actions range not correct"
    mu = next_state_value - state_value
    loss_positive = mu * act_distri.log_prob(actions)[:, None]
    loss_negative = mu * torch.log(
        1.0 - torch.exp(act_distri.log_prob(actions)[:, None])
    )
    # log_probs = act_distri.log_prob(actions)[:, None]  # (b,1)
    loss = -(mu > 0).float() * loss_positive - (mu < 0).float() * loss_negative

    return torch.mean(loss)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Por(nn.Module):
    def __init__(
        self,
        state_size,
        action_size,
        episode_step=20,
        tau=0.05,
        batch_size=64,
        epsilon=0.9,
        epsilon_decay=0.95,
        epsilon_min=0.01,
        lr=1e-3,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.hidden_dim = 512

        self.backbone = Backbone(
            hidden_dim=self.hidden_dim,
            embed_dim=64,
            groups=1,
            res_expansion=1.0,
            activation="relu",
            bias=False,
            use_xyz=False,
            normalize="anchor",
            dim_expansion=[2, 2, 2],
            pre_blocks=[2, 2, 2],
            pos_blocks=[2, 2, 2],
            k_neighbors=[16, 16, 16],
            reducers=[2, 2, 2],
        ).to(device)

        # value function
        self.value = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        ).to(device)

        self.policy = GaussianPolicy(action_size, self.backbone, self.hidden_dim).to(
            device
        )

        # maximum steps per episode
        self.episode_step = episode_step
        self.tau = tau
        self.gamma = 0.99
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.device = device
        print(f"agent training device is loaded: {self.device}")

        self.lr = lr
        self.value_optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.value.parameters()), self.lr
        )
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), self.lr)
        # show expert path planning animation
        self.show_animation = False

        # loss
        # lambda_optimizer = lambda epoch: 0.5 * (
        #     1.0 + math.cos(math.pi * epoch / episode_step)
        # )
        #
        # self.value_lr_schedule = torch.optim.lr_scheduler.LambdaLR(
        #     self.value_optimizer, lr_lambda=lambda_optimizer
        # )
        self.value_lr_schedule = CosineAnnealingLR(
            self.value_optimizer, T_max=episode_step, eta_min=0
        )

        self.policy_lr_schedule = CosineAnnealingLR(
            self.policy_optimizer, T_max=episode_step, eta_min=0
        )

        self.loss = nn.MSELoss()
        self.value_loss = nn.L1Loss()
        # self.value_loss = nn.MSELoss()
        self.cql = True

    def train_state_preprocess(
        self, state: torch.Tensor, goal: torch.Tensor
    ) -> torch.Tensor:
        """
        args:
            state: torch.float (b,360,2)
            goal: torch.float (b,2)
        return:
            preprocessed_state: torch.float, (b,3,n)
        """
        batch = state.shape[0]
        state_full = torch.zeros((batch, 361, 3), device=self.device)

        state_full[:, :360, :2] = state
        state_full[:, -1, :2] = goal
        state_full[:, -1, 2] = 1.0

        return state_full.permute(0, 2, 1).contiguous()

    def state_preprocess(self, state: torch.Tensor) -> torch.Tensor:
        """
        args:
            state: torch.float,(1,366),(scan+position+heading+goal)
        return:
            preprocessed_state: torch.float, (b,c,n)
        """
        # pdb.set_trace()
        # print(f"original: {state[0,360:362]},target:{state[0,364:]}")
        idx = state[0, :360] < 4.0
        xy = torch.empty((360, 3), device=self.device)[idx, :]

        x = (
            torch.cos(torch.arange(0, 360).to(self.device) * torch.pi / 180)
            * state[0, :360]
        )
        y = (
            torch.sin(torch.arange(0, 360).to(self.device) * torch.pi / 180)
            * state[0, :360]
        )

        xy[:, 0] = x[idx]
        xy[:, 1] = y[idx]
        xy[:, 2] = 0.0

        goal = torch.empty((3,), device=self.device)
        goal[0:2] = state[0, 364:]
        goal[2] = 1.0

        # concat scan and goal
        preprocessed_state = torch.cat([xy, goal[None, :]], dim=-2)

        return preprocessed_state.unsqueeze(0).permute(0, 2, 1)

    def expert_path(self, state: torch.Tensor) -> Tuple[List, List, bool]:
        """
        args:
            state: torch.float,(1,366),(scan+position+heading+goal)
        return:
            rx: x axes of path
            ry: y axes of path
        """
        # pdb.set_trace()
        idx = state[0, :360] < 4.0
        xy = torch.empty((360, 3), device=self.device)[idx, :]

        x = (
            torch.cos(torch.arange(0, 360).to(self.device) * torch.pi / 180)
            * state[0, :360]
        )
        y = (
            torch.sin(torch.arange(0, 360).to(self.device) * torch.pi / 180)
            * state[0, :360]
        )

        xy[:, 0] = x[idx]
        xy[:, 1] = y[idx]
        xy[:, 2] = 0.0

        # pdb.set_trace()
        ox = []
        oy = []
        for i in range(xy.shape[0]):
            ox.append(xy[i][0].item())
            oy.append(xy[i][1].item())

        # pdb.set_trace()
        goal_pose = state[0, 364:].cpu().numpy()

        if self.show_animation:  # pragma: no cover
            plt.plot(ox, oy, ".k")
            plt.plot(0.0, 0.0, "og")
            # plt.plot(goal_pose[0], goal_pose[1], "xb")
            plt.plot(goal_pose[0], goal_pose[1], "xb")
            plt.grid(True)
            plt.axis("equal")

        dijkstra = Dijkstra(ox, oy, 0.1, 0.13)
        rx, ry, state_check = dijkstra.planning(
            0.0, 0.0, goal_pose[0], goal_pose[1], ox, oy, self.show_animation
        )

        if self.show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.01)
            plt.show()

        return rx, ry, state_check

    def learn_act(self, state, goal, next_state, next_state_goal, actions):
        # pdb.set_trace()
        state = self.train_state_preprocess(state, goal)
        next_state = self.train_state_preprocess(next_state, next_state_goal)

        state_value = self.get_value(state).detach()
        next_state_value = self.get_value(next_state).detach()

        act_distri = self.policy(state)

        w_loss = wloss(act_distri, state_value, next_state_value, actions, self.device)
        self.policy_optimizer.zero_grad()

        w_loss.backward()

        self.policy_optimizer.step()

        return w_loss

    def choose_action(self, state) -> Tuple[torch.Tensor, torch.Tensor]:
        """return actions based on sampled from policy distributions
        args:
            state: np.ndarray, (366,)
        return:
            action: torch.tensor, (2,)
            action_logprob: torch.tensor, (2,)
        """
        # pdb.set_trace()
        state = torch.from_numpy(state).to(self.device)[None, :]
        state = self.state_preprocess(state)

        # val = self.get_value(state)
        # print(f"current value is: {val}")
        dist = self.policy(state)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: torch.tensor, float, (b,3,n)
        Returns:
            value: (b,1)
        """
        # b,c,n
        hidden_state = self.backbone(state)
        value = self.value(hidden_state)

        return value

    def learn_value(
        self, state: torch.Tensor, goal: torch.Tensor, path_len: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            state: torch.tensor, (b,360,2)
            goal: torch.tensor, (b,2)
            path_len: torch.tensor, (b,), int, cpu
        Returns:
            vloss: (1,) float
        """
        state = self.train_state_preprocess(state, goal)
        # (b,1)
        value = self.get_value(state)

        # exponential based value function
        # pdb.set_trace()
        base = 100.0
        # expert_value = base * torch.pow(0.99, path_len[:, None].to(self.device))

        # step value based function
        expert_value = base - path_len[:, None].to(self.device)

        vloss = self.value_loss(value, expert_value)

        self.value_optimizer.zero_grad()

        vloss.backward()

        self.value_optimizer.step()

        return vloss

    def cql_loss(self, q_values, current_action):
        """
        Description: Computes the CQL loss for a batch of Q-values and actions.
        args:
            q_values: (b,a)
            current_actions: (b,1)
        return:
            loss: float
        """
        logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
        q_a = q_values.gather(1, current_action)

        return (logsumexp - q_a).mean()

    # def choose_action(self, x):
    #     """
    #     Description:
    #     args:
    #         x: numpy, (state_size,)n
    #     return:
    #     """
    #     # import ipdb;ipdb.set_trace()
    #     x = torch.tensor(x, dtype=torch.float, device=self.device).unsqueeze(
    #         0
    #     )  # (1, state_size)
    #     if np.random.uniform() <= self.epsilon:
    #         return np.random.randint(0, self.action_size)
    #     else:
    #         q_value = self.eval_net(x).detach()
    #         action = q_value.max(1)[1].item()
    #         return action

    def store_transition(self, state, action, reward, next_state, done):
        # import ipdb;ipdb.set_trace()
        state = torch.tensor(state, device=self.device)
        action = torch.tensor(action, device=self.device).unsqueeze(0)  # (1,)
        reward = torch.tensor(reward, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, device=self.device)
        done = torch.tensor(done, device=self.device).unsqueeze(0)
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        # import ipdb;ipdb.set_trace()

        # buffer sampling
        mini_batch = self.memory.sample(self.batch_size)
        states = mini_batch[:, : self.state_size]
        next_states = mini_batch[:, self.state_size + 3 :]
        rewards = mini_batch[:, self.state_size + 1]

        # actions to int
        actions = mini_batch[:, self.state_size].to(dtype=int)
        q_evals = self.eval_net(states)
        q_eval = q_evals.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            q_next = self.tgt_net(next_states).detach()
            q_target = rewards + self.gamma * q_next.max(1)[0]

        loss = self.loss(q_eval, q_target)
        if self.cql:
            loss += self.cql_loss(q_evals, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        soft_update(self.tgt_net, self.eval_net, self.tau)
