import torch
import torch.nn as nn

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    def __init__(self, act_dim, backbone, hidden_dim=512):
        super().__init__()
        self.backbone = backbone

        # guassian policy
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))

    def forward(self, obs):
        latent_obs = self.backbone(obs).detach()
        mean = self.net(latent_obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        scale_tril = torch.diag(std)
        return torch.distributions.MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self(obs)
            return dist.mean if deterministic else dist.sample()


class DeterministicPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, n_hidden=2):
        super().__init__()
        self.net = mlp(
            [obs_dim, *([hidden_dim] * n_hidden), act_dim], output_activation=nn.Tanh
        )

    def forward(self, obs):
        return self.net(obs)

    def act(self, obs, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(obs)
