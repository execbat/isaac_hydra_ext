import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, n_obs, n_actions,
                 log_std_init=-0.5, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 256), nn.ELU(),
            nn.Linear(256, 256),   nn.ELU(),
        )
        self.mu_head = nn.Linear(256, n_actions)
        self.log_std_head = nn.Linear(256, n_actions)      # голова для log_std
        self.log_std_bias = nn.Parameter(torch.ones(n_actions) * log_std_init)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mu_head.bias, 0.0)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)                              # [B, act_dim]
        log_std = self.log_std_bias + self.log_std_head(x)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()                               # [B, act_dim]
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ELU	(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)
