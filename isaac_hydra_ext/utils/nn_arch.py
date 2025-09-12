import torch
import torch.nn as nn

'''
class Actor(nn.Module):
    def __init__(self, n_obs, n_actions,
                 log_std_init=-1.2, log_std_min=-4.0, log_std_max=1.0,
                 scale=0.5, squash=True):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 512), nn.ELU(),
            nn.Linear(512, 512),   nn.ELU(),
        )
        self.mu_head = nn.Linear(512, n_actions)
        self.ls_head = nn.Linear(512, n_actions)   
        self.log_std_bias = nn.Parameter(torch.full((n_actions,), log_std_init))
        self.scale = scale 
        self.register_buffer("ls_min", torch.tensor(log_std_min))
        self.register_buffer("ls_max", torch.tensor(log_std_max))
        self.squash = squash

        nn.init.uniform_(self.mu_head.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mu_head.bias, 0.0)
        nn.init.zeros_(self.ls_head.weight)
        nn.init.zeros_(self.ls_head.bias)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        if self.squash:
            mu = torch.tanh(mu)
        
        delta = self.scale * torch.tanh(self.ls_head(x)) 
        raw = self.log_std_bias + delta                   
        log_std = self.ls_min + 0.5*(self.ls_max-self.ls_min)*(torch.tanh(raw)+1.0)
        std = torch.exp(log_std)
        return mu, std
'''

class Actor(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
        )
        self.mu_head = nn.Linear(256, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))  
        
        # init weights and bias
        nn.init.uniform_(self.mu_head.weight, -0.003, 0.003)
        nn.init.constant_(self.mu_head.bias, 0.0)
        
    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        std = self.log_std.exp().expand_as(mu)  
        return mu, std        

class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)
