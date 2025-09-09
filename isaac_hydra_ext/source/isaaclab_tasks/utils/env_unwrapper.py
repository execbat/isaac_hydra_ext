# single_env_unwrapper.py
from __future__ import annotations
import gymnasium as gym
import numpy as np
import torch

class UnbatchedEnv(gym.Env):
    def __init__(self, env: gym.Env, agent_key: str = "policy", index: int = 0):
        super().__init__()
        self.env = env
        self.agent_key = agent_key
        self.index = index
        obs_space = getattr(env, "single_observation_space", env.observation_space)
        act_space = getattr(env, "single_action_space", env.action_space)
        if isinstance(obs_space, gym.spaces.Dict):
            obs_space = obs_space.spaces.get(agent_key, next(iter(obs_space.spaces.values())))
        self.observation_space = gym.spaces.flatten_space(obs_space)
        self.action_space = gym.spaces.flatten_space(act_space)

    def _to_np(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _extract_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs.get(self.agent_key, next(iter(obs.values())))
        obs = self._to_np(obs)
        if obs.ndim >= 2:
            obs = obs[self.index]
        return obs.reshape(-1)

    def _batch_action(self, action):
        action = self._to_np(action).astype(np.float32, copy=False)
        return action[None, :] if action.ndim == 1 else action

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._extract_obs(obs), info

    def step(self, action):
        o, r, d, t, info = self.env.step(self._batch_action(action))
        o = self._extract_obs(o)
        r = float(self._to_np(r).squeeze()[()])
        d = bool(self._to_np(d).squeeze()[()])
        t = bool(self._to_np(t).squeeze()[()])
        return o, r, d, t, info

    def render(self): return self.env.render()
    def close(self):  return self.env.close()
