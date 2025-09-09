import os
import time
import yaml
import torch
import argparse

import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import custom_envs

from utils import Actor, Critic


def get_latest_actor_file(folder_path: str) -> str | None:
    """Find the latest actor model checkpoint in the given folder."""
    files = [f for f in os.listdir(folder_path) if "actor" in f and f.endswith(".pt")]
    files = [os.path.join(folder_path, f) for f in files if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        return None
    return max(files, key=os.path.getctime)


if __name__ == "__main__":
    # Load config
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ppo_params = config['ppo_params']
    log_params = config['log_params']
    env_name = config['env_name']
    checkpoint_params = config['checkpoint_params']
    experiment_name = config['experiment_name']
    print("Config loaded successfully.")

    # Find latest actor model
    load_dir = os.path.join("experiments", experiment_name)
    last_model = get_latest_actor_file(load_dir)

    if last_model is None:
        raise FileNotFoundError(f"No actor model found in {load_dir}")
    print(f"Loading model: {last_model}")

    # Load actor model
    actor = torch.load(last_model, map_location='cpu', weights_only = False)
    actor.eval()  # Optional: uncomment if you want strict eval mode

    # Create environment
    env = gym.make(env_name, max_episode_steps=2000, render=True, spawn_ball_every=150)
    print("Gymnasium environment created.")

    state, _ = env.reset()

    # Play with the trained agent
    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            mu, std = actor(state_tensor.unsqueeze(0))
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()

        state, reward, terminated, truncated, _ = env.step(action.squeeze(0).detach().numpy())
        #print("reward", reward)
        if terminated or truncated:
            state, _ = env.reset()

        time.sleep(0.01)

