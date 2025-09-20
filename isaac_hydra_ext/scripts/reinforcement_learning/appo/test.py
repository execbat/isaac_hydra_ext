#!/usr/bin/env python3
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
"""
Test runner to play a trained APPO checkpoint in Isaac Lab with a visible viewer.

Usage (as requested):
  TEST:
    ./isaaclab.sh -p -m isaac_hydra_ext.scripts.reinforcement_learning.appo.test \
      --task Isaac-Velocity-Sber-Unitree-Go1-Play-v0

Optional flags:
  --ckpt-root logs/ppo_run       # search recursively for */checkpoints/actor_*.pt
  --ckpt /path/to/actor_*.pt     # pin a specific checkpoint (overrides search)
  --num_envs 1                   # number of envs to instantiate (viewer-friendly: 1)
  --deterministic                # use mean action (mu) instead of sampling
  --video --video-length 1000    # record a short video with gym's RecordVideo
  --real-time                    # sleep to match env step_dt (if available)
"""

import os
import sys
import time
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser("Play a trained APPO policy (Isaac Lab viewer)")
parser.add_argument("--task", type=str, default=None, help="Isaac-Lab Gym env ID (e.g., Isaac-Velocity-...-v0)")

parser.add_argument("--ckpt-root", type=str, default="logs/ppo_run",
                    help="Root dir to search recursively for */checkpoints/actor_*.pt")
parser.add_argument("--ckpt", type=str, default=None,
                    help="Explicit path to actor checkpoint .pt (overrides --ckpt-root search)")

parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of envs to instantiate (>=1). Viewer is most useful with 1.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed")
parser.add_argument("--max-steps", type=int, default=4000, help="Max simulation steps")
parser.add_argument("--deterministic", action="store_true", help="Use mean action (mu) instead of sampling")
parser.add_argument("--video", action="store_true", help="Record a video (rgb_array mode)")
parser.add_argument("--video-length", type=int, default=1000, help="Length of recorded video in steps")
parser.add_argument("--real-time", action="store_true", help="Sleep to approximate real-time using env dt")

# Append AppLauncher CLI args (viewer/headless, experience, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.task is None:
    print("[ERR] --task is required (e.g., --task Isaac-Velocity-Sber-Unitree-Go1-Play-v0)")
    sys.exit(2)

# Always enable cameras when recording
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Lab app
# -----------------------------------------------------------------------------

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[APP] Isaac Lab viewer started.")

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def find_latest_actor_checkpoint(ckpt_root: str) -> Optional[str]:
    """Find newest actor_*.pt under */checkpoints recursively; fallback to any actor*.pt."""
    if not os.path.exists(ckpt_root):
        return None
    candidates = []
    for root, _, files in os.walk(ckpt_root):
        if "checkpoints" in root:
            for f in files:
                if f.startswith("actor") and f.endswith(".pt"):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        for root, _, files in os.walk(ckpt_root):
            for f in files:
                if f.startswith("actor") and f.endswith(".pt"):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        return None
    return max(candidates, key=os.path.getctime)

def _flatdim(space) -> int:
    """Flattened dimension of a gym space."""
    try:
        return int(gym.spaces.flatdim(space))
    except Exception:
        shape = getattr(space, "shape", None)
        if shape is None:
            return 1
        n = 1
        for s in shape:
            n *= int(s)
        return int(n)

def resolve_obs_act_dims(env) -> Tuple[int, int]:
    """Resolve per-env observation and action dims (handles Dict('policy', ...))."""
    obs_space = env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        obs_space = obs_space.spaces.get("policy", next(iter(obs_space.spaces.values())))
    act_space = env.action_space
    if isinstance(act_space, gym.spaces.Dict):
        act_space = act_space.spaces.get("policy", next(iter(act_space.spaces.values())))
    return _flatdim(obs_space), _flatdim(act_space)

def policy_slice(x):
    """If observation/action is a dict with 'policy' key, return that tensor/ndarray."""
    if isinstance(x, dict) and "policy" in x:
        return x["policy"].squeeze(0)
    return x

def get_action_bounds(env):
    """Return (low, high) as float32 tensors, handling Dict('policy', Box)."""
    space = env.action_space
    if isinstance(space, gym.spaces.Dict):
        space = space.spaces.get("policy", next(iter(space.spaces.values())))
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return None, None
    return torch.as_tensor(low, dtype=torch.float32), torch.as_tensor(high, dtype=torch.float32)
    
def to2d_tensor(x, device):
  
    if isinstance(x, dict) and "policy" in x:
        x = x["policy"]

    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=FORCE_DTYPE, device=device)
    else:
        x = x.to(device=device, dtype=FORCE_DTYPE)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x   
    

# -----------------------------------------------------------------------------
# Build env from registry (enable viewer, set num_envs)
# -----------------------------------------------------------------------------


# Ensure task registry is loaded
import isaac_hydra_ext.source.isaaclab_tasks  # noqa: F401
import isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1
from isaac_hydra_ext.source.isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    

env_key = args_cli.task.split(":")[-1]
env_cfg = load_cfg_from_registry(env_key, "env_cfg_entry_point")
    

if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "enable"):
    env_cfg.viewer.enable = True
if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
    env_cfg.scene.num_envs = int(max(1, args_cli.num_envs))

if torch.cuda.is_available() and hasattr(env_cfg, "sim"):
    try:
        env_cfg.sim.device = "cuda:0"
        if hasattr(env_cfg.sim, "use_gpu"):
            env_cfg.sim.use_gpu = True
        if hasattr(env_cfg.sim, "use_gpu_pipeline"):
            env_cfg.sim.use_gpu_pipeline = True
    except Exception:
        pass

    render_mode = "rgb_array" if args_cli.video else "human"
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
#except Exception as e:
#    print(f"[WARN] Failed to build env from registry ({e}). Falling back to gym.make defaults.")
#    render_mode = "rgb_array" if args_cli.video else "human"
#    env = gym.make(args_cli.task, render_mode=render_mode)

# If recording video, wrap env
if args_cli.video:
    video_dir = os.path.join("logs", "videos", "eval")
    os.makedirs(video_dir, exist_ok=True)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=int(args_cli.video_length),
        disable_logger=True,
    )
    print(f"[VIDEO] Recording to: {os.path.abspath(video_dir)} (length={args_cli.video_length} steps)")
    
 

# -----------------------------------------------------------------------------
# Load Actor and checkpoint
# -----------------------------------------------------------------------------

ckpt_path = args_cli.ckpt or find_latest_actor_checkpoint(args_cli.ckpt_root)
if ckpt_path is None:
    print(f"[ERR] No actor checkpoint found under: {args_cli.ckpt_root}")
    env.close()
    simulation_app.close()
    sys.exit(1)
print(f"[CKPT] Using: {ckpt_path}")

obs_dim, act_dim = resolve_obs_act_dims(env)
print(f"[ENV] obs_dim={obs_dim}  act_dim={act_dim} (per env)")

if args_cli.device == "auto":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(args_cli.device)

from isaac_hydra_ext.utils import Actor
actor = Actor(obs_dim, act_dim).to(device)
sd = torch.load(ckpt_path, map_location=device, weights_only=True)
# be tolerant to small mismatches (e.g., extra buffers)
missing, unexpected = actor.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(f"[WARN] State dict mismatch. missing={missing}, unexpected={unexpected}")
actor.eval()
print(f"[MODEL] Actor loaded on {device}")

low_t, high_t = get_action_bounds(env)
if low_t is not None and high_t is not None:
    low_t = low_t.to(device)
    high_t = high_t.to(device)

# -----------------------------------------------------------------------------
# Rollout loop
# -----------------------------------------------------------------------------
dt = env.unwrapped.step_dt
print(f'dt: {dt}')
print(f'args_cli.max_steps: {args_cli.max_steps}')


FORCE_DTYPE = torch.float32

# --- reset
obs, _ = env.reset()
obs_t = to2d_tensor(obs, device)  # [1, obs_dim], float32

steps = 0
print("[EVAL] Running with viewer... (Ctrl+C to stop)")

while simulation_app.is_running():
    t0 = time.time()
    with torch.no_grad():
        mu, std = actor(obs_t.float())              # [1, 12] float32

        dist   = torch.distributions.Normal(mu.float(), std.float())
        action = dist.sample().to(dtype=FORCE_DTYPE)  # [1, 12] float32

        # sanity-check раз в 50 шагов
        if steps % 50 == 0:
            print(f"[DBG] obs {tuple(obs_t.shape)} {obs_t.dtype} | "
                  f"mu {tuple(mu.shape)} {mu.dtype} | "
                  f"act {tuple(action.shape)} {action.dtype}")

    action_to_env = action
    #print(f'Action {action_to_env}')

    next_obs, reward, terminated, truncated, info = env.step(action_to_env)
    #print(f'Observation {next_obs}')
    print(f'reward {reward}')

    term = torch.as_tensor(terminated, dtype=torch.bool, device=device)
    trunc = torch.as_tensor(truncated,  dtype=torch.bool, device=device)
    done  = term | trunc
    if done.any().item():
        print("Done detected:", done.nonzero().squeeze(-1).tolist())

    obs_t = to2d_tensor(next_obs, device)           # [1, obs_dim] float32

    if isinstance(dt, (int, float)) and dt > 0:
        sleep_t = dt - (time.time() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)

    steps += 1
    if args_cli.video and steps >= args_cli.video_length:
        break



try:
    env.close()
except Exception:
    pass
try:
    simulation_app.close()
except Exception:
    pass
print("[CLEANUP] Closed Isaac Lab.")
