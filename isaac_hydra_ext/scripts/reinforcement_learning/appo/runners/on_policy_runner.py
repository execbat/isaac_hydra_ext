# appo_multiproc_runner.py
from __future__ import annotations

import os
import io
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import multiprocessing as mp

from typing import List, Tuple, Dict, Any
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Queue, Process

from isaac_hydra_ext.utils import Actor, Critic



def move_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}

def serialize_state_dict(state_dict: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getvalue()

def get_latest_model(folder_path: str, key: str = "actor") -> str | None:
    files = [f for f in os.listdir(folder_path) if key in f and f.endswith(".pt")]
    if not files:
        return None
    return max([os.path.join(folder_path, f) for f in files], key=os.path.getctime)

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    values = torch.cat([values, torch.zeros_like(values[0:1])], dim=0)
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
    returns = torch.zeros_like(deltas)
    gae = 0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        returns[t] = gae + values[t]
    return returns

def combine_batches(all_data: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    states     = torch.cat([batch["states"]     for batch in all_data], dim=0)
    actions    = torch.cat([batch["actions"]    for batch in all_data], dim=0)
    log_probs  = torch.cat([batch["log_probs"]  for batch in all_data], dim=0)
    returns    = torch.cat([batch["returns"]    for batch in all_data], dim=0)
    advantages = torch.cat([batch["advantages"] for batch in all_data], dim=0)
    mus        = torch.cat([batch["mus"]        for batch in all_data], dim=0)
    stds       = torch.cat([batch["stds"]       for batch in all_data], dim=0)
    rewards    = sum([batch["reward_sum"]       for batch in all_data])
    return states, actions, log_probs, returns, advantages, mus, stds, rewards

def collect_samples(envs, actor, critic, gamma: float, lam: float, steps_per_env: int, device: torch.device) -> Dict[str, Any]:
    state, _ = envs.reset()
    states, actions, log_probs, rewards, dones, values, mus, stds = [], [], [], [], [], [], [], []

    for _ in range(steps_per_env):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        if np.any(done):
            if hasattr(envs, "reset_done"):
                reset_obs, _ = envs.reset_done()
                next_state[done] = reset_obs[done]
            else:
                done_indices = np.where(done)[0]
                for idx in done_indices:
                    obs_i, _ = envs.envs[idx].reset()
                    next_state[idx] = obs_i

        value = critic(state_tensor).squeeze(-1)

        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob.detach())
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        dones.append(torch.tensor(done, dtype=torch.float32, device=device))
        values.append(value.detach())
        mus.append(mu.detach())
        stds.append(std.detach())

        state = next_state

    rewards = torch.stack(rewards)
    values = torch.stack(values)
    dones = torch.stack(dones)

    returns = compute_gae(rewards, values, dones, gamma, lam)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "returns": returns,
        "advantages": advantages,
        "mus": torch.stack(mus),
        "stds": torch.stack(stds),
        "reward_sum": rewards.sum().item()
    }

def _worker_collect_and_push(worker_id: int, env_id: str, actor_bytes: bytes, critic_bytes: bytes,
                             steps_per_env: int, gamma: float, lam: float,
                             envs_per_worker: int, queue: Queue, model_queue: Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)
    device = torch.device("cpu")

    import gymnasium as gym
    from isaac_hydra_ext.utils import Actor, Critic
    import isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1  # noqa
    from isaac_hydra_ext.source.isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaac_hydra_ext.scripts.reinforcement_learning.appo.unbatched_env import UnbatchedEnv

    actor_state = torch.load(io.BytesIO(actor_bytes), map_location=device)
    critic_state = torch.load(io.BytesIO(critic_bytes), map_location=device)

    def make_env():
        env_cfg = load_cfg_from_registry(env_id.split(":")[-1], "env_cfg_entry_point")
        if hasattr(env_cfg, "env") and hasattr(env_cfg.env, "num_envs"):
            env_cfg.env.num_envs = 1
        if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "enable"):
            env_cfg.viewer.enable = False
        base_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
        return UnbatchedEnv(base_env, agent_key="policy")

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(envs_per_worker)])

    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))

    actor = Actor(obs_dim, act_dim); actor.load_state_dict(actor_state); actor.eval()
    critic = Critic(obs_dim);      critic.load_state_dict(critic_state); critic.eval()

    while True:
        try:
            new_actor_bytes, new_critic_bytes = model_queue.get_nowait()
            actor.load_state_dict(torch.load(io.BytesIO(new_actor_bytes), map_location=device))
            critic.load_state_dict(torch.load(io.BytesIO(new_critic_bytes), map_location=device))
        except Exception:
            pass

        with torch.no_grad():
            samples = collect_samples(envs, actor.to(device), critic.to(device),
                                      gamma, lam, steps_per_env, device)
        queue.put(samples)


class APPOMultiProcRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        


        self.device = torch.device(device)
        self.cfg = train_cfg
        self.log_dir = log_dir

        p = self.cfg["appo_params"]
        self.gamma = p["gamma"]
        self.lam = p["lam"]
        self.clip_eps_start = p["clip_eps_start"]
        self.clip_eps_max = p["clip_eps_max"]
        self.clip_eps_min = p["clip_eps_min"]
        self.lr = p["lr"]
        self.entropy_coef = p["entropy_coef"]
        self.episodes = self.cfg["max_iterations"]
        self.update_epochs = p["update_epochs"]
        self.batch_size = p["batch_size"]
        self.steps_per_env = p["steps_per_env"]
        self.num_workers = p["num_workers"]
        self.envs_per_worker = p["envs_per_worker"]
        self.kl_treshold = p["kl_treshold"]

        self.save_model_every = self.cfg["save_model_every"]
        self.experiment_name = self.cfg["experiment_name"]
        self.env_name = self.cfg["env_name"]

        
        
        self.obs_dim, self.act_dim = self._resolve_dims_from_env(env)
        

        # models init
        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        # logging
        if self.log_dir is None and "log_params" in self.cfg:
            self.log_dir = self.cfg["log_params"].get("log_dir", None)
        self.writer = SummaryWriter(log_dir=self.log_dir) if self.log_dir else None

        # buffer
        self.reward_history: List[float] = []
        self.episode = 0

        # multiprocessing objects
        self.queue: Queue | None = None
        self.workers: list[Process] = []
        self.model_queues: list[Queue] = []

        self.clip_eps = self.clip_eps_start

        # if need to resume training
        self._maybe_resume()
        print("APPO Runner Created")
        
    def _resolve_dims_from_env(self, env) -> tuple[int, int]:
        if hasattr(env, "single_observation_space") and hasattr(env, "single_action_space"):
            obs_space = env.single_observation_space
            act_space = env.single_action_space
        else:
            obs_space = env.observation_space
            act_space = env.action_space

        if isinstance(obs_space, gym.spaces.Dict):
            obs_space = obs_space.spaces.get("policy", next(iter(obs_space.spaces.values())))
        obs_dim = gym.spaces.flatdim(obs_space)
        act_dim = gym.spaces.flatdim(act_space)
        return int(obs_dim), int(act_dim)    



    def _maybe_resume(self):
        save_dir = os.path.join("experiments", self.experiment_name)
        if os.path.exists(save_dir) and self.cfg.get("continue", False):
            actor_path = get_latest_model(save_dir, "actor")
            critic_path = get_latest_model(save_dir, "critic")
            if actor_path and critic_path:
                self.actor = torch.load(actor_path, map_location=self.device, weights_only=False)
                self.critic = torch.load(critic_path, map_location=self.device, weights_only=False)
                self.actor.train(); self.critic.train()
                print("Models were loaded successfully (resume).")
        elif not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print("Models have been created (new run).")

    # ---------- API ----------
    def learn(self, num_learning_iterations: int | None = None):
    
        episodes = num_learning_iterations if num_learning_iterations is not None else self.episodes

        mp.set_start_method("spawn", force=True)
        print("Number of CPU cores: ", mp.cpu_count())

        # share initial weights into subprocesses
        actor_serialized = serialize_state_dict(move_state_dict_to_cpu(self.actor.state_dict()))
        critic_serialized = serialize_state_dict(move_state_dict_to_cpu(self.critic.state_dict()))

        # launch workers
        self.queue = mp.Queue()
        self.model_queues = [mp.Queue() for _ in range(self.num_workers)]
        self.workers = []
        for i in range(self.num_workers):
            p = mp.Process(
                target=_worker_collect_and_push,
                args=(
                    i, self.env_name, actor_serialized, critic_serialized,
                    self.steps_per_env, self.gamma, self.lam,
                    self.envs_per_worker, self.queue, self.model_queues[i]
                ),
            )
            p.daemon = False
            p.start()
            self.workers.append(p)

        # training
        try:
            print("Training started")
            while self.episode < episodes:
                # collect batches from workers
                all_data = []
                for _ in range(self.num_workers):
                    batch = self.queue.get()
                    all_data.append(batch)

                states, actions, old_log_probs, returns, advantages, mus, stds, rewards = combine_batches(all_data)
                device = self.device
                states = states.to(device)
                actions = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                mus = mus.to(device)
                stds = stds.to(device)

                self.reward_history.append(rewards)

                # updates
                for _ in range(self.update_epochs):
                    idx = torch.randperm(states.size(0))
                    for start in range(0, states.size(0), self.batch_size):
                        end = start + self.batch_size
                        b_idx = idx[start:end]

                        batch_states = states[b_idx]
                        batch_actions = actions[b_idx]
                        batch_old_log_probs = old_log_probs[b_idx]
                        batch_returns = returns[b_idx]
                        batch_advantages = advantages[b_idx]
                        batch_mus = mus[b_idx]
                        batch_stds = stds[b_idx]

                        mu, std = self.actor(batch_states)
                        dist = torch.distributions.Normal(mu, std)
                        new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()

                        with torch.no_grad():
                            old_dist = torch.distributions.Normal(batch_mus, batch_stds)
                            kl_div = torch.distributions.kl_divergence(old_dist, dist).sum(dim=-1).mean()

                        # adapted eps_clip
                        if kl_div > self.kl_treshold * 1.5:
                            self.clip_eps = max(self.clip_eps * 0.9, self.clip_eps_min)
                        if kl_div < self.kl_treshold * 0.5:
                            self.clip_eps = min(self.clip_eps * 1.1, self.clip_eps_max)

                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                        critic_loss = nn.MSELoss()(self.critic(batch_states).squeeze(-1), batch_returns)

                        self.actor_optim.zero_grad()
                        actor_loss.backward()
                        self.actor_optim.step()

                        self.critic_optim.zero_grad()
                        critic_loss.backward()
                        self.critic_optim.step()

                # logs
                if self.writer:
                    self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.episode)
                    self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.episode)
                    self.writer.add_scalar("Metrics/KL_Div", kl_div.item(), self.episode)
                    self.writer.add_scalar("Metrics/Entropy", entropy.item(), self.episode)
                    self.writer.add_scalar("Metrics/Clip_eps", self.clip_eps, self.episode)
                    if (self.episode + 1) % 10 == 0:
                        avg_reward = np.mean(self.reward_history[-10:])
                        self.writer.add_scalar("Rewards/Avg_Reward_10", avg_reward, self.episode)
                        print(f"Episode {self.episode+1}: Avg reward = {avg_reward:.2f}")

                # checkpoints
                if (self.episode + 1) % self.save_model_every == 0:
                    save_path = os.path.join("experiments", self.experiment_name)
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(self.actor, os.path.join(save_path, f"actor_{self.episode+1}.pt"))
                    torch.save(self.critic, os.path.join(save_path, f"critic_{self.episode+1}.pt"))
                    print(f"Saved models at episode {self.episode+1}")

                self.episode += 1

                # send updated weights to workers
                actor_serialized = serialize_state_dict(move_state_dict_to_cpu(self.actor.state_dict()))
                critic_serialized = serialize_state_dict(move_state_dict_to_cpu(self.critic.state_dict()))
                for q in self.model_queues:
                    q.put((actor_serialized, critic_serialized))

            print("Training finished")

        finally:
            for p in self.workers:
                p.terminate()
            if self.queue is not None:
                self.queue.close()

