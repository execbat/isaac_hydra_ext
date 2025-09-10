# appo_multiproc_runner.py (patched)
from __future__ import annotations

import os
import io
import time
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


# ---------- utils ----------

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

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float) -> torch.Tensor:
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


def collect_samples(envs, actor, critic, gamma: float, lam: float,
                    steps_per_env: int, device: torch.device) -> Dict[str, Any]:
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
                try:
                    reset_obs, _ = envs.reset_done()
                    next_state[done] = reset_obs[done]
                except Exception:
                    # fallback: entire reset
                    state, _ = envs.reset()
                    next_state = state
            else:
                state, _ = envs.reset()
                next_state = state

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
        "reward_sum": rewards.sum().item(),
    }


# ---------- worker ----------

def _worker_collect_and_push(worker_id: int, env_id: str, actor_bytes: bytes, critic_bytes: bytes,
                             steps_per_env: int, gamma: float, lam: float,
                             envs_per_worker: int, queue: Queue, model_queue: Queue) -> None:
    # --- headless + CPU-only for workers ---
    os.environ["IS_WORKER"] = "1"
    os.environ["KIT_WINDOWMODE"] = "headless"
    os.environ["OMNI_KIT_WINDOW_FLAGS"] = "headless"
    os.environ["PYTHON_NO_USD_RENDER"] = "1"
    os.environ.setdefault("DISPLAY", "")

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCH_CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PHYSX_DISABLE_GPU"] = "1"   

    torch.set_num_threads(1)
    device = torch.device("cpu")

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=True,
        device="cpu",                             
        experience="isaaclab.python.headless.kit" 
    )
    simulation_app = app_launcher.app
    print(f"[WORKER {worker_id}] Headless Kit up (CPU)", flush=True)

    import gymnasium as gym
    from isaac_hydra_ext.utils import Actor, Critic
    import isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1  # noqa: F401
    from isaac_hydra_ext.source.isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from isaac_hydra_ext.source.isaaclab_tasks.utils import UnbatchedEnv

    actor_state  = torch.load(io.BytesIO(actor_bytes),  map_location=device, weights_only=True)
    critic_state = torch.load(io.BytesIO(critic_bytes), map_location=device, weights_only=True)

    def make_env():
        env_key = env_id.split(":")[-1]
        env_cfg = load_cfg_from_registry(env_key, "env_cfg_entry_point")

        # ---- force CPU pipeline in workers ----
        if hasattr(env_cfg, "sim"):
            try:
                env_cfg.sim.device = "cpu"
            except Exception:
                pass
            if hasattr(env_cfg.sim, "use_gpu"):
                env_cfg.sim.use_gpu = False
            if hasattr(env_cfg.sim, "use_gpu_pipeline"):
                env_cfg.sim.use_gpu_pipeline = False

        # все N под-сред создаёт сама Isaac-сцена
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = int(envs_per_worker)

        
        if hasattr(env_cfg, "viewer") and hasattr(env_cfg.viewer, "enable"):
            env_cfg.viewer.enable = False

        
        if hasattr(env_cfg, "seed") and (env_cfg.seed is None or env_cfg.seed == 0):
            env_cfg.seed = 1000 + int(worker_id)

        base_env = gym.make(env_id, cfg=env_cfg, render_mode=None)
        return base_env

    
    envs = make_env()


    obs_space = getattr(envs, "observation_space")
    act_space = getattr(envs, "action_space")
    if isinstance(obs_space, gym.spaces.Dict):
        obs_space = obs_space.spaces.get("policy", next(iter(obs_space.spaces.values())))
    if isinstance(act_space, gym.spaces.Dict):
        act_space = act_space.spaces.get("policy", next(iter(act_space.spaces.values())))
    obs_dim = int(gym.spaces.flatdim(obs_space))
    act_dim = int(gym.spaces.flatdim(act_space))

    actor = Actor(obs_dim, act_dim)
    actor.load_state_dict(torch.load(io.BytesIO(actor_bytes), map_location=device, weights_only=True))
    actor.eval()
    critic = Critic(obs_dim)
    critic.load_state_dict(torch.load(io.BytesIO(critic_bytes), map_location=device, weights_only=True))
    critic.eval()


    try:
        while True:
            try:
                new_actor_bytes, new_critic_bytes = model_queue.get_nowait()
                actor.load_state_dict(torch.load(io.BytesIO(new_actor_bytes),  map_location=device, weights_only=True))
                critic.load_state_dict(torch.load(io.BytesIO(new_critic_bytes), map_location=device, weights_only=True))
            except Exception:
                pass

            with torch.no_grad():
                samples = collect_samples(envs, actor.to(device), critic.to(device),
                                          gamma, lam, steps_per_env, device)
            queue.put(samples)
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


# ---------- runner ----------

class APPOMultiProcRunner:
    def __init__(self, env_name, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        self.device = device
        self.cfg = train_cfg

        # === log roots ===
        # prefer dir from train(); otherwise fallback/default
        self.run_root_dir = log_dir
        if self.run_root_dir is None and "log_params" in self.cfg:
            self.run_root_dir = self.cfg["log_params"].get("log_dir", None)
        self.experiment_name = self.cfg["experiment_name"]
        if self.run_root_dir is None:
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.run_root_dir = os.path.join("experiments", self.experiment_name, ts)
        os.makedirs(self.run_root_dir, exist_ok=True)

        # TB under run_root/tb
        self.tb_dir = os.path.join(self.run_root_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        print(f"[TB] TensorBoard logs -> {os.path.abspath(self.tb_dir)}")

        # checkpoints under run_root/checkpoints
        self.ckpt_dir = os.path.join(self.run_root_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        print(f"[CKPT] Checkpoints -> {os.path.abspath(self.ckpt_dir)}")

        # === training params ===
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
        self.env_name = self.cfg["env_name"]
        self.save_model_every = self.cfg["log_params"]["save_model_every"]

        self.obs_dim, self.act_dim = self.cfg["models"]["obs_dim"], self.cfg["models"]["act_dim"]
        print('SIZES', self.obs_dim, self.act_dim)

        self.actor = Actor(self.obs_dim, self.act_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.reward_history: List[float] = []
        self.episode = 0

        self.queue: Queue | None = None
        self.workers: list[Process] = []
        self.model_queues: list[Queue] = []

        self.clip_eps = self.clip_eps_start

        self._maybe_resume()
        print(f"APPO Runner Created on {self.device}")

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
        # look for ckpts in current run_root/checkpoints
        if self.cfg.get("continue", False):
            actor_path = get_latest_model(self.ckpt_dir, "actor")
            critic_path = get_latest_model(self.ckpt_dir, "critic")
            if actor_path and critic_path:
                actor_sd  = torch.load(actor_path,  map_location=self.device, weights_only=True)
                critic_sd = torch.load(critic_path, map_location=self.device, weights_only=True)
                self.actor.load_state_dict(actor_sd)
                self.critic.load_state_dict(critic_sd)
                self.actor.train(); self.critic.train()
                print("Models were loaded successfully (resume).")
        else:
            print("Starting a new run.")

    # ---------- API ----------
    def learn(self, num_learning_iterations: int | None = None):
        episodes = num_learning_iterations if num_learning_iterations is not None else self.episodes

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        print("Number of CPU cores:", mp.cpu_count())

        actor_serialized = serialize_state_dict(move_state_dict_to_cpu(self.actor.state_dict()))
        critic_serialized = serialize_state_dict(move_state_dict_to_cpu(self.critic.state_dict()))

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
                daemon=False,
            )
            p.start()
            self.workers.append(p)

        try:
            print("Training started")
            while self.episode < episodes:
                all_data = [self.queue.get() for _ in range(self.num_workers)]

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

                for _ in range(self.update_epochs):
                    idx = torch.randperm(states.size(0), device=device)
                    for start in range(0, states.size(0), self.batch_size):
                        b_idx = idx[start:start + self.batch_size]

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

                if self.writer:
                    self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.episode)
                    self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.episode)
                    self.writer.add_scalar("Metrics/KL_Div", kl_div.item(), self.episode)
                    self.writer.add_scalar("Metrics/Entropy", entropy.item(), self.episode)
                    self.writer.add_scalar("Metrics/Clip_eps", self.clip_eps, self.episode)
                    if (self.episode + 1) % 10 == 0:
                        avg_reward = float(np.mean(self.reward_history[-10:]))
                        self.writer.add_scalar("Rewards/Avg_Reward_10", avg_reward, self.episode)
                        print(f"Episode {self.episode + 1}: Avg reward = {avg_reward:.2f}")

                # save to run_root/checkpoints
                if (self.episode + 1) % self.save_model_every == 0:
                    actor_path = os.path.join(self.ckpt_dir, f"actor_{self.episode + 1}.pt")
                    critic_path = os.path.join(self.ckpt_dir, f"critic_{self.episode + 1}.pt")
                    torch.save(self.actor.state_dict(),  os.path.join(self.ckpt_dir, f"actor_{self.episode+1}.pt"))
                    torch.save(self.critic.state_dict(), os.path.join(self.ckpt_dir, f"critic_{self.episode+1}.pt"))
                    print(f"[CKPT] Saved: {actor_path} | {critic_path}")

                self.episode += 1

                # broadcast updated weights
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
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

