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

#def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
#                gamma: float, lam: float) -> torch.Tensor:
#    values = torch.cat([values, torch.zeros_like(values[0:1])], dim=0)
#    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]
#    returns = torch.zeros_like(deltas)
#    gae = 0
#    for t in reversed(range(len(deltas))):
#        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
#        returns[t] = gae + values[t]
#    return returns

def compute_gae(rewards, values, dones, gamma, lam, last_value):
    # rewards, values, dones: [T, B]
    T, B = rewards.shape
    returns = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        returns[t] = gae + values[t]
    return returns

def combine_batches(all_data):
    states     = torch.cat([d["states"]     for d in all_data], dim=0)  # [W*T*B, D]
    actions    = torch.cat([d["actions"]    for d in all_data], dim=0)  # [W*T*B, A]
    log_probs  = torch.cat([d["log_probs"]  for d in all_data], dim=0)  # [W*T*B]
    returns    = torch.cat([d["returns"]    for d in all_data], dim=0)  # [W*T*B]
    advantages = torch.cat([d["advantages"] for d in all_data], dim=0)  # [W*T*B]
    mus        = torch.cat([d["mus"]        for d in all_data], dim=0)  # [W*T*B, A]
    stds       = torch.cat([d["stds"]       for d in all_data], dim=0)  # [W*T*B, A]
    values     = torch.cat([d["values"]     for d in all_data], dim=0)
    rewards    = float(np.mean([d["reward_sum"] for d in all_data]))
    return states, actions, log_probs, returns, advantages, mus, stds, rewards, values


def collect_samples(envs, actor, critic, gamma: float, lam: float,
                    steps_per_env: int, device: torch.device) -> Dict[str, Any]:
    observation, _ = envs.reset()
    state_tensor = observation['policy']
    
    states, actions, log_probs, rewards, dones, values, mus, stds = [], [], [], [], [], [], [], []

    for _ in range(steps_per_env):
        #state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        #action = dist.sample()
        raw_action = dist.rsample()
        squashed = torch.tanh(raw_action)
        #log_prob = dist.log_prob(action).sum(-1)
        action = squashed
        
        log_prob = dist.log_prob(raw_action).sum(-1) - torch.sum(torch.log(1 - squashed.pow(2) + 1e-6), dim=-1)

        next_observation, reward, terminated, truncated, _ = envs.step(action)
        next_state_tensor = next_observation['policy']
        
        
        terminated_t = torch.as_tensor(terminated, dtype=torch.bool, device=device)
        truncated_t  = torch.as_tensor(truncated,  dtype=torch.bool, device=device)
        done    = terminated_t #| truncated_t 

        # Commented because Isaac-Sim makes reset the envs which has got DONE by itself. No need extra reset manually.
        #if done.any().item():
        #    if hasattr(envs, "reset_done"):
        #        try:
        #            observation, _ = envs.reset_done()
        #            next_state_tensor[done] = observation['policy'][done]
        #        except Exception:
        #            # fallback: entire reset
        #            observation, _ = envs.reset()
        #            state_tensor = observation['policy']
        #            next_state_tensor = state_tensor
        #    else:
        #        observation, _ = envs.reset()
        #        state_tensor = observation['policy']
        #        next_state_tensor = state_tensor


        value = critic(state_tensor).squeeze(-1)
        
        

        states.append(state_tensor)
        actions.append(action.detach().cpu())
        log_probs.append(log_prob.detach().cpu())
        rewards.append(reward.to(dtype=torch.float32, device=device))
        dones.append(done.to(dtype=torch.float32, device=device))
        values.append(value.detach())
        mus.append(mu.detach())
        stds.append(std.detach())

        state_tensor = next_state_tensor

    rewards    = torch.stack(rewards)
    values     = torch.stack(values)
    dones      = torch.stack(dones)
    states     = torch.stack(states)      # [T, B, D]
    actions    = torch.stack(actions)     # [T, B, A]
    log_probs  = torch.stack(log_probs)   # [T, B]
    mus        = torch.stack(mus)         # [T, B, A]
    stds       = torch.stack(stds)        # [T, B, A]    
    
    with torch.no_grad():
        last_value = critic(state_tensor).squeeze(-1).detach()
    returns = compute_gae(rewards, values, dones, gamma, lam, last_value)

    #returns = compute_gae(rewards, values, dones, gamma, lam) # [T, B]
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # [T, B]
    
    T, B = states.shape[:2]
    
    # making 2D/1D
    values_flat = values.reshape(T*B).contiguous()  # [T*B]
    states     = states.reshape(T*B, -1).contiguous()          # [T*B, D]
    actions    = actions.reshape(T*B, -1).contiguous()         # [T*B, A]
    log_probs  = log_probs.reshape(T*B).contiguous()           # [T*B]
    returns    = returns.reshape(T*B).contiguous()             # [T*B]
    advantages = advantages.reshape(T*B).contiguous()          # [T*B]
    mus        = mus.reshape(T*B, -1).contiguous()             # [T*B, A]
    stds       = stds.reshape(T*B, -1).contiguous()            # [T*B, A]
 
    return {
        "states": states.cpu(),
        "actions": actions.cpu(),
        "log_probs": log_probs.cpu(),
        "returns": returns.cpu(),
        "advantages": advantages.cpu(),
        "mus": mus.cpu(),
        "stds": stds.cpu(),
        "values": values_flat.cpu(), 
        "reward_sum": rewards.mean().item(),  
    }


# ---------- worker ----------

def _worker_collect_and_push(worker_id: int, env_id: str, actor_bytes: bytes, critic_bytes: bytes,
                             steps_per_env: int, gamma: float, lam: float,
                             envs_per_worker: int, queue: Queue, model_queue: Queue,
                             start_event: mp.Event) -> None:
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
    
    os.environ["CARB_LOG_LEVEL"] = "error"   # suppress Warning from PhysX
    os.environ["OMNI_LOG_LEVEL"] = "error"   # suppress Warning from PhysX

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=True,
        device="cpu",                             
        experience="isaaclab.python.headless.kit" 
    )
    simulation_app = app_launcher.app
    import carb
    try:
        carb.log.set_default_level(carb.log.Severity.ERROR)
        carb.log.set_channel_level("omni.physx", carb.log.Severity.ERROR)
        carb.log.set_channel_level("omni.physx.plugin", carb.log.Severity.ERROR)
    except Exception as e:
        print("[LOG] couldn't set carb levels:", e)
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

    appo_cfg = load_cfg_from_registry(env_id.split(":")[-1], "appo_cfg_entry_point")    
    obs_dim = appo_cfg['models']['obs_dim']
    act_dim = appo_cfg['models']['act_dim']

    print(f"[WORKER {worker_id}] network dimensions: obs_dim {obs_dim}, act_dim {act_dim}", flush=True)

    actor = Actor(obs_dim, act_dim)
    actor.load_state_dict(torch.load(io.BytesIO(actor_bytes), map_location=device, weights_only=True))
    actor.eval()
    critic = Critic(obs_dim)
    critic.load_state_dict(torch.load(io.BytesIO(critic_bytes), map_location=device, weights_only=True))
    critic.eval()

    # say Hello to the main process
    queue.put({"_hello": True, "worker_id": worker_id, "pid": os.getpid(), "envs": int(envs_per_worker)})

    try:
        while True:
            start_event.wait()  # wait for permission from Main process to go to collect samples  
        
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
    except Exception as e:
        import traceback
        msg = f"[WORKER {worker_id}] crashed:\n{traceback.format_exc()}"
        print(msg, flush=True)
        if errs_path:
            with open(errs_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        # сообщим главному процессу явным сообщением
        try:
            queue.put({"_worker_error": msg})
        except Exception:
            pass
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass


# ---------- runner ----------

class APPOMultiProcRunner:
    def __init__(self, env_name, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        from pathlib import Path
        from datetime import datetime
        
        self.device = device
        self.cfg = train_cfg
        
        self.kl_ema = 0.0
        self.beta_ema = 0.1   

        # === log roots ===
        # prefer dir from train(); otherwise fallback/default
        base_dir = Path(log_dir) if log_dir else Path(self.cfg.get("log_params", {}).get("log_dir", "logs/ppo_run"))
        exp_name = self.cfg.get("experiment_name", "run")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")

        self.resume = bool(self.cfg.get("resume", False))
        
        # logs/ppo_run/<date-time-experiment_name>
        if self.resume:
            # load last checkpoint
            candidates = [p for p in base_dir.iterdir() if p.is_dir()]
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            self.run_root_dir = None
            for c in candidates:
                if (c / "checkpoints").exists():
                    self.run_root_dir = c
                    break
            if self.run_root_dir is None:
                # create new if not found the checkpoint
                self.run_root_dir = base_dir / f"{ts}-{exp_name}"
        else:
            self.run_root_dir = base_dir / f"{ts}-{exp_name}"        
        
    
        if mp.current_process().name == "MainProcess":
            (self.run_root_dir / "tb").mkdir(parents=True, exist_ok=True)
            (self.run_root_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

        # directories
        self.tb_dir = str(self.run_root_dir / "tb")
        self.ckpt_dir = str(self.run_root_dir / "checkpoints")

        # subdirectories
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        print(f"[TB] TensorBoard logs -> {os.path.abspath(self.tb_dir)}")
        print(f"[CKPT] Checkpoints     -> {os.path.abspath(self.ckpt_dir)}")

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

#    def _maybe_resume(self):
#        # look for ckpts in current run_root/checkpoints
#        if self.cfg.get("continue", False):
#            actor_path = get_latest_model(self.ckpt_dir, "actor")
#            critic_path = get_latest_model(self.ckpt_dir, "critic")
#            if actor_path and critic_path:
#                actor_sd  = torch.load(actor_path,  map_location=self.device, weights_only=True)
#                critic_sd = torch.load(critic_path, map_location=self.device, weights_only=True)
#                self.actor.load_state_dict(actor_sd)
#                self.critic.load_state_dict(critic_sd)
#                self.actor.train(); self.critic.train()
#                print("Models were loaded successfully (resume).")
#        else:
#            print("Starting a new run.")

    def _maybe_resume(self):
        # look for ckpts in current run_root/checkpoints
        if self.resume:
            actor_path = get_latest_model(self.ckpt_dir, "actor")
            critic_path = get_latest_model(self.ckpt_dir, "critic")
            if actor_path and critic_path:
                actor_sd  = torch.load(actor_path,  map_location=self.device, weights_only=True)
                critic_sd = torch.load(critic_path, map_location=self.device, weights_only=True)
                self.actor.load_state_dict(actor_sd)
                self.critic.load_state_dict(critic_sd)
                self.actor.train(); self.critic.train()
                # восстановим номер эпизода из имени файла (actor_123.pt)
                import re, os
                def _ep(p):
                    m = re.search(r"_(\d+)\.pt$", os.path.basename(p))
                    return int(m.group(1)) if m else 0
                self.episode = max(_ep(actor_path), _ep(critic_path))
                print(f"Resumed from: {self.run_root_dir} @ episode {self.episode}")
            else:
                print("Resume requested, but no checkpoints found — starting a new run.")
        else:
            print("Starting a new run.")


    def _apply_lr(self):
        for opt in (self.actor_optim, self.critic_optim):
            for g in opt.param_groups:
                g['lr'] = self.lr
                
    # ---------- API ----------
    def learn(self, num_learning_iterations: int | None = None):
        episodes = num_learning_iterations if num_learning_iterations is not None else self.episodes

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        print("Number of CPU cores:", mp.cpu_count())
        self.start_event = mp.Event()   # <- CMD EVENT «GO» to start collect samples for workers	

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
                    self.envs_per_worker, self.queue, self.model_queues[i],
                    self.start_event,
                ),
                daemon=False,
            )
            p.start()
            self.workers.append(p)
            
        # === wait for all workers alive ===
        ready = set()
        while len(ready) < self.num_workers:
            m = self.queue.get()
            if isinstance(m, dict) and m.get("_hello"):
                ready.add(m["worker_id"])
                print(f"[READY] W{m['worker_id']} pid={m['pid']} envs={m['envs']}")
        # ============================================   
        # 10 second cool down to allow all Isaac-Sim envs to be loaded completely
        print("[ BARRIER ] All workers ready. Releasing start in 10 seconds ...")
        time.sleep(7.0)
        print("3")
        time.sleep(1.0)
        print("2")
        time.sleep(1.0)
        print("1")
        time.sleep(1.0)
        self.start_event.set() # permission to workers to start collect samples

        try:
            print("[ TRAINING STARTED ]")
            while self.episode < episodes:
            
                
            
                all_data = [self.queue.get() for _ in range(self.num_workers)] # get data from workers
                
                self.start_event.clear() # block workers next epoch   
                time.sleep(0.001)
                self.start_event.set()   # !!! permit workers to collect again  !!!       
                
                
                states, actions, old_log_probs, returns, advantages, mus, stds, rewards, old_values = combine_batches(all_data)
                # print(f'Episode {self.episode} State shape of received buffer {states.shape}') (16384, 235) with W × B × T = 4 × 128 × 32 = 16384
                device = self.device
                states = states.to(device)
                actions = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns = returns.to(device)
                advantages = advantages.to(device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize
                mus = mus.to(device)
                stds = stds.to(device)
                old_values = old_values.to(device)

                self.reward_history.append(rewards)
                
                kl_epoch = []
	

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
                        batch_values = old_values[b_idx]

                        mu, std = self.actor(batch_states)
                        dist = torch.distributions.Normal(mu, std)

                        eps = 1e-6
                        a = torch.clamp(batch_actions, -1 + eps, 1 - eps)
                        raw = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh(a)

                        new_log_probs = dist.log_prob(raw).sum(dim=-1) - torch.sum(torch.log(1 - a.pow(2) + eps), dim=-1)
                        entropy = dist.entropy().sum(dim=-1).mean()

                        with torch.no_grad():
                            old_dist = torch.distributions.Normal(batch_mus, batch_stds)
                            kl_div = torch.distributions.kl_divergence(old_dist, dist).sum(dim=-1).mean()
                            kl_epoch.append(kl_div.detach())
                            

                        #if kl_div > self.kl_treshold * 1.5:
                        #    self.clip_eps = max(self.clip_eps * 0.9999, self.clip_eps_min)
                        #    self.lr = max(self.lr * 0.999, 5e-5)
                        #    self._apply_lr()
                        #    
                        #    # break #
                        #if kl_div < self.kl_treshold * 0.66:
                        #    self.clip_eps = min(self.clip_eps * 1.0001, self.clip_eps_max)
                        #    self.lr = min(self.lr * 1.001, 5e-2)
                        #    self._apply_lr()
                            

                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                        
                        
                        #critic_loss = nn.MSELoss()(self.critic(batch_states).squeeze(-1), batch_returns)
                        v_pred = self.critic(batch_states).squeeze(-1)
                        v_old  = batch_values 

                        v_clip = (v_pred - v_old).clamp(-0.2, 0.2) + v_old
                        critic_loss = 0.5 * torch.max((v_pred - batch_returns).pow(2), (v_clip - batch_returns).pow(2)).mean()
                        #critic_loss = (v_pred - batch_returns).pow(2).mean()
                        
                        
                        self.actor_optim.zero_grad(set_to_none=True) #
                        actor_loss.backward()
                        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8) # 0.5
                        self.actor_optim.step()

                        self.critic_optim.zero_grad(set_to_none=True) #
                        critic_loss.backward()
                        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.5) # 0.5
                        self.critic_optim.step()

                
                
                kl_mean = torch.stack(kl_epoch).mean().item()
                self.kl_ema = (1 - self.beta_ema) * self.kl_ema + self.beta_ema * kl_mean
                
                if self.kl_ema > self.kl_treshold * 1.5:
                    self.lr = max(self.lr * 0.5, 1e-5)
                    self.clip_eps = max(self.clip_eps * 0.9, self.clip_eps_min)
                    self._apply_lr()
                elif self.kl_ema < self.kl_treshold * 0.5:
                    self.lr = min(self.lr * 1.2, 1e-3)     
                    self.clip_eps = min(self.clip_eps * 1.05, self.clip_eps_max)
                    self._apply_lr()

                
                if self.writer:
                    try:
                        avg_reward = float(np.mean(self.reward_history[-1]))
                        self.writer.add_scalar("Rewards/Avg_Reward_10", avg_reward, self.episode)
                        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.episode)
                        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.episode)
                        self.writer.add_scalar("Metrics/LR", self.lr, self.episode)
                        self.writer.add_scalar("Metrics/Entropy", entropy.item(), self.episode)
                        self.writer.add_scalar("Metrics/Clip_eps", self.clip_eps, self.episode)
                        self.writer.add_scalar("Metrics/KL_Mean", kl_mean, self.episode)
                        self.writer.add_scalar("Metrics/KL_EMA",  self.kl_ema, self.episode)
                        if (self.episode + 1) % 10 == 0:                     
                            print(f"Episode {self.episode + 1}: Avg reward = {avg_reward:.5f}")
                    except:
                        pass        

                # save to run_root/checkpoints
                if (self.episode + 1) % self.save_model_every == 0:
                    actor_path = os.path.join(self.ckpt_dir, f"actor_{self.episode + 1}.pt")
                    critic_path = os.path.join(self.ckpt_dir, f"critic_{self.episode + 1}.pt")
                    torch.save(self.actor.state_dict(),  os.path.join(self.ckpt_dir, f"actor_{self.episode+1}.pt"))
                    torch.save(self.critic.state_dict(), os.path.join(self.ckpt_dir, f"critic_{self.episode+1}.pt"))
                    print(f"[CKPT] Saved: {actor_path} | {critic_path}")

                self.episode += 1
                
                

                # broadcast updated weights
                if self.episode > 0 and self.episode % 1 == 0:
                    actor_serialized = serialize_state_dict(move_state_dict_to_cpu(self.actor.state_dict()))
                    critic_serialized = serialize_state_dict(move_state_dict_to_cpu(self.critic.state_dict()))
                
                    # send new weights to workers
                    for q in self.model_queues:
                        q.put((actor_serialized, critic_serialized))
                    
                # --- resurrection of the dead workers ---
                for i, p in enumerate(self.workers):
                    if not p.is_alive():
                        exitcode = p.exitcode
                        print(f"[WARN] worker {i} died (exit={exitcode}). Respawning...")
                        try:
                            p.join(timeout=0.1)
                        except Exception:
                            pass
                        new_q = mp.Queue()
                        self.model_queues[i] = new_q

                        # send actual weights with new queue
                        p2 = mp.Process(
                            target=_worker_collect_and_push,
                            args=(
                                i, self.env_name, actor_serialized, critic_serialized,
                                self.steps_per_env, self.gamma, self.lam,
                                self.envs_per_worker, self.queue, new_q,
                                self.start_event
                            ),
                            daemon=False,
                        )
                        p2.start()
                        self.workers[i] = p2
                # ------------------------------------  
                # self.start_event.set()   # !!! permit workers to collect again  !!!
                
                    

            print("Training finished")
        finally:
            for p in self.workers:
                p.terminate()
            if self.queue is not None:
                self.queue.close()
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()

