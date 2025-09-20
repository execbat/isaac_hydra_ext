# appo_multiproc_runner.py (patched)
from __future__ import annotations

import os
import io
import time
import queue as pyqueue
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import multiprocessing as mp

from typing import List, Tuple, Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Queue, Process

from isaac_hydra_ext.utils import Actor, Critic


# ---------- utils ----------
def get_latest(q, default=None):
    """
    drain all elems from Q and keep last.
    """
    last = default
    while True:
        try:
            last = q.get_nowait()
        except pyqueue.Empty:
            return last
        except (OSError, EOFError):
            return last

def drain_queue(q) -> int:
    """
    drain Q completely
    """
    dropped = 0
    while True:
        try:
            q.get_nowait()
            dropped += 1
        except pyqueue.Empty:
            break
        except (OSError, EOFError):
            # очередь закрыта/сломана — считаем, что опустошена
            break
    return dropped
    
def keep_latest(q, new_item):
    """
    Drain Q and put item into it
    """
    try:
        drain_queue(q)
        q.put_nowait(new_item)
    except (OSError, EOFError):
        pass    

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
    states     = torch.cat([d["states"]     for d in all_data], dim=0)
    actions    = torch.cat([d["actions"]    for d in all_data], dim=0)
    log_probs  = torch.cat([d["log_probs"]  for d in all_data], dim=0)
    returns    = torch.cat([d["returns"]    for d in all_data], dim=0)
    advantages = torch.cat([d["advantages"] for d in all_data], dim=0)
    mus        = torch.cat([d["mus"]        for d in all_data], dim=0)
    stds       = torch.cat([d["stds"]       for d in all_data], dim=0)
    values     = torch.cat([d["values"]     for d in all_data], dim=0)

    # aggregating raw sums
    ep_ret_sum = sum(float(d.get("ep_return_sum_batch", 0.0)) for d in all_data)
    ep_len_sum = sum(int(d.get("ep_length_sum_batch", 0))     for d in all_data)
    ep_cnt_sum = sum(int(d.get("ep_count_batch", 0))          for d in all_data)

    rew_sum    = sum(float(d.get("reward_sum_batch", 0.0))    for d in all_data)
    step_sum   = sum(int(d.get("step_count_batch", 0))        for d in all_data)

    # global means
    avg_ep_return = (ep_ret_sum / ep_cnt_sum) if ep_cnt_sum > 0 else float("nan")
    avg_ep_length = (ep_len_sum / ep_cnt_sum) if ep_cnt_sum > 0 else float("nan")
    avg_reward_per_step = (rew_sum / step_sum) if step_sum > 0 else 0.0

    return (states, actions, log_probs, returns, advantages,
            mus, stds, avg_reward_per_step, values, avg_ep_return, avg_ep_length)




def collect_samples(envs, state_tensor, actor, critic, gamma: float, lam: float,
                    steps_per_env: int, device: torch.device) -> Dict[str, Any]:
                    
    
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
        done    = terminated_t | truncated_t 

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
    #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # [T, B]
    
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
    #rewards    = rewards.reshape(T*B, -1).contiguous() 
    #dones      = dones.reshape(T*B, -1).contiguous() 
    
 
    return {
        "states": states.cpu(),
        "actions": actions.cpu(),
        "log_probs": log_probs.cpu(),
        "returns": returns.cpu(),
        "advantages": advantages.cpu(),
        "mus": mus.cpu(),
        "stds": stds.cpu(),
        "values": values_flat.cpu(),   
        
        "rewards_seq": rewards.cpu(),   # [T, B] float32
        "dones_seq":   dones.cpu(),     
        },  state_tensor


import math
import torch
from typing import Dict, Tuple, Optional

def update_episode_stats_from_batch(
    samples: Dict,
    ep_ret: torch.Tensor,          # [B] float32 
    ep_len: torch.Tensor,          # [B] int32   
    last_avg: Optional[Dict[str, float]] = None,  # {"avg_ep_return": float, "avg_ep_length": float}
    done_threshold: float = 0.5,
) -> Tuple[Dict, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Обновляет статистику эпизодов по последовательности шагов из батча.
    Никогда не возвращает NaN: если нет завершённых эпизодов в батче,
    используется прошлое значение средних (last_avg).

    Возвращает: (samples, ep_ret, ep_len, last_avg)
      - samples дополнен ключами:
          "avg_ep_return", "avg_ep_length", "avg_reward_per_step",
          "ep_return_sum_batch", "ep_length_sum_batch", "ep_count_batch"
      - last_avg обновлён, если были завершённые эпизоды
    """
    if last_avg is None:
        last_avg = {"avg_ep_return": 0.0, "avg_ep_length": 0.0}

    if not math.isfinite(last_avg.get("avg_ep_return", 0.0)):
        last_avg["avg_ep_return"] = 0.0
    if not math.isfinite(last_avg.get("avg_ep_length", 0.0)):
        last_avg["avg_ep_length"] = 0.0

    rewards_seq = samples.pop("rewards_seq")   # [T, B] float32
    dones_seq   = samples.pop("dones_seq")     # [T, B] bool|float

    if dones_seq.dtype != torch.bool:
        dones_seq = dones_seq > done_threshold

    T, B = rewards_seq.shape
    assert ep_ret.numel() == B and ep_len.numel() == B, "B mismatch with accumulators"

    batch_ep_ret_sum = 0.0
    batch_ep_len_sum = 0
    batch_ep_cnt     = 0

    for t in range(T):
        ep_ret += rewards_seq[t]      # [B]
        ep_len += 1                   # [B]
        done_idx = torch.nonzero(dones_seq[t], as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            batch_ep_ret_sum += float(ep_ret[done_idx].sum().item())
            batch_ep_len_sum += int(ep_len[done_idx].sum().item())
            batch_ep_cnt     += int(done_idx.numel())
            # reset of new episodes
            ep_ret[done_idx] = 0.0
            ep_len[done_idx] = 0

    # average data per batch
    if batch_ep_cnt > 0:
        avg_ep_return = batch_ep_ret_sum / batch_ep_cnt
        avg_ep_length = batch_ep_len_sum / batch_ep_cnt
        # update
        last_avg["avg_ep_return"] = avg_ep_return
        last_avg["avg_ep_length"] = avg_ep_length
    else:
        # no dones - use previous data
        avg_ep_return = last_avg["avg_ep_return"]
        avg_ep_length = last_avg["avg_ep_length"]

    reward_sum_batch = float(rewards_seq.sum().item())
    step_count_batch = int(rewards_seq.numel())

    # pack metrics
    samples["avg_ep_return"]        = float(avg_ep_return)
    samples["avg_ep_length"]        = float(avg_ep_length)
    samples["avg_reward_per_step"]  = (reward_sum_batch / step_count_batch) if step_count_batch > 0 else 0.0    
    samples["ep_return_sum_batch"]  = float(batch_ep_ret_sum)  # raw data
    samples["ep_length_sum_batch"]  = int(batch_ep_len_sum)    # raw data
    samples["ep_count_batch"]       = int(batch_ep_cnt)        # raw data
    samples["reward_sum_batch"]     = float(reward_sum_batch)  # raw data
    samples["step_count_batch"]     = int(step_count_batch)    # raw data

    return samples, ep_ret, ep_len, last_avg


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
    
    # reset env only once at start
    observation, _ = envs.reset()
    state_tensor = observation['policy']
    
    # accumulators for statistics
    B = state_tensor.shape[0]
    ep_ret = torch.zeros(B, dtype=torch.float32, device = device)
    ep_len = torch.zeros(B, dtype=torch.int32, device = device)
    last_avg = {"avg_ep_return": 0.0, "avg_ep_length": 0.0}

    try:
        while True:
            start_event.wait()  # wait for permission from Main process to go to collect samples
            time.sleep(0.00001)  
        
            try:
                new_actor_bytes, new_critic_bytes = get_latest(model_queue) # get the last model weights from the Q
                actor.load_state_dict(torch.load(io.BytesIO(new_actor_bytes),  map_location=device, weights_only=True))
                critic.load_state_dict(torch.load(io.BytesIO(new_critic_bytes), map_location=device, weights_only=True))
            except Exception:
                pass

            with torch.no_grad():
                # receive last state_tensor from previous collection to continue from that point
                samples, state_tensor = collect_samples(envs, state_tensor, actor.to(device), critic.to(device),
                                          gamma, lam, steps_per_env, device)
                                          
                # exctact statistics from samples                          
                samples, ep_ret, ep_len, last_avg = update_episode_stats_from_batch(samples, ep_ret, ep_len, last_avg)
               
            # send samples to master process                              
            #queue.put(samples)
            keep_latest(queue, samples)
            
    except Exception as e:
        import traceback
        msg = f"[WORKER {worker_id}] crashed:\n{traceback.format_exc()}"
        print(msg, flush=True)
        if errs_path:
            with open(errs_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        
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
        
        # stats
        self._last_global_avg_return = 0.0
        self._last_global_avg_length = 0.0

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
                time.sleep(0.00001)
                
                drain_queue(self.queue)  # drain all samples from Q.
                
                self.start_event.set()   # !!! permit workers to collect again  !!!     
                
                states, actions, old_log_probs, returns, advantages, mus, stds, avg_reward_per_step, old_values, avg_ep_return, avg_ep_length = combine_batches(all_data)
                
                #------------------------------------------------------------------
                # working with statistics                
                if not np.isfinite(avg_ep_return) or not np.isfinite(avg_ep_length):
                    # if no finished episodes
                    avg_ep_return = self._last_global_avg_return
                    avg_ep_length = self._last_global_avg_length
                else:
                    self._last_global_avg_return = float(avg_ep_return)
                    self._last_global_avg_length = float(avg_ep_length)
                #------------------------------------------------------------------
                
                
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

                self.reward_history.append(avg_reward_per_step)
                
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
                        
                        
                        critic_loss = nn.MSELoss()(self.critic(batch_states).squeeze(-1), batch_returns)
                        #v_pred = self.critic(batch_states).squeeze(-1)
                        #v_old  = batch_values 

                        #v_clip = (v_pred - v_old).clamp(-0.2, 0.2) + v_old
                        #critic_loss = 0.5 * torch.max((v_pred - batch_returns).pow(2), (v_clip - batch_returns).pow(2)).mean()
                        #critic_loss = (v_pred - batch_returns).pow(2).mean()
                        
                        
                        self.actor_optim.zero_grad() # set_to_none=True
                        actor_loss.backward()
                        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.8) # 0.5
                        self.actor_optim.step()

                        self.critic_optim.zero_grad() # set_to_none=True
                        critic_loss.backward()
                        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.5) # 0.5
                        self.critic_optim.step()

                
                
                kl_mean = torch.stack(kl_epoch).mean().item()
                self.kl_ema = (1 - self.beta_ema) * self.kl_ema + self.beta_ema * kl_mean
                
                if self.kl_ema > self.kl_treshold * 1.5:
                    self.lr = max(self.lr * 0.99, 1e-5)
                    self.clip_eps = max(self.clip_eps * 0.995, self.clip_eps_min)
                    self._apply_lr()
                elif self.kl_ema < self.kl_treshold * 0.5:
                    self.lr = min(self.lr * 1.01, 5e-3)     
                    self.clip_eps = min(self.clip_eps * 1.005, self.clip_eps_max)
                    self._apply_lr()

                
                if self.writer:
                    try:
                        
                        self.writer.add_scalar("Rewards/avg_reward_per_step",   avg_reward_per_step ,        self.episode)
                        self.writer.add_scalar("Rewards/avg_episode_return",   avg_ep_return,        self.episode)
                        self.writer.add_scalar("Rewards/avg_episode_length",   avg_ep_length ,        self.episode)
                        self.writer.add_scalar("Loss/Actor", actor_loss.item(), self.episode)
                        self.writer.add_scalar("Loss/Critic", critic_loss.item(), self.episode)
                        self.writer.add_scalar("Metrics/LR", self.lr, self.episode)
                        self.writer.add_scalar("Metrics/Entropy", entropy.item(), self.episode)
                        self.writer.add_scalar("Metrics/Clip_eps", self.clip_eps, self.episode)
                        self.writer.add_scalar("Metrics/KL_Mean", kl_mean, self.episode)
                        self.writer.add_scalar("Metrics/KL_EMA",  self.kl_ema, self.episode)
                        if (self.episode + 1) % 10 == 0:                     
                            print(f"Episode {self.episode + 1}: Avg episode reward = {avg_ep_return:.5f}")
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
                        keep_latest(q, (actor_serialized, critic_serialized)) # drop out everything and put
                        #q.put((actor_serialized, critic_serialized))
                    
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
