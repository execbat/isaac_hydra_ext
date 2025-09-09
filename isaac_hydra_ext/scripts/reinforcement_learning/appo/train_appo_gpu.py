import os
import io
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
#import custom_envs  # your custom environments
#import mujoco
import multiprocessing as mp

from typing import List, Tuple, Dict, Any
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import Queue, Process

from isaac_hydra_ext.utils import Actor, Critic

# Set CUDA environment options
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    """
    Compute Generalized Advantage Estimation (GAE).
    """
    values = torch.cat([values, torch.zeros_like(values[0:1])], dim=0)
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]

    returns = torch.zeros_like(deltas)
    gae = 0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        returns[t] = gae + values[t]
    return returns


def combine_batches(all_data: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    """
    Concatenate tensors from all worker batches.
    """
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
    """
    Collect rollout samples from environments.
    """
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
            done_indices = np.where(done)[0]
            reset_obs, _ = envs.reset()
            for idx in done_indices:
                next_state[idx] = reset_obs[idx]

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


def worker_collect_and_push(worker_id: int, env_name: str, actor_bytes: bytes, critic_bytes: bytes,
                            steps_per_env: int, gamma: float, lam: float,
                            envs_per_worker: int, queue: Queue, model_queue: Queue) -> None:
    """
    Worker process that collects data and sends it to the main process via queue.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Disable GPU in worker
    torch.set_num_threads(1)
    device = torch.device("cpu")

    import gymnasium as gym
    from utils import Actor, Critic

    actor_state = torch.load(io.BytesIO(actor_bytes), map_location=device)
    critic_state = torch.load(io.BytesIO(critic_bytes), map_location=device)

    def make_env():
        return gym.make(env_name, max_episode_steps=2000, render=False, spawn_ball_every=500)

    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(envs_per_worker)])

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim)
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    actor.eval()
    critic.eval()

    while True:
        # Update weights if new models are available
        try:
            new_actor_bytes, new_critic_bytes = model_queue.get_nowait()
            actor.load_state_dict(torch.load(io.BytesIO(new_actor_bytes), map_location=device))
            critic.load_state_dict(torch.load(io.BytesIO(new_critic_bytes), map_location=device))
        except:
            pass  # Continue with existing weights

        with torch.no_grad():
            samples = collect_samples(envs, actor.to(device), critic.to(device),
                                      gamma, lam, steps_per_env, device)
        queue.put(samples)


def train(env_name: str,
          actor: nn.Module,
          critic: nn.Module,
          num_workers : int = 8,
          envs_per_worker : int = 2,
          gamma: float = 0.99,
          lam: float = 0.95,
          clip_eps_start: float = 0.2,
          clip_eps_max: float = 0.2,
          clip_eps_min: float = 0.05,
          lr: float = 0.001,
          entropy_coef: float = 0.001,
          episodes: int = 1000,
          update_epochs: int = 5,
          batch_size: int = 512,
          steps_per_env: int = 128,
          log_dir: str = "runs/ppo_run/",
          save_model_every: int = 10,
          kl_treshold: float = 0.03) -> None:
    """
    Train PPO agent using parallel environments.

    Args:
        envs: Vectorized parallel environments.
        actor: Policy network.
        critic: Value network.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        clip_eps: Clipping epsilon for PPO.
        lr: Learning rate.
        entropy_coef: Entropy coefficient for exploration bonus.
        episodes: Number of training episodes.
        update_epochs: Number of epochs per training step.
        batch_size: Minibatch size.
        steps_per_env: Steps collected per environment before update.
        log_dir: Directory for TensorBoard logs.
        save_model_every: Frequency (episodes) to save models.
        kl_treshold: KL-divergence threshold for adaptive clip adjustment.
    """
    
    
    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    actor.share_memory()
    critic.share_memory()

    actor_serialized = serialize_state_dict(move_state_dict_to_cpu(actor.state_dict()))
    critic_serialized = serialize_state_dict(move_state_dict_to_cpu(critic.state_dict()))

    queue = mp.Queue()
    workers = []
    model_queues = [mp.Queue() for _ in range(num_workers)]

    for i in range(num_workers):
        p = mp.Process(target=worker_collect_and_push,
                       args=(i, env_name, actor_serialized, critic_serialized,
                             steps_per_env, gamma, lam, envs_per_worker, queue, model_queues[i]))
        p.daemon = False
        p.start()
        workers.append(p)

    clip_eps = clip_eps_start
    reward_history = []
    episode = 0

    try:
        while episode < episodes:
            all_data = []
            for _ in range(num_workers):
                batch = queue.get()
                all_data.append(batch)

            states, actions, old_log_probs, returns, advantages, mus, stds, rewards = combine_batches(all_data)

            states = states.to(device)
            actions = actions.to(device)
            old_log_probs = old_log_probs.to(device)
            returns = returns.to(device)
            advantages = advantages.to(device)
            mus = mus.to(device)
            stds = stds.to(device)

            reward_history.append(rewards)

            for _ in range(update_epochs):
                idx = torch.randperm(states.size(0))
                for start in range(0, states.size(0), batch_size):
                    end = start + batch_size
                    b_idx = idx[start:end]

                    batch_states = states[b_idx]
                    batch_actions = actions[b_idx]
                    batch_old_log_probs = old_log_probs[b_idx]
                    batch_returns = returns[b_idx]
                    batch_advantages = advantages[b_idx]
                    batch_mus = mus[b_idx]
                    batch_stds = stds[b_idx]

                    mu, std = actor(batch_states)
                    dist = torch.distributions.Normal(mu, std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()

                    with torch.no_grad():
                        old_dist = torch.distributions.Normal(batch_mus, batch_stds)
                        kl_div = torch.distributions.kl_divergence(old_dist, dist).sum(dim=-1).mean()

                    if kl_div > kl_treshold * 1.5:
                        clip_eps = max(clip_eps * 0.9, clip_eps_min)
                    if kl_div < kl_treshold * 0.5:
                        clip_eps = min(clip_eps * 1.1, clip_eps_max)

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
                    critic_loss = nn.MSELoss()(critic(batch_states).squeeze(-1), batch_returns)

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()

                    critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_optim.step()

            writer.add_scalar("Loss/Actor", actor_loss.item(), episode)
            writer.add_scalar("Loss/Critic", critic_loss.item(), episode)
            writer.add_scalar("Metrics/KL_Div", kl_div.item(), episode)
            writer.add_scalar("Metrics/Entropy", entropy.item(), episode)
            writer.add_scalar("Metrics/Clip_eps", clip_eps, episode)

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(reward_history[-10:])
                writer.add_scalar("Rewards/Avg_Reward_10", avg_reward, episode)
                print(f"Episode {episode+1}: Avg reward = {avg_reward:.2f}")

            if (episode + 1) % save_model_every == 0:
                save_path = os.path.join("experiments", experiment_name)
                os.makedirs(save_path, exist_ok=True)
                torch.save(actor, os.path.join(save_path, f"actor_{episode+1}.pt"))
                torch.save(critic, os.path.join(save_path, f"critic_{episode+1}.pt"))
                print(f"Saved models at episode {episode+1}")

            episode += 1
            
            # Сериализуем актуальные веса
            actor_serialized = serialize_state_dict(move_state_dict_to_cpu(actor.state_dict()))
            critic_serialized = serialize_state_dict(move_state_dict_to_cpu(critic.state_dict()))

            # Отправляем во все очереди
            for q in model_queues:
                q.put((actor_serialized, critic_serialized))
            
    finally:
        for p in workers:
            p.terminate()
        queue.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("Number of CPU cores: ", mp.cpu_count())

    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='cont', action='store_true', help='Continue training from checkpoint')
    args = parser.parse_args()

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ppo_params = config['ppo_params']
    log_params = config['log_params']
    env_name = config['env_name']
    checkpoint_params = config['checkpoint_params']
    experiment_name = config['experiment_name']

    #num_envs = 32  # <- how many envs in parallel
    save_dir = os.path.join("experiments", experiment_name)
    

    #envs = gym.vector.AsyncVectorEnv([make_env(env_name) for _ in range(num_envs)])
    env = gym.make(env_name, max_episode_steps=ppo_params['max_episode_steps'], render=False, spawn_ball_every=500)
    #n_obs = envs.single_observation_space.shape[0]
    #n_actions = envs.single_action_space.shape[0]
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]


    if os.path.exists(save_dir):
        if not args.cont:
            print("This experiment already exists. Use '--continue' to resume training.")
            exit(0)
        actor = torch.load(get_latest_model(save_dir, "actor"), map_location=device, weights_only = False)
        critic = torch.load(get_latest_model(save_dir, "critic"), map_location=device, weights_only = False)
        actor.train()
        critic.train()
        print("Models were loaded successfully")  
    else:    
        os.makedirs(save_dir, exist_ok=True)
        actor = Actor(n_obs, n_actions).to(device)
        critic = Critic(n_obs).to(device)
        print("Models have been created")

    print("Training started")
    train(env_name, 
          actor, 
          critic,
          num_workers = ppo_params["num_workers"],
          envs_per_worker = ppo_params["envs_per_worker"],
          gamma=ppo_params["gamma"],
          lam=ppo_params["lam"],
          clip_eps_start=ppo_params["clip_eps_start"],
          clip_eps_max=ppo_params["clip_eps_max"],
          clip_eps_min=ppo_params["clip_eps_min"],
          lr=ppo_params["lr"],
          entropy_coef=ppo_params["entropy_coef"],
          episodes=ppo_params["episodes"],
          update_epochs=ppo_params["update_epochs"],
          batch_size=ppo_params["batch_size"],
          steps_per_env=ppo_params["steps_per_env"],  # <- steps between updates
          log_dir=log_params["log_dir"],
          save_model_every=checkpoint_params["save_model_every"],
          kl_treshold=ppo_params["kl_treshold"])
    print("Training finished")

