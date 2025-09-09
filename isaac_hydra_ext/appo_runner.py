import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from typing import Callable, Optional

from train_appo_gpu import train
from utils import Actor, Critic

def make_isaaclab_factory(task_cfg: DictConfig) -> Callable[[], object]:
    
    import gymnasium as gym
    def _factory():
        return gym.make(task_cfg.gym_id, **task_cfg.get("kwargs", {}))
    return _factory

@hydra.main(config_path="conf", config_name="train_appo", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor  = Actor(cfg.model.obs_dim, cfg.model.act_dim).to(device)
    critic = Critic(cfg.model.obs_dim).to(device)

    env_factory: Optional[Callable] = None
    env_name = cfg.env.name

    if cfg.env.kind == "isaaclab":
        env_factory = make_isaaclab_factory(cfg.env.task)
    elif cfg.env.kind == "gym":
        pass
    else:
        raise ValueError(f"Unknown env.kind={cfg.env.kind}")

    train(
        env_name=env_name,
        actor=actor,
        critic=critic,
        num_workers=cfg.ppo.num_workers,
        envs_per_worker=cfg.ppo.envs_per_worker,
        gamma=cfg.ppo.gamma,
        lam=cfg.ppo.lam,
        clip_eps_start=cfg.ppo.clip_eps_start,
        clip_eps_max=cfg.ppo.clip_eps_max,
        clip_eps_min=cfg.ppo.clip_eps_min,
        lr=cfg.ppo.lr,
        entropy_coef=cfg.ppo.entropy_coef,
        episodes=cfg.ppo.episodes,
        update_epochs=cfg.ppo.update_epochs,
        batch_size=cfg.ppo.batch_size,
        steps_per_env=cfg.ppo.steps_per_env,
        log_dir=cfg.logging.log_dir,
        save_model_every=cfg.checkpoint.save_model_every,
        kl_treshold=cfg.ppo.kl_treshold,
        experiment_name=cfg.experiment.name,
        env_factory=env_factory,
    )

if __name__ == "__main__":
    main()

