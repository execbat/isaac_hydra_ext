# train.py
from __future__ import annotations
import argparse, os, multiprocessing as mp, re, yaml, importlib.util as ilu
from pathlib import Path

# -------- utils --------
def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def clean(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items() if v is not None}

def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def default_appo_params() -> dict:
    return {
        "num_workers": 1,
        "envs_per_worker": 1,
        "gamma": 0.99,
        "lam": 0.95,
        "clip_eps_start": 0.2,
        "clip_eps_max": 0.2,
        "clip_eps_min": 0.05,
        "lr": 3e-4,
        "entropy_coef": 1e-3,
        "update_epochs": 5,
        "batch_size": 128,
        "steps_per_env": 24,
        "kl_treshold": 0.03,
    }

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9]+", (s or "").lower()))

def find_appo_cfg(task: str, explicit_path: str | None) -> str | None:
    # 1) explicit
    if explicit_path and Path(explicit_path).is_file():
        return explicit_path
    # 2) ./appo_cfg.yaml
    cwd_cfg = Path("appo_cfg.yaml")
    if cwd_cfg.is_file():
        return str(cwd_cfg)
    # 3) scan inside isaac_hydra_ext package
    spec = ilu.find_spec("isaac_hydra_ext")
    if not spec or not spec.submodule_search_locations:
        return None
    base = Path(list(spec.submodule_search_locations)[0]) / "source" / "isaaclab_tasks"
    candidates = list(base.rglob("agents/appo_cfg.yaml"))
    if not candidates:
        return None
    T = _tokens(task)
    def score(p: Path) -> int:
        return len(T & _tokens(str(p)))
    candidates.sort(key=score, reverse=True)
    return str(candidates[0])

# -------- parser --------
def build_parser():
    p = argparse.ArgumentParser(
        description="Train APPO (main process is Isaac/Kit-free).",
        conflict_handler="resolve",
    )
    p.add_argument("--task", type=str, required=True, help="e.g. Isaac-Velocity-Sber-Unitree-Go1-v0")
    p.add_argument("--appo_cfg_path", type=str, default=None, help="Path to appo_cfg.yaml")
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max_iterations", type=int, default=None)

    # devices/mode
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--sim_device", type=str, default=None)   # cpu / cuda:0
    p.add_argument("--rl_device", type=str, default=None)    # cpu / cuda:0
    p.add_argument("--device", type=str, default=None)       # optional hint

    # video
    p.add_argument("--video", action="store_true", default=False)
    p.add_argument("--video_length", type=int, default=200)
    p.add_argument("--video_interval", type=int, default=2000)

    # APPO overrides
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--envs_per_worker", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--clip_eps_start", type=float, default=None)
    p.add_argument("--clip_eps_max", type=float, default=None)
    p.add_argument("--clip_eps_min", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--entropy_coef", type=float, default=None)
    p.add_argument("--update_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--steps_per_env", type=int, default=None)
    p.add_argument("--kl_treshold", type=float, default=None)
    return p

# -------- main --------
def main():
    mp.set_start_method("spawn", force=True)

    args = build_parser().parse_args()

    # env for workers that will start Kit
    if args.video:
        os.environ.setdefault("ENABLE_CAMERAS", "1")
    os.environ.setdefault("KIT_WINDOWMODE", "headless")
    os.environ.setdefault("OMNI_KIT_WINDOW_FLAGS", "headless")
    os.environ.setdefault("PYTHON_NO_USD_RENDER", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # base cfg
    train_cfg = {
        "env_name": args.task,
        "experiment_name": "appo_run",
        "seed": 42,
        "algo": "appo",
        "device": args.device,
        "resume": False,
        "max_iterations": 1000,
        "num_envs": args.num_envs,

        "sim_device": args.sim_device,
        "rl_device": args.rl_device,

        "video": bool(args.video),
        "video_length": args.video_length,
        "video_interval": args.video_interval,
        "headless": bool(args.headless),

        "log_params": {
            "log_dir": "logs/ppo_run",
            "save_model_every": 10,
            "logger": "tensorboard",
        },

        "appo_params": default_appo_params(),
        "models": {"obs_dim": None, "act_dim": None},
    }

    # merge YAML (auto-discovery)
    cfg_path = find_appo_cfg(args.task, args.appo_cfg_path)
    if cfg_path:
        yaml_cfg = load_yaml(cfg_path)
        deep_update(train_cfg, yaml_cfg)

    # CLI overrides
    if args.seed is not None:
        train_cfg["seed"] = args.seed
    if args.max_iterations is not None:
        train_cfg["max_iterations"] = args.max_iterations
    deep_update(train_cfg["appo_params"], clean({
        "num_workers": args.num_workers,
        "envs_per_worker": args.envs_per_worker,
        "gamma": args.gamma,
        "lam": args.lam,
        "clip_eps_start": args.clip_eps_start,
        "clip_eps_max": args.clip_eps_max,
        "clip_eps_min": args.clip_eps_min,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
        "update_epochs": args.update_epochs,
        "batch_size": args.batch_size,
        "steps_per_env": args.steps_per_env,
        "kl_treshold": args.kl_treshold,
    }))

    # sanity for experiment_name
    if not train_cfg.get("experiment_name"):
        train_cfg["experiment_name"] = "appo_run"

    # must have model dims now (your YAML has them)
    m = train_cfg.get("models", {})
    if m.get("obs_dim") is None or m.get("act_dim") is None:
        raise ValueError("models.obs_dim and models.act_dim must be provided (e.g. 235/12 for Go1).")

    # import only the runner (it must not import Isaac/Kit at module top)
    from isaac_hydra_ext.scripts.reinforcement_learning.appo.runners.on_policy_runner import (
        APPOMultiProcRunner as OnPolicyRunner
    )
    runner = OnPolicyRunner(env_name=args.task, train_cfg=train_cfg, log_dir=train_cfg['log_params']['log_dir'], device =train_cfg['device'])
    runner.learn(num_learning_iterations=train_cfg["max_iterations"])

if __name__ == "__main__":
    main()

