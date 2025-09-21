# isaac_hydra_ext/scripts/run_test_with_ext.py
import os
import sys

def _ensure_orbit_paths():
    orbit_root = os.environ.get("ORBIT_ROOT", os.getcwd())
    if orbit_root not in sys.path:
        sys.path.insert(0, orbit_root)

    rsl_rl_dir = os.path.join(orbit_root, "scripts", "reinforcement_learning", "rsl_rl")
    if rsl_rl_dir not in sys.path:
        sys.path.insert(0, rsl_rl_dir)

    cli_args_py = os.path.join(rsl_rl_dir, "cli_args.py")
    if not os.path.exists(cli_args_py):
        raise RuntimeError(f"cli_args.py not found: {cli_args_py}. check ORBIT_ROOT={orbit_root}")

def _register_envs():
    import gymnasium as gym 
    import isaac_hydra_ext  # noqa: F401
    import isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1  # noqa: F401

def main():
    _ensure_orbit_paths()
    _register_envs()
    from scripts.reinforcement_learning.rsl_rl import play
    play.main()

if __name__ == "__main__":
    main()

