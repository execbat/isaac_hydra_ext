# isaac_hydra_ext/scripts/run_train_with_ext.py
# -*- coding: utf-8 -*-
"""
Start RSL-RL training + save/restore the environment state:
- weights of all active reward/penalty terms (RewardManager)
- ALL parameters of ALL active commands (CommandManager), not just base_velocity
- common_step_counter step counter (to ensure schedulers continue from the correct phase)
"""

from __future__ import annotations

import os
import sys
import json
import math
import types
import dataclasses
from typing import Any, Dict, Optional, Sequence


# -----------------------------------------------------------------------------
# Bootstrapping ORBIT/RSL-RL paths and registering our tasks
# -----------------------------------------------------------------------------
def _ensure_orbit_paths():
    orbit_root = os.environ.get("ORBIT_ROOT", os.getcwd())
    if orbit_root not in sys.path:
        sys.path.insert(0, orbit_root)

    rsl_rl_dir = os.path.join(orbit_root, "scripts", "reinforcement_learning", "rsl_rl")
    if rsl_rl_dir not in sys.path:
        sys.path.insert(0, rsl_rl_dir)

    cli_args_py = os.path.join(rsl_rl_dir, "cli_args.py")
    if not os.path.exists(cli_args_py):
        raise RuntimeError(
            f"cli_args.py not found: {cli_args_py}. Check ORBIT_ROOT={orbit_root}"
        )


def _register_envs():
    # register the extension/tasks so that train can import them.
    import isaac_hydra_ext  # noqa: F401
    import isaac_hydra_ext.source.isaaclab_tasks.manager_based.locomotion.velocity.config.go1  # noqa: F401


# -----------------------------------------------------------------------------
# Useful utilities for wrappers/manager access
# -----------------------------------------------------------------------------
def _unwrap_to_base_env(env):
    """extract the base env from the vectorized/wrapped instances."""
    env = getattr(env, "unwrapped", env)

    get_env = getattr(env, "get_env", None)
    if callable(get_env):
        try:
            env0 = get_env(0)
            if env0 is not None:
                env = getattr(env0, "unwrapped", env0)
        except Exception:
            pass

    if hasattr(env, "envs") and getattr(env, "envs"):
        try:
            e0 = getattr(env.envs[0], "unwrapped", env.envs[0])
            if e0 is not None:
                env = e0
        except Exception:
            pass
    return env


def _get_reward_manager(env) -> Optional[Any]:
    """RewardManager is available as env.reward_manager or env.task.reward_manager."""
    base = _unwrap_to_base_env(env)
    for holder in (base, getattr(base, "task", None)):
        if holder is None:
            continue
        rm = getattr(holder, "reward_manager", None) or getattr(holder, "_reward_manager", None)
        # sanity: должен иметь публичный API
        if rm is not None and hasattr(rm, "active_terms") and hasattr(rm, "get_term_cfg") and hasattr(rm, "set_term_cfg"):
            return rm
    return None


def _get_command_manager(env) -> Optional[Any]:
    """CommandManager is available as env.command_manager or env.task.command_manager."""
    base = _unwrap_to_base_env(env)
    for holder in (base, getattr(base, "task", None)):
        if holder is None:
            continue
        cm = getattr(holder, "command_manager", None) or getattr(holder, "_command_manager", None)
        if cm is None:
            continue

        if (hasattr(cm, "get_term_cfg") and hasattr(cm, "set_term_cfg")) or hasattr(cm, "_terms"):
            return cm
    return None


# -----------------------------------------------------------------------------
# Saving/Loading Reward WEIGHTS (via RewardManager API)
# -----------------------------------------------------------------------------
def _collect_reward_weights(env) -> Dict[str, float]:
    rm = _get_reward_manager(env)
    out: Dict[str, float] = {}
    if rm is None:
        return out
    try:
        names = list(getattr(rm, "active_terms", []) or [])
    except Exception:
        names = []
    for name in names:
        try:
            cfg = rm.get_term_cfg(name)  # dataclass-like cfg
            w = getattr(cfg, "weight", None)
            if isinstance(w, (int, float)) and math.isfinite(float(w)):
                out[name] = float(w)
        except Exception:
            pass
    return out


def _apply_reward_weights(env, mapping: Dict[str, float]) -> int:
    rm = _get_reward_manager(env)
    if rm is None:
        return 0
    applied = 0
    for name, w in (mapping or {}).items():
        try:
            cfg = rm.get_term_cfg(name)
            if hasattr(cfg, "weight"):
                cfg.weight = float(w)
                rm.set_term_cfg(name, cfg)
                applied += 1
        except Exception:
            pass
    return applied


# -----------------------------------------------------------------------------
# Save/load ALL parameters of ALL active commands (CommandManager API)
# -----------------------------------------------------------------------------
_SIMPLE_JSON_TYPES = (type(None), bool, int, float, str)


def _is_jsonable_simple(x: Any) -> bool:
    return isinstance(x, _SIMPLE_JSON_TYPES)


def _to_jsonable(x: Any) -> Any:
    """
    convert the cfg to a JSON-friendly format:
    - dataclass -> dictionary by fields (recursively)
    - tuple -> list (recursively)
    - list/dictionary -> element-by-element
    - callables/modules/types/tensors/ndarrays — skip
    """
    # simple
    if _is_jsonable_simple(x):
        return x

    # tensors/ndarrays — not serializable; convert to float/list if 0-D or 1-D scalar.
    try:
        import torch  # noqa
        if isinstance(x, torch.Tensor):
            if x.ndim == 0:
                v = x.item()
                return float(v) if isinstance(v, (int, float)) else v
            return None  # skip complicated
    except Exception:
        pass
    try:
        import numpy as np  # noqa
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                v = x.item()
                return float(v) if isinstance(v, (int, float)) else v
            return None
    except Exception:
        pass

    # dataclass
    if dataclasses.is_dataclass(x):
        out = {}
        for f in dataclasses.fields(x):
            key = f.name
            try:
                val = getattr(x, key)
            except Exception:
                continue
            j = _to_jsonable(val)
            if j is not None:
                out[key] = j
        return out

    # tuple -> list
    if isinstance(x, tuple):
        conv = [_to_jsonable(v) for v in x]
        if any(v is not None for v in conv):
            return [v for v in conv if v is not None]
        return []

    # list
    if isinstance(x, list):
        conv = [_to_jsonable(v) for v in x]
        return [v for v in conv if v is not None]

    # dict
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            j = _to_jsonable(v)
            if j is not None:
                out[str(k)] = j
        return out

    # skipping explicit non-serializable ones
    if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, types.ModuleType, type)):
        return None

    # other objects - try __dict__ 
    if hasattr(x, "__dict__"):
        out = {}
        for k, v in x.__dict__.items():
            j = _to_jsonable(v)
            if j is not None:
                out[str(k)] = j
        return out

    return None


def _from_jsonable_set(cfg_obj: Any, data: Any) -> int:
    """ 
    apply the JSON parameters back to the CFG (recursively, where possible).
    return the number of applied fields.
    """
    applied = 0
    if cfg_obj is None or data is None:
        return applied

    # dataclass: we walk through the fields
    if dataclasses.is_dataclass(cfg_obj) and isinstance(data, dict):
        for f in dataclasses.fields(cfg_obj):
            key = f.name
            if key not in data:
                continue
            try:
                cur_val = getattr(cfg_obj, key)
            except Exception:
                continue
            new_val = data[key]
            # recursively in dataclass
            if dataclasses.is_dataclass(cur_val) and isinstance(new_val, dict):
                applied += _from_jsonable_set(cur_val, new_val)
                continue
            # tuple from list
            if isinstance(cur_val, tuple) and isinstance(new_val, list):
                try:
                    t = tuple(new_val)
                    setattr(cfg_obj, key, t)
                    applied += 1
                    continue
                except Exception:
                    pass
            # list
            if isinstance(cur_val, list) and isinstance(new_val, list):
                try:
                    setattr(cfg_obj, key, list(new_val))
                    applied += 1
                    continue
                except Exception:
                    pass
            # simple types, incl None/bool/int/float/str
            if _is_jsonable_simple(new_val):
                try:
                    setattr(cfg_obj, key, new_val)
                    applied += 1
                    continue
                except Exception:
                    pass
            # dict -> dict
            if isinstance(cur_val, dict) and isinstance(new_val, dict):
                try:
                    cur_val.update(new_val)
                    applied += 1
                    continue
                except Exception:
                    pass
            # otherwise, we try to replace it entirely (best effort)
            try:
                setattr(cfg_obj, key, new_val)
                applied += 1
            except Exception:
                pass
        return applied

    # не dataclass: best-effort by dict
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                old = getattr(cfg_obj, k, None)
                if dataclasses.is_dataclass(old) and isinstance(v, dict):
                    applied += _from_jsonable_set(old, v)
                else:
                    setattr(cfg_obj, k, v)
                    applied += 1
            except Exception:
                pass
        return applied

    return applied


def _collect_command_cfgs(env) -> dict:
    """
    A complete JSON-friendly cfg snapshot for all active commands.
    Supports both the public API (get_term_cfg) and fallback via _terms[name].cfg.
    """
    cm = _get_command_manager(env)
    out = {}
    if cm is None:
        return out

    # 1) ter, names
    try:
        if hasattr(cm, "active_terms"):
            names = list(getattr(cm, "active_terms") or [])
        elif hasattr(cm, "_terms"):
            names = list(getattr(cm, "_terms").keys())
        else:
            names = []
    except Exception:
        names = []

    # 2) getting cfg по for every term
    for name in names:
        cfg_obj = None
        try:
            if hasattr(cm, "get_term_cfg"):
                cfg_obj = cm.get_term_cfg(name)
            elif hasattr(cm, "_terms"):
                term = cm._terms.get(name, None)
                cfg_obj = getattr(term, "cfg", None) if term is not None else None
        except Exception:
            cfg_obj = None

        if cfg_obj is None:
            continue

        try:
            data = _to_jsonable(cfg_obj)
            if isinstance(data, dict) and data:
                out[name] = data
        except Exception:
            # If serialization of a specific cfg failed, we skip only it
            pass

    return out


def _apply_command_cfgs(env, mapping: dict) -> int:
    """
    reapply the saved command parameters.
    If there's no set_term_cfg, we write them directly to term.cfg (fallback).
    """
    cm = _get_command_manager(env)
    if cm is None:
        return 0

    applied_terms = 0
    for name, data in (mapping or {}).items():
        cfg_obj = None
        try:
            if hasattr(cm, "get_term_cfg"):
                cfg_obj = cm.get_term_cfg(name)
            elif hasattr(cm, "_terms"):
                term = cm._terms.get(name, None)
                cfg_obj = getattr(term, "cfg", None) if term is not None else None
        except Exception:
            cfg_obj = None

        if cfg_obj is None:
            continue

        changed_fields = 0
        try:
            changed_fields = _from_jsonable_set(cfg_obj, data)
            if hasattr(cm, "set_term_cfg"):
                cm.set_term_cfg(name, cfg_obj)
            elif hasattr(cm, "_terms"):
                
                term = cm._terms.get(name, None)
                if term is not None:
                    setattr(term, "cfg", cfg_obj)
            if changed_fields > 0:
                applied_terms += 1
        except Exception:
            pass

    return applied_terms


# -----------------------------------------------------------------------------
# Saving/loading the common_step_counter counter (for continuity of schedulers)
# -----------------------------------------------------------------------------
def _get_common_step_counter(env) -> Optional[int]:
    base = _unwrap_to_base_env(env)
    for holder in (base, getattr(base, "task", None)):
        if holder is None:
            continue
        val = getattr(holder, "common_step_counter", None)
        if isinstance(val, (int, float)):
            return int(val)
    return None


def _set_common_step_counter(env, value: int) -> bool:
    base = _unwrap_to_base_env(env)
    ok = False
    for holder in (base, getattr(base, "task", None)):
        if holder is None:
            continue
        if hasattr(holder, "common_step_counter"):
            try:
                setattr(holder, "common_step_counter", int(value))
                ok = True
            except Exception:
                pass
    return ok


# -----------------------------------------------------------------------------
# JSON paths next to the model
# -----------------------------------------------------------------------------
def _rewards_json_for_model(path_pt: str) -> str:
    root, _ = os.path.splitext(path_pt)
    return f"{root}__rewards.json"


def _envstate_json_for_model(path_pt: str) -> str:
    root, _ = os.path.splitext(path_pt)
    return f"{root}__envstate.json"


# -----------------------------------------------------------------------------
# Restoring tot_timesteps after resume (for RSL-RL logic/graphs)
# -----------------------------------------------------------------------------
def _init_tot_timesteps_after_resume(runner):
    try:
        per_iter = runner.num_steps_per_env * runner.env.num_envs * getattr(runner, "gpu_world_size", 1)
        runner.tot_timesteps = int(per_iter * runner.current_learning_iteration)
        if getattr(runner, "writer", None) and not getattr(runner, "disable_logs", False):
            runner.writer.add_scalar("Perf/total_fps_resume_stub", 0, runner.current_learning_iteration)
        print(f"[resume] tot_timesteps set to {runner.tot_timesteps} "
              f"(iter={runner.current_learning_iteration}, per_iter={per_iter})")
    except Exception as e:
        print(f"[resume] can't set tot_timesteps: {e}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    _ensure_orbit_paths()
    _register_envs()

   
    try:
        from rsl_rl.runners.on_policy_runner import OnPolicyRunner
    except Exception:
        from rsl_rl.runners import OnPolicyRunner

    # patch save/load to save/restore the environment state
    __orig_save = OnPolicyRunner.save
    __orig_load = OnPolicyRunner.load

    def _patched_save(self: OnPolicyRunner, path: str, infos=None):
        ret = __orig_save(self, path, infos)

        # 1) rewar weights
        try:
            weights = _collect_reward_weights(self.env)
            json_path = _rewards_json_for_model(path)
            if weights:
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(weights, f, ensure_ascii=False, indent=2)
                print(f"[rewards-json] saved {len(weights)} weights -> {os.path.basename(json_path)}")
            else:
                print("[rewards-json] nothing to save")
        except Exception as e:
            print(f"[rewards-json] save skipped: {e}")

        # 2) cmd states + counter
        try:
            side_path = _envstate_json_for_model(path)
            state = {
                "common_step_counter": _get_common_step_counter(self.env),
                "commands_cfg": _collect_command_cfgs(self.env),
            }
            os.makedirs(os.path.dirname(side_path), exist_ok=True)
            with open(side_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"[envstate-json] saved -> {os.path.basename(side_path)} "
                  f"(cmd_terms={len(state.get('commands_cfg', {}))}, "
                  f"common_step={state.get('common_step_counter')})")
        except Exception as e:
            print(f"[envstate-json] save skipped: {e}")

        return ret

    def _patched_load(self: OnPolicyRunner, path: str, load_optimizer: bool = True):
        infos = __orig_load(self, path, load_optimizer)

        # 1) restore reward weights
        try:
            json_path = _rewards_json_for_model(path)
            if os.path.isfile(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    mapping = json.load(f)
                applied = _apply_reward_weights(self.env, mapping)
                print(f"[rewards-json] restored {applied} weights from {os.path.basename(json_path)}")
            else:
                print("[rewards-json] no JSON next to checkpoint (skip)")
        except Exception as e:
            print(f"[rewards-json] load skipped: {e}")

        # 2) restore cmd + couter
        try:
            side_path = _envstate_json_for_model(path)
            if os.path.isfile(side_path):
                with open(side_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                # счётчик
                if "common_step_counter" in state and state["common_step_counter"] is not None:
                    _set_common_step_counter(self.env, int(state["common_step_counter"]))
                # команды
                if "commands_cfg" in state and isinstance(state["commands_cfg"], dict):
                    applied_terms = _apply_command_cfgs(self.env, state["commands_cfg"])
                    print(f"[envstate-json] restored command cfgs ({applied_terms} terms)")
            else:
                print("[envstate-json] no sidecar JSON next to checkpoint (skip)")
        except Exception as e:
            print(f"[envstate-json] load skipped: {e}")

        # rsl-rl counters
        _init_tot_timesteps_after_resume(self)
        return infos

    OnPolicyRunner.save = _patched_save
    OnPolicyRunner.load = _patched_load

    # run standard train
    from scripts.reinforcement_learning.rsl_rl import train
    train.main()


if __name__ == "__main__":
    main()

