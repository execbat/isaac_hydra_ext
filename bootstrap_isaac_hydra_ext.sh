#!/usr/bin/env bash
# bootstrap_isaac_hydra_ext.sh
# Создаёт подключаемый пакет Hydra для Isaac (локально в ./source/isaac_hydra_ext)

set -euo pipefail

BASE_DIR="source/isaac_hydra_ext"
PKG_DIR="$BASE_DIR/isaac_hydra_ext"

echo ">>> Creating directories..."
mkdir -p "$PKG_DIR"/{plugins,conf/task,conf/agent}

echo ">>> Writing __init__.py ..."
cat > "$PKG_DIR/__init__.py" <<'PY'
"""
isaac_hydra_ext: локальное расширение Hydra для Isaac-Lab / Isaac-Sim.
Здесь можно хранить свои конфиги (conf/*) и код (награды, термы и пр.).
"""
__all__ = []
PY

echo ">>> Writing rewards.py ..."
cat > "$PKG_DIR/rewards.py" <<'PY'
import torch

def masked_dwell_bonus(
    env,
    mask_name: str = "dof_mask",
    eps: float = 0.04,          # допуск по позе (норм. [-1,1])
    vel_eps: float = 0.03,      # допуск по скорости (норм.)
    hold_steps: int = 8,        # после скольки подряд «ОК» шагов начать платить бонус
    bonus: float = 0.5,         # базовый бонус (когда выдержали hold_steps)
    growth: float = 0.1,        # добавка за каждый следующий «ОК» шаг
    max_bonus: float = 3.0,     # верхняя планка
) -> torch.Tensor:
    """
    Платит всё больше, чем дольше ВСЕ активные DOF (mask==1) удерживаются в допуске
    по позиции и скорости. Счётчик сбрасывается, как только вышли из окна «ОК».
    """
    robot = env.scene["robot"]
    q, qd = robot.data.joint_pos, robot.data.joint_vel
    device, dtype = q.device, q.dtype

    # нормировка в [-1,1]
    qmin = robot.data.soft_joint_pos_limits[..., 0]
    qmax = robot.data.soft_joint_pos_limits[..., 1]
    mid  = 0.5 * (qmin + qmax)
    rng  = (qmax - qmin).clamp_min(1e-6)
    qn, qdn = 2.0 * (q - mid) / rng, 2.0 * qd / rng

    # цель и маска
    tgt = torch.as_tensor(
        env.command_manager.get_term("target_joint_pose").command,
        dtype=dtype, device=device
    )
    msk = (env.command_manager.get_term(mask_name).command > 0.5)

    # «ОК» одновременно по позе и скорости (только по активным DOF)
    ok_pos = (qn - tgt).abs() <= eps
    ok_vel = qdn.abs() <= vel_eps
    ok_all_j = torch.where(msk, ok_pos & ok_vel, torch.ones_like(ok_pos, dtype=torch.bool))
    ok_all = ok_all_j.all(dim=1)  # (N,)

    # держим счётчик по env
    if not hasattr(env, "_dwell_cnt"):
        env._dwell_cnt = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._dwell_cnt = torch.where(ok_all, env._dwell_cnt + 1, torch.zeros_like(env._dwell_cnt))

    # обнуляем счётчики на ресетах
    resets = env.termination_manager.terminated | env.termination_manager.time_outs
    if resets.any():
        env._dwell_cnt[resets] = 0

    # бонус: начинается после hold_steps и растёт на growth каждый шаг, ограничен max_bonus
    extra = (env._dwell_cnt - hold_steps).clamp_min(0).to(q.dtype)
    r = torch.clamp(bonus + growth * extra, max=max_bonus)
    r = torch.where(env._dwell_cnt >= hold_steps, r, torch.zeros_like(r))
    return r
PY

echo ">>> Writing plugins/searchpath_plugin.py ..."
cat > "$PKG_DIR/plugins/searchpath_plugin.py" <<'PY'
"""
Hydra SearchPath плагин: добавляет pkg://isaac_hydra_ext.conf в поисковый путь,
чтобы Hydra видела YAML-файлы из этого пакета без ручных +hydra.searchpath.
"""
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.config_search_path import ConfigSearchPath

class IsaacHydraExtSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Добавляем корень конфигов пакета (папку conf)
        search_path.append("pkg://isaac_hydra_ext.conf")
PY

echo ">>> Writing plugins/__init__.py ..."
cat > "$PKG_DIR/plugins/__init__.py" <<'PY'
# package marker
PY

echo ">>> Writing conf/task/go1_nav_v1.yaml ..."
cat > "$PKG_DIR/conf/task/go1_nav_v1.yaml" <<'YAML'
# Пример таска, который расширяет базовый (переопредели имя при необходимости)
# В Isaac Lab чаще группы называются /task, /agent, /env и т.д.
defaults:
  - override /task: go1_flat   # замени на актуальный базовый конфиг из isaaclab_tasks
  - _self_

task:
  mdp:
    rewards:
      terms:
        dwell_bonus:
          func: isaac_hydra_ext.rewards.masked_dwell_bonus  # путь к вашей функции
          weight: 1000.0
          params:
            mask_name: "dof_mask"
            eps: 0.04
            vel_eps: 0.03
            hold_steps: 2
            bonus: 0.5
            growth: 0.1
            max_bonus: 3.0
YAML

echo ">>> Writing conf/agent/rsl_rl.yaml ..."
cat > "$PKG_DIR/conf/agent/rsl_rl.yaml" <<'YAML'
# Опциональный оверрайд агента (пример-рыба)
# Обычно в Isaac Lab уже есть agent=rsl_rl; этот файл можно использовать для локальных правок.
agent:
  max_iterations: 100000  # пример параметра
  # добавь свои поля/оверрайды при необходимости
YAML

echo ">>> Writing pyproject.toml ..."
cat > "$BASE_DIR/pyproject.toml" <<'TOML'
[project]
name = "isaac_hydra_ext"
version = "0.1.0"
description = "Local Hydra extension for Isaac (configs + rewards)"
authors = [{name="You"}]
requires-python = ">=3.10"
dependencies = []  # при необходимости добавь зависимости (hydra-core уже в Isaac)

[build-system]
requires = ["setuptools>=62"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["isaac_hydra_ext", "isaac_hydra_ext.plugins"]

[tool.setuptools.package-data]
isaac_hydra_ext = ["conf/**/*"]

# Регистрируем Hydra SearchPath плагин
[project.entry-points."hydra.searchpath"]
isaac_hydra_ext = "isaac_hydra_ext.plugins.searchpath_plugin:IsaacHydraExtSearchPathPlugin"
TOML

echo ">>> Done."
echo
echo "Установка (в активном env Isaac):"
echo "  export PYTHONPATH=\$PWD/source:\$PYTHONPATH"
echo "  pip install -e source/isaac_hydra_ext"
echo
echo "Проверка запуска (пример):"
echo "  python scripts/reinforcement_learning/rsl_rl/train.py task=go1_nav_v1 --cfg job"

