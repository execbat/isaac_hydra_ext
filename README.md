# isaac_hydra_ext

A Hydra extension and lightweight runner that connects my training algorithm (APPO/PPO) to NVIDIA Isaac Sim / Isaac Lab running in an external container. It lets you launch training via Hydra configs without changing the trainer code.

## Installation (local)

```bash
git clone https://github.com/<you>/isaac_hydra_ext.git
cd isaac_hydra_ext
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m isaac_hydra_ext.appo_runner env=isaac_go1_nav experiment.name=local_test

```
isaac_hydra_ext/
├─ README.md
├─ pyproject.toml
├─ scripts/
│  ├─ run_gym_local.sh
│  ├─ run_isaacsim_docker.sh
│  └─ install_inside_container.sh
└─ isaac_hydra_ext/
   ├─ __init__.py
   ├─ appo_runner.py
   ├─ plugins/
   │  ├─ __init__.py
   │  └─ searchpath_plugin.py
   └─ conf/
      ├─ train_appo.yaml
      ├─ env/
      │  ├─ gym_pendulum.yaml
      │  └─ isaac_go1_nav.yaml
      ├─ ppo/
      │  └─ default.yaml
      ├─ logging/
      │  └─ default.yaml
      ├─ checkpoint/
      │  └─ default.yaml
      └─ experiment/
         └─ default.yaml

```

