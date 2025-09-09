# isaac_hydra_ext

![Isaac Lab](go1_rough.jpg)

---

A Hydra extension and lightweight runner that connects my training algorithm (APPO/PPO) to NVIDIA **Isaac Sim / Isaac Lab** as an external add-on.  
It lets you launch training via Hydra configs without modifying the trainer.

---

## Features

- **Hydra plugin** that registers this repo’s `conf/` as a config search path.
- **Runner** (`isaac_hydra_ext.appo_runner`) that loads your env/agent configs and starts training (APPO/PPO).
- Configs organized under `conf/` (`env/`, `ppo/`, `logging/`, `checkpoint/`, `experiment/`).
- Installable as an editable package (`pip install -e .`) on host or inside Isaac-Sim container.

---

## Requirements

- Python 3.9–3.11
- `hydra-core >= 1.3`
- Isaac Lab / Isaac Sim (5.0+ recommended) or any Gym env (for the Gym example).

---

## Installation (local)

```bash
git clone https://github.com/<you>/isaac_hydra_ext.git
cd isaac_hydra_ext
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Quick smoke test:

```bash
python -m isaac_hydra_ext.appo_runner env=isaac_go1_nav experiment.name=local_test
```

> If Hydra can’t find configs, ensure the package is actually installed (`pip show isaac-hydra-ext`)—the search-path plugin is registered via entry points on install.

---

## Using inside an Isaac-Sim container

1) **Start** your Isaac-Sim container (example, adjust to your setup):

```bash
docker run --name isaac-sim --rm -it \
  --gpus all --runtime=nvidia --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  nvcr.io/nvidia/isaac-sim:5.0.0 bash
```

2) **Install this extension** inside the container:

```bash
apt-get update && apt-get install -y git
git clone https://github.com/<you>/isaac_hydra_ext.git /root/isaac_hydra_ext
pip install -e /root/isaac_hydra_ext
```

3) **Run training**:

```bash
python -m isaac_hydra_ext.appo_runner env=isaac_go1_nav experiment.name=docker_test
```

You can also call it from the provided script:

```bash
bash /root/isaac_hydra_ext/scripts/run_isaacsim_docker.sh
```

---

## Configuration

Hydra config groups live under `isaac_hydra_ext/conf`:

- `env/` – environment selection (e.g., `isaac_go1_nav.yaml`, `gym_pendulum.yaml`)
- `ppo/` – algorithm hyper-parameters
- `logging/` – loggers, wandb, tensorboard, etc.
- `checkpoint/` – saving/loading policy
- `experiment/` – naming, seeds, paths
- `train_appo.yaml` – top-level defaults / wiring

### Examples

Run Gym pendulum locally:

```bash
python -m isaac_hydra_ext.appo_runner env=gym_pendulum experiment.name=pendulum_debug
```

Override a few params on the fly:

```bash
python -m isaac_hydra_ext.appo_runner \
  env=isaac_go1_nav \
  experiment.name=go1_fast \
  ppo.num_steps=1024 ppo.batch_size=32768 logging.stdout=true
```

Select a different experiment preset:

```bash
python -m isaac_hydra_ext.appo_runner env=isaac_go1_nav experiment=default +seed=0
```

---

## Project layout

```text
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

> Tip: you can regenerate the tree automatically:
>
> ```bash
> sudo apt-get install -y tree
> tree -a -I ".git|.venv|__pycache__|outputs|multirun" > TREE.txt
> ```

---

## How the Hydra plugin works

The package exposes a plugin (`isaac_hydra_ext.plugins.searchpath_plugin.IsaacHydraSearchPathPlugin`) via `entry_points` so that, once installed, Hydra automatically appends this repo’s `conf/` to its search path. You can keep your Isaac-Lab checkout untouched and still resolve configs from this extension.

---

## Troubleshooting

- **“Can’t find config …”**  
  Ensure `pip show isaac-hydra-ext` shows an installed dist **and** you’re running Python from that environment. If running inside Docker, install with `pip install -e /path/to/isaac_hydra_ext`.

- **Mixed Python versions**  
  Isaac-Sim images include their own Python. Use the container’s `pip` to install this package.

- **Permissions**  
  When writing logs/checkpoints, point to a writable dir, e.g. `experiment.output_dir=/root/Documents/runs`.

---

## License

MIT (see `LICENSE`).
