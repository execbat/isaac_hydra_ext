# isaac_hydra_ext

![Isaac Lab](go1_rough.jpg)

---

A Hydra extension and lightweight runner that connects my training algorithm (APPO/PPO) to NVIDIA **Isaac Sim / Isaac Lab** as an external add-on.  
It lets you launch training via Hydra configs without modifying the trainer.

---

## APPO at a glance

![APPO multiprocess demo](demo.png)

**Architecture (what runs where):**

- **Main process (GPU)** — holds the **Policy (Actor)** and **Critic**, does optimization and logging.
- **Workers (CPU)** — `W = num_workers` independent OS processes.  
  Each worker launches its own headless **Isaac-Lab** vectorized environment with `B = envs_per_worker` envs.
- **Rollout Buffer (CPU↔GPU boundary)** — a shared “place” where workers drop trajectories and the GPU reads mini-batches.

**Data flow:**

1. Each worker (CPU) collects `T = steps_per_env` steps from **each** of its `B` envs → pushes trajectories to the **Buffer**.
2. The **Main (GPU)** pulls mini-batches from the Buffer and runs several SGD passes to update the Policy/Critic.
3. The Main **broadcasts updated weights** back to **every** worker.  
   Workers do **not** talk to each other; each only talks to the Main/Buffer.

**Sizes per learning iteration:**

- Samples produced per worker: `B × T`.
- Total rollout size (buffer):  
  `N_buffer = W × B × T` (number of time–env samples in the Buffer).
- Mini-batching on the GPU:
  - `batch_size = M` (from config)
  - `num_minibatches_per_epoch = ceil(N_buffer / M)`
  - `total_gradient_steps_per_iteration = update_epochs × num_minibatches_per_epoch`

### Key knobs (from your config)

```yaml
# Collected on CPU by workers
steps_per_env: 24        # T, steps per environment before each policy update
num_workers: 4           # W, how many CPU worker processes
envs_per_worker: 128     # B, vectorized envs per worker

# Applied on GPU by the main process
update_epochs: 4         # how many passes over the same Buffer per iteration
batch_size: 2048         # SGD minibatch size for the Policy/Critic updates
```

**Intuition:**

- Increase **T** (steps_per_env) → longer rollouts, better advantage estimates, but the policy is “older” by the end of collection.
- Increase **W** or **B** → bigger `N_buffer` per iteration (more data, more stable gradients, more VRAM/CPU needed).
- Increase **batch_size** → fewer optimizer steps per epoch but smoother gradients.
- Increase **update_epochs** → squeeze more learning signal out of the same data; watch KL to avoid over-fitting on on-policy data.

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
docker run --name isaac-sim --rm -it   --gpus all --runtime=nvidia --network=host   -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y   -v ~/docker/isaac-sim/documents:/root/Documents:rw   nvcr.io/nvidia/isaac-sim:5.0.0 bash
```

2) **Install this extension** inside the container:

```bash
apt-get update && apt-get install -y git
git clone https://github.com/<you>/isaac_hydra_ext.git /root/isaac_hydra_ext
pip install -e /root/isaac_hydra_ext
```

3) **Run training**:

```bash
./isaaclab.sh -p -m isaac_hydra_ext.scripts.reinforcement_learning.appo.train \
 --task Isaac-Velocity-Sber-Unitree-Go1-v0 --num_envs 1 --headless 
```
---

## Configuration


### Examples


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
