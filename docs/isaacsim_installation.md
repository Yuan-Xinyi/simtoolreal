# IsaacSim (Isaac Lab) installation

The `isaacsimenvs/` package runs the SimToolReal task in Isaac Sim via Isaac Lab. It requires **Python 3.11** (Isaac Sim 5.x / Isaac Lab 2.3.x requirement) and lives in a **second venv** at `.venv_isaacsim/`, separate from the Python 3.8 Isaac Gym venv that [isaacgym_installation.md](isaacgym_installation.md) sets up. The two environments are independent — never install Isaac Gym packages into `.venv_isaacsim` or vice versa.

## Prerequisites

- Python 3.11
- NVIDIA GPU with driver >= 525.60
- CUDA 12+
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

## Install

```bash
uv venv .venv_isaacsim --python 3.11

# PyTorch for CUDA 12.6 (Isaac Sim 5.x is built against this)
uv pip install --python .venv_isaacsim/bin/python torch --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games + inference deps.
# Do NOT skip this: train.py / eval do `from rl_games.torch_runner import Runner`,
# and the repo's `rl_games/` wrapper dir on sys.path (from the editable root
# install below) otherwise resolves `import rl_games` to an empty namespace
# package — imports of `rl_games.*` submodules then fail at runtime.
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/
uv pip install --python .venv_isaacsim/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro "imageio[ffmpeg]" wandb termcolor

# Isaac Lab + Isaac Sim (~15 GB download; first launch builds RTX shaders, takes ~2-5 min)
uv pip install --python .venv_isaacsim/bin/python \
  "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# Offline collision decomposition (CoACD) + tyro CLI fix. Install AFTER isaaclab so it
# wins the resolution: tyro.cli (used by dextoolbench/generate_collision_meshes.py) needs
# NoExtraItems from typing_extensions>=4.13, but the isaaclab install pulls in 4.12.2.
# typing_extensions 4.15 is verified compatible with isaaclab 2.3.2.post1.
uv pip install --python .venv_isaacsim/bin/python coacd "typing_extensions>=4.13"

# Register repo-local packages (isaacsimenvs, isaacgymenvs, deployment, ...)
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
```

`--no-deps` on the last line is required: the root `pyproject.toml` pins `numpy==1.23.0`, `warp-lang==0.10.1`, and `isaacgym-stubs`, which all conflict with Python 3.11 / Isaac Sim. The repo packages themselves install cleanly.

Keep the version pins exact (`isaaclab==2.3.2.post1`, torch 2.7.x+cu126) — newer Isaac Lab releases change `DirectRLEnv` / `UrdfConverter` APIs that `isaacsimenvs` depends on.

**`numpy<2` guard.** `isaaclab` requires `numpy<2`. If you ever (re)install
`rl_games` *after* isaaclab, its `opencv-python` dep can resolve to opencv 5.x,
which drags in `numpy>=2` and silently breaks Isaac Lab (`import isaaclab` then
fails). If that happens, pin back down:

```bash
uv pip install --python .venv_isaacsim/bin/python "numpy<2" "opencv-python<5"
```

Verify. Note the import order: `isaacsimenvs` (and any `isaaclab.envs` /
`isaaclab.sim` sub-namespace) only resolves *after* `AppLauncher` boots — see
[Gotchas](#gotchas) — so the check has to launch a headless app before importing
the repo package, not just `import` it in a bare `python -c`:

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python - <<'PY'
import argparse, os, sys
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
app = AppLauncher(parser.parse_args(["--headless"])).app

import isaaclab, isaaclab.envs  # noqa: F401
import isaacsimenvs             # noqa: F401  triggers gym.register(...)

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("isaaclab:", isaaclab.__file__)
print("VERIFY OK")

del app
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)  # Kit teardown can hang; force-exit
PY
```

A bare `.venv_isaacsim/bin/python -c "import torch, isaaclab, isaacsimenvs"`
fails with `ModuleNotFoundError: No module named 'isaaclab.envs'` — that's the
import-order gotcha, not a broken install.

## Running

Non-interactive launches need the EULA env var:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

Smoke tests (one Kit boot per process, ~1-2 min each; run individually, not via pytest):

```bash
.venv_isaacsim/bin/python isaacsimenvs/tests/test_load_isaacsim.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_gym_register.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 10
```

Training:

```bash
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --headless
```

## Gotchas

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`, `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call `os._exit(0)` rather than waiting for a clean Kit teardown.
- **First SimToolReal startup is slow**: object URDFs are procedurally generated and converted to USD per run (no cache); startup scales with `num_assets_per_type`.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo processes to finish before launching another.
