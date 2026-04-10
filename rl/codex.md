

# RL Pipeline Refactor Plan (Exoskeleton Project)

## Goal

Refactor the current monolithic script (`two_stage_exo.py`) into a clean, modular structure without changing core logic.

We want to separate:
- Walker loading (Stage 1 baseline)
- Exoskeleton environment
- Training (Stage 2)
- Evaluation

## Target Structure

    rl/
      baselines/
        load_reference_walker.py

      envs/
        exo_with_walker_env.py

      scripts/
        train_exo_stage2.py
        eval_exo_stage2.py

      notebooks/
        stage2_analysis.ipynb

      rl_output/

## Design Principles

- Do not change algorithm logic
- Do not rewrite reward or physics
- Only reorganize code
- Keep CPU compatibility
- Preserve debug prints and diagnostics
- Use MyoSuite’s gym wrapper, not plain gym

## Step 1 — Walker Loader

**File:** `rl/baselines/load_reference_walker.py`

```python
from stable_baselines3 import PPO


def load_reference_walker(path):
    return PPO.load(path, device="cpu")
```

Purpose:
- Load pretrained walker
- Replace Stage 1 training

## Step 2 — Environment Module

**File:** `rl/envs/exo_with_walker_env.py`

Move class `ExoWithWalkerSB3` from `two_stage_exo.py` into this file.

### Required changes

Replace:

```python
self.walker = PPO.load(walker_path, device="cpu")
```

with:

```python
from rl.baselines.load_reference_walker import load_reference_walker
self.walker = load_reference_walker(walker_path)
```

### Keep all methods unchanged

- `__init__`
- `_map_joints`
- `_map_root`
- `_measure_baseline`
- `_estimate_phase`
- `_get_obs`
- `_compute_reward`
- `_terminated`
- `reset`
- `step`
- `close`

## Step 3 — Training Script

**File:** `rl/scripts/train_exo_stage2.py`

### Imports

```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from rl.envs.exo_with_walker_env import ExoWithWalkerSB3
import os
import time
```

### Constant

```python
OUT_DIR = "rl_output"
```

### Main function

```python
def train_exo(walker_path, total_timesteps=1_000_000):
```

### Responsibilities

- Create train and eval environments
- Wrap with `Monitor`
- Use the same PPO hyperparameters as original Stage 2
- Add:
  - `CheckpointCallback`
  - `EvalCallback`
  - TensorBoard logging

### Save model

Save to:

`rl_output/exo_policy.zip`

### CLI entry

```python
if __name__ == "__main__":
    walker_path = "rl_output/walker_policy"
    train_exo(walker_path)
```

## Step 4 — Evaluation Script

**File:** `rl/scripts/eval_exo_stage2.py`

### Purpose

Compare:
- walker-only
- walker + exo

### Per-step logging

- reward
- effort
- torque
- termination reason

### Output

Print:
- mean reward
- mean effort
- mean torque
- episode length

Do not add plotting here.

## Step 5 — Clean Original File

Modify `two_stage_exo.py`.

### Remove

- `ExoWithWalkerSB3` class
- Stage 2 training logic
- evaluation utilities

### Recommended replacement

```python
# Deprecated: refactored into modular pipeline
# See rl/scripts/train_exo_stage2.py
```

### Optional alternative

Keep only Stage 1 walker training if needed.

## Important Notes

### Use MyoSuite gym wrapper

```python
from myosuite.utils import gym as myogym
```

Do not use:

```python
import gym
```

### Gymnasium API compatibility

```python
obs, reward, terminated, truncated, info = env.step(...)
```

### Do not break

- `qfrc_applied` torque injection
- baseline effort computation
- debug prints
- termination diagnostics

## Final Check

After refactor, these must work:

```bash
python rl/scripts/train_exo_stage2.py
python rl/scripts/eval_exo_stage2.py
```

## Future Extensions

After this refactor:
- Stage 2.5 → co-adaptation (unfreeze walker)
- Add locomotion modes:
  - incline
  - stairs
- Add classifier:
  - mode-conditioned policy

## Summary

Old:
- `two_stage_exo.py` with everything mixed together

New:
- walker loader
- env wrapper
- training
- evaluation

Cleaner, more scalable, and better aligned with the research workflow.