# AGENTS.md

## Project Overview
This repository implements the current two-stage RL pipeline for a hip exoskeleton:

- Stage 1: fixed local DEP-RL reference walker
- Stage 2: SB3 PPO exoskeleton policy on top of the fixed walker

---

## Directory Structure

rl/
  baselines/          # Canonical local baseline loading
  baselines_DEPRL/    # Local DEP-RL reference walker files
  envs/               # Stage 2 environment
  scripts/            # Authoritative Stage 1 / Stage 2 entrypoints
  stage1_compare/     # Comparison / evaluation / rendering only
  stage2_analysis/    # Plotting / analysis only
  rl_output/          # Organized outputs
    stage1/
    stage2/
    analysis/

---

## Script Roles (STRICT)

### Training scripts
- train_*.py
- Responsible for:
  - creating environments
  - training PPO
  - saving models

### Evaluation scripts
- eval_*.py
- Responsible for:
  - loading trained models
  - running episodes
  - saving metrics (JSON/CSV)
- MUST NOT contain training logic

### Comparison scripts
- compare_*.py
- Read saved outputs and summarize results
- MUST NOT run environments or training

### Rendering / plotting scripts
- render_*.py = video only
- plot_*.py = plotting only

---

## Authoritative Entry Points

- Stage 1 reference evaluation:
  `python rl/scripts/run_stage1_reference.py`
- Stage 2 training:
  `python rl/scripts/train_exo_stage2.py`
- Stage 2 evaluation:
  `python rl/scripts/eval_exo_stage2.py`

Comparison / analysis:
- `python rl/stage1_compare/eval_stage1_deprl.py`
- `python rl/stage1_compare/compare_stage1_results.py`
- `python rl/stage1_compare/render_stage1_reference.py`
- `python rl/stage2_analysis/plot_torque_vs_gait.py`

Note:
- legacy PPO comparison scripts may require user-supplied checkpoints because the old PPO experiment artifacts are not part of the cleaned repo checkpoint

---

## Rules for Agents

1. DO NOT mix training and evaluation in the same script
2. DO NOT overwrite result files — always use unique filenames
3. Keep output paths under `rl/rl_output/stage1`, `rl/rl_output/stage2`, or `rl/rl_output/analysis`
4. Preserve reproducibility: local DEP-RL files are the canonical Stage 1 walker source
5. Prefer small direct scripts over duplicated experiments or wrappers

---

## Definition of Done

Changes are complete when:
- scripts run without errors
- outputs are saved under the standardized `rl/rl_output/` layout
- no existing functionality is broken
