# AGENTS.md

## Project Overview

This repository is now a handoff repository for exploratory hip-exoskeleton assistance work.

It preserves:

- historical OpenSim/Moco result artifacts
- the surviving two-stage MyoSuite/DEP-RL RL path
- a small derived mocap-reference snapshot used by the stage-2 environment
- archived exploratory branches that should no longer be treated as active development paths

The current RL pipeline is still:

- Stage 1: fixed local DEP-RL reference walker
- Stage 2: SB3 PPO exoskeleton policy on top of the frozen walker

Important:

- do not present this repo as a completed or validated exoskeleton-control solution
- do not move archived material back into the active path unless explicitly asked

## Directory Structure

```text
docs/               # Handoff docs, cleanup record, commands, next steps
opensim/            # Preserved OpenSim/Moco artifacts and context
rl/                 # Surviving active RL code and outputs
data/               # Active derived data snapshots used by the surviving RL path
outputs/            # Top-level placeholder; active RL still writes inside rl/rl_output
scripts/            # Top-level placeholder; active commands still live under rl/
archived/           # Exploratory, stalled, duplicate, or deprecated material
```

## Active RL Structure

```text
rl/
  baselines/        # Canonical local baseline loading helpers
  baselines_DEPRL/  # Canonical local DEP-RL reference walker files
  envs/             # Stage 2 environment
  reference/        # Mocap-reference helpers used by Stage 2
  scripts/          # Authoritative Stage 1 / Stage 2 entrypoints
  stage1_compare/   # Evaluation and rendering helpers
  stage2_analysis/  # Plotting / analysis
  rl_output/        # Active RL outputs kept in place for backward compatibility
```

## Script Roles

### Training scripts

- `train_*.py`
- Responsible for:
  - creating environments
  - training PPO
  - saving models

### Evaluation scripts

- `eval_*.py`
- Responsible for:
  - loading trained models
  - running episodes
  - saving metrics or rollout artifacts
- MUST NOT contain training logic

### Comparison scripts

- comparison scripts may summarize or evaluate saved behavior
- no active comparison-summary script currently survives in the repo

### Rendering / plotting scripts

- `render_*.py` = video only
- `plot_*.py` = plotting only

## Authoritative Entry Points

- Stage 1 reference evaluation:
  `python rl/scripts/run_stage1_reference.py`
- Stage 2 training:
  `python rl/scripts/train_exo_stage2.py`
- Stage 2 evaluation:
  `python rl/scripts/eval_exo_stage2.py`

Supporting active commands:

- `python rl/stage1_compare/eval_stage1_deprl.py`
- `python rl/stage1_compare/render_stage1_reference.py`
- `python rl/stage2_analysis/plot_torque_vs_gait.py`

Important:

- `compare_stage1_results.py` is not present in the cleaned active repo
- archived commands live under `archived/` and are not part of the current path

## Active Data And Outputs

- Active derived reference data:
  `data/mocap_reference/`
- Active RL outputs remain under:
  - `rl/rl_output/stage1/`
  - `rl/rl_output/stage2/`
  - `rl/rl_output/analysis/`

Note:

- do not relocate the active RL output paths unless you also update the surviving scripts

## Rules For Agents

1. DO NOT mix training and evaluation in the same script.
2. DO NOT overstate results or imply stable exoskeleton assistance unless there is direct script/output evidence.
3. Treat `archived/` as reference only unless the user explicitly asks to work there.
4. Keep output paths for the surviving RL code under `rl/rl_output/` unless explicitly refactoring the active scripts.
5. Preserve reproducibility of the surviving DEP-RL walker path: `rl/baselines_DEPRL/` is the canonical active baseline location.
6. Prefer small direct scripts over wrappers or duplicated entrypoints.
7. If a file or command is missing from the active tree, document that honestly instead of recreating placeholder success.

## Definition of Done

Changes are complete when:

- the active path remains readable and internally consistent
- archived material stays out of the active path
- docs reflect what actually works, what is partial, and what failed
- no existing active functionality is broken by cleanup edits
