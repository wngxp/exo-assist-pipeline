# AGENTS.md

## Project Overview
This repository implements a two-stage RL pipeline for a hip exoskeleton:

- Stage 1: Learn a walking policy (MyoSuite)
- Stage 2: Add exoskeleton torque on top of the learned walker

---

## Directory Structure

rl/
  stage1_compare/     # Evaluation + comparison ONLY (no training)
  rl_output/          # Saved models and VecNormalize stats
  envs/               # Custom environments
  baselines/          # Reference or baseline implementations
  scripts/            # Stage 2 scripts
  legacy/             # Deprecated / unused scripts (do not delete)

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

---

## Stage 1 (Walker)

- Simple baseline: tutorial PPO (no normalization)
- v2: uses VecNormalize

Important:
- v2 models REQUIRE VecNormalize stats to evaluate correctly
- Always load:
  rl_output/walker_policy_stage1_v2_vecnormalize.pkl

---

## Stage 2 (Exoskeleton)

- Must NOT modify Stage 1 training scripts
- Depends on a stable Stage 1 walker

---

## Rules for Agents

1. DO NOT mix training and evaluation in the same script
2. DO NOT overwrite result files — always use unique filenames
3. DO NOT delete scripts — move to rl/legacy/ instead
4. DO NOT modify Stage 2 code when working on Stage 1
5. Preserve reproducibility: paths and model names must stay consistent

---

## Naming Conventions

- simple → baseline / tutorial implementation
- v2 / v3 → iterative improvements
- best → best checkpoint from training
- final → last checkpoint

---

## Definition of Done

Changes are complete when:
- scripts run without errors
- outputs are saved under rl/stage1_compare/results/
- no existing functionality is broken