# Current Pipeline

This repository now preserves only the current working RL pipeline.

## Authoritative Workflow

### Stage 1 reference walker

- Canonical source: local DEP-RL baseline under `rl/baselines_DEPRL/myoLegWalk_20230514/myoLeg/`
- Official entrypoint:
  `python rl/scripts/run_stage1_reference.py`
- Primary outputs:
  - `rl/rl_output/stage1/reference/stage1_reference.json`
  - `rl/rl_output/stage1/reference/stage1_reference_lengths.csv`

### Stage 2 exoskeleton training

- Environment:
  `rl/envs/exo_with_walker_env.py`
- Canonical Stage 2 trainer:
  `python rl/scripts/train_exo_stage2.py`
- Canonical Stage 2 evaluator:
  `python rl/scripts/eval_exo_stage2.py`
- Primary outputs:
  - `rl/rl_output/stage2/exo_policy.zip`
  - `rl/rl_output/stage2/best/best_model.zip`
  - `rl/rl_output/stage2/eval/stage2_eval_rollout.npz`

### Stage 1 comparison and rendering

- Comparison / evaluation tools:
  `rl/stage1_compare/`
- Video renderer:
  `python rl/stage1_compare/render_stage1_reference.py`
- Comparison outputs:
  `rl/rl_output/stage1/compare/`
- The legacy simple PPO and Stage 1 v2 comparison scripts are kept as tools, but their old checkpoint artifacts are not preserved in this cleaned checkpoint.

### Stage 2 analysis

- Torque-vs-gait analysis:
  `python rl/stage2_analysis/plot_torque_vs_gait.py`
- Outputs:
  - `rl/rl_output/analysis/stage2_torque_vs_gait_v2.png`
  - `rl/rl_output/analysis/stage2_torque_vs_gait_v2_summary.json`

## Surviving Structure

```text
rl/
  baselines/
  baselines_DEPRL/
  envs/
  scripts/
  stage1_compare/
  stage2_analysis/
  rl_output/
    stage1/
    stage2/
    analysis/
```

Everything else from the earlier experimental SB3 walker phase, deprecated wrappers, notebooks, OpenSim work, EMG artifacts, and stale outputs has been removed.
