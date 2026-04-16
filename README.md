# Exo Assist Pipeline

Current authoritative pipeline:

- Stage 1 reference walker: local DEP-RL baseline evaluated with [`rl/scripts/run_stage1_reference.py`](rl/scripts/run_stage1_reference.py)
- Stage 2 exoskeleton training: [`rl/scripts/train_exo_stage2.py`](rl/scripts/train_exo_stage2.py)
- Stage 2 evaluation: [`rl/scripts/eval_exo_stage2.py`](rl/scripts/eval_exo_stage2.py)
- Stage 1 comparison tools: [`rl/stage1_compare/`](rl/stage1_compare)
- Stage 2 analysis: [`rl/stage2_analysis/plot_torque_vs_gait.py`](rl/stage2_analysis/plot_torque_vs_gait.py)

Outputs are organized under `rl/rl_output/stage1`, `rl/rl_output/stage2`, and `rl/rl_output/analysis`.

Legacy PPO comparison scripts are still present in `rl/stage1_compare/`, but the old PPO checkpoints and stale experiment outputs are intentionally not retained in this cleaned checkpoint.

See [`docs/current_pipeline.md`](docs/current_pipeline.md) for the cleaned repo layout, authoritative scripts, and expected outputs.
