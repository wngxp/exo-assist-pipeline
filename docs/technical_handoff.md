# Technical Handoff

## What Was Changed During Cleanup

The repo was reorganized to separate the active handoff path from exploratory or stalled branches.

Created or added during cleanup:

- `docs/project_summary.md`
- `docs/repo_cleanup_plan.md`
- `docs/technical_handoff.md`
- `docs/failed_attempts.md`
- `docs/literature_context.md`
- `docs/next_steps.md`
- `docs/commands.md`
- `rl/reference/`
- `data/mocap_reference/`
- top-level `archived/`
- top-level `scripts/`
- top-level `outputs/`

Moved to `archived/`:

- `emg/`
- duplicate top-level `baselines_DEPRL/`
- `rl_isaac/`
- `rl/mocap_study/`
- AddBiomechanics conversion and inspection scripts
- stale `docs/current_pipeline.md`
- unused `rl/stage1_compare/common.py`
- generated `rl/MUJOCO_LOG.TXT`
- exploratory stage-1 extraction outputs that were not part of the active reference-eval path

Kept active:

- `opensim/`
- `rl/baselines/`
- `rl/baselines_DEPRL/`
- `rl/envs/exo_with_walker_env.py`
- `rl/scripts/run_stage1_reference.py`
- `rl/scripts/train_exo_stage2.py`
- `rl/scripts/eval_exo_stage2.py`
- `rl/stage1_compare/eval_stage1_deprl.py`
- `rl/stage1_compare/render_stage1_reference.py`
- `rl/stage2_analysis/plot_torque_vs_gait.py`
- `rl/rl_output/`
- `data/mocap_reference/`

## What Each Major Folder Does

- `docs/`
  - human-facing handoff documentation
- `opensim/`
  - preserved OpenSim/Moco result artifacts and biomechanics context
- `rl/`
  - surviving DEP-RL/MyoSuite code, reference helpers, and active RL outputs
- `data/`
  - derived mocap reference files needed by the current stage-2 environment
- `archived/`
  - old experiments, duplicate data, side scaffolds, and deprecated scripts
- `scripts/`
  - top-level placeholder so the repo reads clearly; no duplicate wrappers were added
- `outputs/`
  - top-level placeholder; current RL scripts still write to `rl/rl_output/`

## What Was Tested

Locally verified during cleanup:

- `python3 -m unittest discover -s archived/rl_isaac/tests`
- `python3 -m compileall rl archived/rl_isaac`

Audited but not re-run end-to-end on this Mac:

- `python rl/scripts/run_stage1_reference.py`
- `python rl/scripts/train_exo_stage2.py`
- `python rl/scripts/eval_exo_stage2.py`
- `python rl/stage1_compare/eval_stage1_deprl.py`
- `python rl/stage1_compare/render_stage1_reference.py`
- `python rl/stage2_analysis/plot_torque_vs_gait.py`

Reason:

- the project owner noted that the main MyoSuite/DEP-RL/OpenSim workflow is not runnable on this Mac
- the original runnable host was reported as `wxp@10.24.31.48`

## Errors And Issues Observed In The Audit

- The old README and `docs/current_pipeline.md` overstated how cleaned the repo already was.
- The repo contained a duplicate top-level `baselines_DEPRL/` copy.
- The active stage-2 environment depended on helper code and a reference CSV hidden inside `rl/mocap_study/`.
- `compare_stage1_results.py` was referenced in older notes but is not actually present.
- The repo does not include a fully pinned environment specification for the main workflow.
- The active RL scripts still use fixed output filenames in several places, so reruns can overwrite results.
- OpenSim is represented mainly by artifact files, not a maintained execution path.

## Environment And Dependencies

Cleanup environment:

- Python `3.13.2`
- filesystem audit and documentation done locally on macOS

Inferred project dependencies from the surviving code:

- `myosuite`
- `deprl`
- `stable-baselines3`
- `nimblephysics`
- `numpy`
- `pandas`
- `matplotlib`
- `imageio`
- MuJoCo / rendering stack for video scripts
- OpenSim/Moco tooling for the preserved biomechanics side of the project

Important limitation:

- exact Linux environment, package versions, and OpenSim/Moco setup were not captured in this repo

## Exact Commands Worth Remembering

Active path:

```bash
python rl/scripts/run_stage1_reference.py
python rl/scripts/train_exo_stage2.py
python rl/scripts/eval_exo_stage2.py
python rl/stage1_compare/eval_stage1_deprl.py
python rl/stage1_compare/render_stage1_reference.py
python rl/stage2_analysis/plot_torque_vs_gait.py
```

Local verification commands used in cleanup:

```bash
python3 -m unittest discover -s archived/rl_isaac/tests
python3 -m compileall rl archived/rl_isaac
```

## Bottom Line

The active repo is now much smaller than the original tree implied. The surviving RL code is readable and preserved, but it is still a partial experiment stack. The most important value of this cleanup is not that it made the repo look better; it made the repo honest.
