# exo-assist-pipeline

This repository is a research handoff for exploratory hip-exoskeleton assistance work. It preserves the surviving OpenSim/Moco artifacts, the current MyoSuite/DEP-RL stage-1/stage-2 RL path, a small derived mocap-reference snapshot, and the failed or exploratory side branches under `archived/`.

## Current Status

- This repo is not a polished success story.
- Stable human-like walking and validated exoskeleton assistance were not solved here.
- The active RL path is small and readable, but it should be treated as a partial experiment, not a finished control stack.
- Old experiments, duplicate data, and side scaffolds were moved under `archived/` instead of being hard-deleted.

## Repo Structure

```text
exo-assist-pipeline/
  README.md
  docs/
  opensim/
  rl/
  data/
  outputs/
  scripts/
  archived/
```

- `docs/`: handoff documentation and cleanup record.
- `opensim/`: preserved OpenSim/Moco result artifacts and context.
- `rl/`: surviving DEP-RL/MyoSuite stage-1 and stage-2 code plus active RL outputs.
- `data/`: derived mocap-reference snapshot used by the active stage-2 environment.
- `outputs/`: top-level placeholder for future shared outputs; current RL scripts still write to `rl/rl_output/`.
- `scripts/`: top-level placeholder so the repo reads clearly; active runnable scripts remain under `rl/`.
- `archived/`: exploratory, stalled, duplicate, or deprecated material kept for reference only.

## Setup Notes

- The project owner noted that the full MyoSuite/DEP-RL/OpenSim workflow is not runnable on this Mac.
- The original runnable host was reported as `wxp@10.24.31.48` with the repo under `~/repos/projects/exo-assist-pipeline`.
- This repo does not contain a pinned, reproducible environment file for the full workflow.
- Inferred dependencies from the surviving code include `myosuite`, `deprl`, `stable-baselines3`, `nimblephysics`, `numpy`, `pandas`, `matplotlib`, and `imageio`.
- OpenSim/Moco tooling is represented here mostly by output artifacts, not a maintained runnable script stack.

## Commands Still In The Active Path

These are the commands worth keeping in mind for the surviving RL/OpenSim handoff:

```bash
python rl/scripts/run_stage1_reference.py
python rl/scripts/train_exo_stage2.py
python rl/scripts/eval_exo_stage2.py
python rl/stage1_compare/eval_stage1_deprl.py
python rl/stage1_compare/render_stage1_reference.py
python rl/stage2_analysis/plot_torque_vs_gait.py
```

Notes:

- They are part of the active handoff path.
- They were not re-run end-to-end during this Mac cleanup pass.
- Only the archived `rl_isaac` scaffold tests were locally verified.

## Deprecated Or Archived Commands

- Anything under `archived/` is preserved for context, not recommended as the current path.
- The old AddBiomechanics conversion scripts are now under `archived/addbiomechanics/`.
- The old mocap PPO experiments are now under `archived/mocap_study/`.
- The `rl_isaac` scaffold is now under `archived/rl_isaac/`; it is cleaner than the old RL tree, but still only a scaffold.
- `compare_stage1_results.py` is referenced in older notes but is not present in the repo and should not be treated as available.

## Read These First

1. `docs/project_summary.md`
2. `docs/technical_handoff.md`
3. `docs/failed_attempts.md`
4. `docs/literature_context.md`
5. `docs/next_steps.md`
6. `docs/commands.md`
