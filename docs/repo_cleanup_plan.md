# Repository Cleanup Plan

## Status

Prepared from a full repository audit and executed during the cleanup pass on 2026-04-24.

This file is kept as the record of what was audited, why items were kept or archived, and what risks were considered before moving files.

## Goal

Turn this repository into a readable handoff repo without overstating project maturity.

The cleanup should make five things obvious:

1. What still works or appears runnable from code/artifacts.
2. What partially works but is not a trustworthy research result.
3. What failed or stalled.
4. What was exploratory only.
5. What the next developer should do instead of extending the current script pile.

## Audit Summary

### Active code path to keep visible

- `opensim/`
  - Historical OpenSim/Moco result artifacts.
  - Important for biomechanics context, but not a maintained runnable pipeline here.
- `rl/baselines/`
  - Helper for loading the local DEP-RL walker baseline.
- `rl/baselines_DEPRL/`
  - Canonical local DEP-RL walker files used by the current RL pipeline.
- `rl/envs/exo_with_walker_env.py`
  - Current stage-2 exoskeleton-on-top-of-frozen-walker environment.
- `rl/scripts/run_stage1_reference.py`
  - Current stage-1 reference evaluation entrypoint.
- `rl/scripts/train_exo_stage2.py`
  - Current stage-2 training entrypoint.
- `rl/scripts/eval_exo_stage2.py`
  - Current stage-2 evaluation entrypoint.
- `rl/stage1_compare/eval_stage1_deprl.py`
  - Useful as a direct DEP-RL evaluation tool.
- `rl/stage1_compare/render_stage1_reference.py`
  - Useful for rendering the baseline walker.
- `rl/stage2_analysis/plot_torque_vs_gait.py`
  - Useful for post-hoc analysis of saved stage-2 rollout artifacts.
- `rl/rl_output/`
  - Keep because the active RL scripts already read/write here.
  - Do not relocate during this cleanup because that would break the current entrypoints.

### Active dependency that was hidden inside an experimental tree

- `rl/envs/exo_with_walker_env.py` originally imported mocap-reference helpers and a CSV from `rl/mocap_study/`.
- Those helpers were moved into `rl/reference/`, and the reference snapshot was copied into `data/mocap_reference/`, so the old `rl/mocap_study/` tree could be archived.

### Clear archive candidates

- `emg/`
  - Raw binary recordings plus one notebook, no clear active integration.
- Top-level `baselines_DEPRL/`
  - Duplicate of `rl/baselines_DEPRL/`.
- `rl_isaac/`
  - Cleaner than much of the repo, but still a future scaffold and not part of the MyoSuite/DEP-RL pipeline.
- `rl/b3d_to_myoleg_v2/`
  - Exploratory AddBiomechanics conversion tooling.
- `rl/b3d_to_myoleg.tar.gz`
  - Archived bundle tied to the same exploratory conversion work.
- `rl/data_inspect.py`
- `rl/inspect_b3d_trial.py`
- `rl/scripts/extract_reference_gait.py`
- `rl/scripts/inspect_data.py`
- `rl/scripts/plot_hip_curves.py`
  - One-off AddBiomechanics inspection or plotting scripts.
- Most of `rl/mocap_study/`
  - Experimental walker-from-scratch and mocap-tracking work.
- `docs/current_pipeline.md`
  - Misleading and outdated. It claims earlier experimental areas were already removed, which is false.
- `rl/MUJOCO_LOG.TXT`
  - Generated log file, not a handoff-facing source artifact.

### Dead or unclear items

- `rl/stage1_compare/common.py`
  - Utility helpers appear unused by the surviving source files.
- `rl/stage1_compare/compare_stage1_results.py`
  - Referenced in docs/instructions but not actually present in the repo.
- Some existing README/docs text over-claims cleanup completeness.

## Reorganization Outcome

### New or clarified active top-level structure

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

Notes:

- `rl/rl_output/` stays in place for backward compatibility with the current RL scripts.
- `data/` now holds the derived mocap reference snapshot that the active stage-2 environment depends on.
- `scripts/` and `outputs/` were added as explicit top-level handoff folders, even though the runnable code still lives under `rl/`.

## File Moves Performed

### Moved to active data location

- Copied the derived mocap reference files needed for stage-2 tracking into `data/mocap_reference/`.
- Moved active mocap-reference helper code out of `rl/mocap_study/` into the new `rl/reference/` package.

### Moved to `archived/`

- `emg/`
- Top-level duplicate `baselines_DEPRL/`
- `rl_isaac/`
- `rl/b3d_to_myoleg_v2/`
- `rl/b3d_to_myoleg.tar.gz`
- `rl/data_inspect.py`
- `rl/inspect_b3d_trial.py`
- `rl/scripts/extract_reference_gait.py`
- `rl/scripts/inspect_data.py`
- `rl/scripts/plot_hip_curves.py`
- Most or all of `rl/mocap_study/` once active reference helpers/data are extracted
- `docs/current_pipeline.md`
- `rl/MUJOCO_LOG.TXT`
- Derived stage-1 extraction artifacts that belong to archived AddBiomechanics exploration rather than the active stage-1 reference evaluation

## Documentation Work Completed

Created or rewritten:

- `README.md`
- `docs/project_summary.md`
- `docs/repo_cleanup_plan.md`
- `docs/technical_handoff.md`
- `docs/literature_context.md`
- `docs/failed_attempts.md`
- `docs/next_steps.md`
- `docs/commands.md`

All docs should clearly distinguish:

- working
- partial
- failed
- exploratory
- recommended next steps

## Script Header Cleanup

Added short module headers to the important surviving scripts covering:

- purpose
- input files
- output files
- command to run
- status: working / partial / deprecated

## Verification Plan

### What can be checked locally on this Mac

- import-free syntax/bytecode compilation
- pure-Python scaffold tests
- file/path consistency

### What cannot be honestly claimed as locally verified

- MyoSuite + DEP-RL walking/evaluation/training
- OpenSim/Moco execution
- AddBiomechanics extraction end-to-end

### Known local verification results already observed

- `python3 -m unittest discover -s rl_isaac/tests` passes.
- `python3 -m compileall rl rl_isaac` succeeds.

## Risks

- Moving files that the current stage-2 environment imports indirectly through `rl/mocap_study/`.
- Breaking path assumptions baked into old exploratory scripts.
- Accidentally implying scientific validity from artifact presence alone.

## Cleanup Rules

- Prefer archiving over deleting unless a file is an obvious duplicate.
- Do not invent missing results.
- Do not relabel unstable walker experiments as solved locomotion.
- Keep the active path small and readable even if that means being blunt about what did not work.
