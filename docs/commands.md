# Commands

This file lists the useful commands still represented in the repo. Commands are grouped by topic and labeled as active or archived where needed.

## OpenSim / Moco

Active maintained commands:

- None survived as a clean runnable script path in this repo.

What exists instead:

- preserved result artifacts under `opensim/`
- biomechanics context, not a maintained execution workflow

## RL / DEP-RL / MyoSuite

Active path:

```bash
python rl/scripts/run_stage1_reference.py
python rl/scripts/train_exo_stage2.py
python rl/scripts/eval_exo_stage2.py
python rl/stage1_compare/eval_stage1_deprl.py
```

Notes:

- These are the main surviving commands for the DEP-RL/MyoSuite branch.
- They belong to the active handoff path.
- They were not re-run end-to-end during this Mac cleanup pass.

Archived or deprecated:

- `python archived/mocap_study/train/train_mocap_stage1.py`
- `python archived/rl_isaac/scripts/train.py --config archived/rl_isaac/configs/experiments/humanoid_walk.toml`
- `python archived/rl_isaac/scripts/eval.py --config archived/rl_isaac/configs/experiments/humanoid_walk.toml`
- `python archived/rl_isaac/scripts/render.py --config archived/rl_isaac/configs/experiments/humanoid_walk.toml`

## Rendering / Video

Active path:

```bash
python rl/stage1_compare/render_stage1_reference.py
```

Archived:

- `python archived/addbiomechanics/b3d_to_myoleg_v2/gen_video.py`
- `python archived/addbiomechanics/b3d_to_myoleg_v2/run.py`

## AddBiomechanics / Data Extraction

Archived only:

```bash
python archived/addbiomechanics/scripts/extract_reference_gait.py
python archived/addbiomechanics/scripts/inspect_data.py
python archived/addbiomechanics/scripts/data_inspect.py
python archived/addbiomechanics/scripts/inspect_b3d_trial.py
python archived/addbiomechanics/b3d_to_myoleg_v2/convert_b3d_to_myoleg.py <input> <output>
```

Notes:

- These are preserved because they show what was tried.
- They are not part of the active path anymore.
- The AddBiomechanics-to-RL dataset pipeline is incomplete.

## Plotting / Analysis

Active path:

```bash
python rl/stage2_analysis/plot_torque_vs_gait.py
```

Archived:

```bash
python archived/addbiomechanics/scripts/plot_hip_curves.py
```

## Verification Commands Used During Cleanup

Locally verified:

```bash
python3 -m unittest discover -s archived/rl_isaac/tests
python3 -m compileall rl archived/rl_isaac
```

## Missing Or Broken References

- Older notes mention `python rl/stage1_compare/compare_stage1_results.py`, but that file is not present.
- Treat that command as unavailable, not merely undocumented.
