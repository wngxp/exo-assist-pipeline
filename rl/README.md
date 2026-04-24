# rl

This is the surviving active code path for the cleaned handoff repo.

Main pieces:

- `baselines_DEPRL/`
  - canonical local DEP-RL walker baseline
- `reference/`
  - derived mocap-reference helpers used by Stage 2
- `envs/`
  - stage-2 exoskeleton environment
- `scripts/`
  - authoritative stage-1 and stage-2 entrypoints
- `stage1_compare/`
  - direct DEP-RL evaluation and rendering helpers
- `stage2_analysis/`
  - post-hoc plotting and analysis
- `rl_output/`
  - active RL outputs kept in place for backward compatibility

Status:

- preserved because it is the clearest surviving experiment path
- still partial, not a validated exoskeleton-control solution
