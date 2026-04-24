# rl_isaac

`rl_isaac` is an isolated subproject for building a maintainable Isaac Lab / RSL-RL humanoid locomotion pipeline inside this repository.

It is intentionally separate from the legacy `rl/`, DEP-RL, MyoSuite, and OpenSim codepaths. The goal is to create a clean foundation for future humanoid walking, mocap tracking, and exoskeleton-assistance research without inheriting old experiment structure or hidden dependencies.

## What is implemented now

- A self-contained Python package under `src/rl_isaac/`
- Config-driven experiment loading from `configs/`
- Minimal production-style entrypoints:
  - `scripts/train.py`
  - `scripts/eval.py`
  - `scripts/render.py`
- A baseline humanoid walking scaffold with separate modules for:
  - environment state/update logic
  - observations
  - rewards
  - termination
  - task configuration
- Placeholder interfaces for future:
  - mocap reference loading
  - gait phase features
  - residual exoskeleton assistance
- Deterministic smoke-test runners that exercise the pipeline structure without pretending the Isaac Lab / RSL-RL integration is already complete

## What is intentionally not implemented yet

- Isaac Lab scene assembly
- Robot asset registration
- RSL-RL PPO training
- mocap parsing or reference tracking controllers
- exoskeleton control residuals
- video rendering through Isaac Lab cameras/viewers

## Folder Structure

```text
rl_isaac/
  configs/
    experiments/
  data/
  docs/
  outputs/
  scripts/
  src/
    rl_isaac/
  tests/
```

## Intended Workflow

1. Define or copy an experiment config in `configs/experiments/`.
2. Run one of the entrypoints from the `rl_isaac/` directory:

   ```bash
   python3 scripts/train.py --config configs/experiments/humanoid_walk.toml
   python3 scripts/eval.py --config configs/experiments/humanoid_walk.toml
   python3 scripts/render.py --config configs/experiments/humanoid_walk.toml
   ```

3. Inspect unique run directories under `outputs/train/`, `outputs/eval/`, and `outputs/render/`.
4. Extend task modules or add new task packages under `src/rl_isaac/tasks/`.
5. Replace the scaffold backend with Isaac Lab / RSL-RL adapters when the simulation stack is ready to be pinned and tested.

## Design Notes

- Configuration is treated as the experiment contract.
- Task logic is decomposed so observations, rewards, actions, and termination conditions can evolve independently.
- Outputs are always written to unique timestamped folders for reproducibility.
- The current scaffold is small on purpose. It validates structure, interfaces, and experiment flow without adding fake simulator complexity.

See [docs/architecture.md](docs/architecture.md) for the module boundaries and [docs/roadmap.md](docs/roadmap.md) for planned milestones.
