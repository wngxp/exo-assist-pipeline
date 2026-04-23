# Architecture

## Purpose

`rl_isaac` is a standalone research subproject for humanoid locomotion experiments that will eventually target Isaac Lab simulation and RSL-RL training. The current implementation focuses on clean boundaries, typed interfaces, and reproducible experiment layout rather than deep simulator functionality.

## Design Principles

- Keep the subproject isolated from legacy `rl/`, DEP-RL, MyoSuite, and OpenSim code.
- Make experiment configuration the primary control surface.
- Keep task logic decomposed into small modules that are easy to replace independently.
- Prefer explicit run artifacts over implicit state.
- Add placeholders only where they define a real interface or future integration boundary.

## Layering

```text
scripts/
  thin CLI wrappers

src/rl_isaac/runners/
  train/eval/render orchestration

src/rl_isaac/config/
  config loading and validation

src/rl_isaac/tasks/
  task registry and task-specific modules

src/rl_isaac/features/
src/rl_isaac/references/
src/rl_isaac/assistance/
  future integration interfaces
```

## Task Structure

The baseline humanoid task is intentionally split into separate files:

- `env.py`: environment state and step/reset contract
- `actions.py`: action shaping, clipping, and future assistance hook
- `observations.py`: observation assembly from named features
- `rewards.py`: reward term definitions and weighting
- `termination.py`: episode stop conditions
- `config.py`: typed task configuration

This separation keeps common locomotion experiments editable without forcing all changes into one large environment file.

## Runtime Strategy

The current runtime backend is a deterministic scaffold. It exists to verify:

- config loading
- run directory creation
- task construction
- rollout tracing
- artifact writing

It does not claim to be a physics simulator and should not be used for scientific conclusions. When Isaac Lab is integrated, the task-facing contracts should remain stable while the backend implementation changes underneath them.

## Extension Points

The scaffold already reserves interfaces for the next research steps:

- `references/mocap.py`: reference trajectory loading contract
- `features/gait_phase.py`: gait phase feature provider contract
- `assistance/residual.py`: residual exoskeleton action modifier contract

The intent is to plug these into task configuration and runtime assembly without rewriting the whole project layout.
