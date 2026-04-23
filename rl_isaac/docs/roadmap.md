# Roadmap

## Current State

The repository now contains a clean scaffold for humanoid walking experiments with reproducible configs, isolated package structure, unique outputs, and extension interfaces for future locomotion features.

## Near-Term Milestones

### 1. Isaac Lab environment integration

- Select and pin an Isaac Lab version
- Register a humanoid asset and scene setup
- Replace scaffold stepping with simulator-backed stepping
- Add headless and viewer-backed runtime modes

### 2. RSL-RL training integration

- Add trainer configuration mapping
- Wire experiment configs into PPO runner construction
- Save checkpoints and training metrics in stable output formats
- Add reproducibility metadata for seeds and software versions

### 3. Reference and mocap support

- Define a dataset layout under `data/`
- Implement reference loader parsing
- Add optional reference-tracking observations and reward terms
- Add dataset/version metadata to evaluation outputs

### 4. Gait-phase features

- Decide whether phase comes from contacts, kinematics, or reference signals
- Add configurable phase encoding options
- Track feature provenance in configs and manifests

### 5. Residual exoskeleton assistance

- Add assistive actuator abstraction
- Define residual action composition rules
- Separate humanoid base action space from assistive action space
- Add targeted evaluation scripts for assistive policies

### 6. Project hardening

- Pin dependencies and simulator versions
- Add CI for config validation and smoke tests
- Add linting/formatting/type-check steps
- Add experiment comparison and result summarization utilities

## Explicit Non-Goals For This Scaffold

- Recreating old experiment wrappers
- Copying logic from the legacy `rl/` tree
- Implementing placeholder complexity with no immediate integration value
