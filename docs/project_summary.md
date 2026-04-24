# Project Summary

## Goal

The original project goal was to explore hip exoskeleton assistance and control, with an emphasis on musculoskeletal realism rather than a toy rigid-body walker.

## How The Project Evolved

### Early direction

The early direction leaned on OpenSim/Moco to estimate torques, inspect biomechanics, and build intuition about what a hip-assistance problem should look like in a more realistic human model.

### Later direction

The later direction shifted toward reinforcement learning with MyoSuite and a frozen DEP-RL walker:

- Stage 1: use a fixed local DEP-RL reference walker.
- Stage 2: train a small PPO policy that adds bilateral hip torques on top of that walker.

### Parallel explorations

Along the way, the repo also accumulated:

- AddBiomechanics data inspection and conversion attempts
- mocap-tracking reward experiments
- a separate Isaac Lab / RSL-RL scaffold
- EMG side work

Those explorations were useful for learning, but they did not converge into one clean, reproducible pipeline.

## Current Limitation

The core limitation is simple: stable, human-like walking was not solved here in a way that makes the exoskeleton-assistance result convincing.

More specifically:

- PPO-from-scratch walking remained unstable.
- The DEP-RL/MyoSuite baseline was helpful as a starting point, but not reliable enough to claim robust assistive control.
- The stage-2 exoskeleton policy path exists, but it should be treated as a partial experiment with artifact-backed outputs, not a validated scientific result.

## What This Repo Is Good For

- seeing how the project direction changed over time
- reusing a few surviving scripts and artifacts
- understanding what was tried and where the repo got messy
- handing off context to someone starting a cleaner continuation

## Final Recommendation

Do not continue stacking ad hoc PPO scripts into this repository.

If the project continues, the best use of this repo is as a handoff and reference pack:

- use the literature here as the standard for what a serious exoskeleton-simulation pipeline looks like
- keep this repo frozen and readable
- start a cleaner follow-on repo if more development is planned

The strongest lesson from this repo is that it is better to build on a cleaner framework or reproduce a stronger published setup than to keep extending messy PPO-from-scratch walking experiments.
