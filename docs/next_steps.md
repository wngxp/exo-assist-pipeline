# Next Steps

## Recommended Plan

1. Freeze this repository as a handoff and reference repo.
2. Do not keep stacking new scripts into it.
3. If development continues, start a cleaner second repository.
4. Pick one direction and commit to it instead of mixing all of them again.
5. Avoid PPO-from-scratch walking unless the environment is very small and tightly controlled.

## Choose One Direction

### Option 1

AddBiomechanics to mocap reference trajectories to Isaac Lab or RSL-RL humanoid tracking.

Why:

- closest to a cleaner future RL direction
- more aligned with the literature than extending the old MyoSuite script pile

### Option 2

OpenSim/Moco biomechanics report or demo repo.

Why:

- useful if the near-term goal is interpretation, torque estimation, or a readable biomechanics deliverable instead of RL training

### Option 3

A more faithful reproduction of an existing exoskeleton-simulation paper.

Why:

- better than inventing another custom half-stack
- forces the next phase to match a real published simulator and task setup

## Code-Organization Rules For Any Follow-On Work

1. Keep this repo frozen.
2. Treat `archived/` as read-only reference, not a source of new active scripts.
3. Keep only runnable or clearly important scripts in the active path.
4. Keep training, evaluation, rendering, and plotting scripts separate.
5. Require short script headers that state purpose, inputs, outputs, command, and status.
6. Refuse to add unexplained notebooks, side scripts, and duplicate baseline copies to the active tree.

## Practical Advice

- If the next step is RL, move to a cleaner simulator/runtime stack.
- If the next step is biomechanics reporting, keep the scope around OpenSim/Moco and do not pretend it is an RL repo.
- If the next step is exoskeleton control research, use the stronger literature as the design target and assume better simulator fidelity will be necessary.

## Bottom Line

The best next action for the project owner is to stop treating this repo as the place where the final solution will emerge. It is now a cleaned handoff. The continuation, if any, should start fresh and be narrower.
