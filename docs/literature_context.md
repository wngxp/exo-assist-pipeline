# Literature Context

## Why The Literature Matters

This repo only makes sense when compared against stronger published work.

The key lesson is not that the project had the wrong high-level goal. The key lesson is that serious exoskeleton-assistance results require a much more disciplined simulation and training setup than a small MyoSuite PPO experiment stack.

## Luo et al. 2024, Nature

The strongest comparison point is the Nature paper by Luo et al. on learning-in-simulation exoskeleton assistance.

Why it matters:

- it used a custom 50 degree-of-freedom musculoskeletal model
- it modeled 208 muscles
- it combined human motion imitation, muscle coordination, and exoskeleton control
- it treated exoskeleton assistance as part of a serious learning-in-simulation framework, not just a torque bonus on top of a simple walker

Implication for this repo:

- the gap between that paper and this repo is large
- this repo should not be framed as a close reproduction of that level of result

## Exo-Plore

Exo-Plore points in a similar direction:

- a neuromechanical human model
- deep reinforcement learning
- an explicit human-exoskeleton interaction objective
- surrogate optimization over exoskeleton parameters such as hip gain and delay

Why that matters:

- it suggests that exoskeleton assistance benefits from a structured optimization setup, not casual reward tuning
- it reinforces that serious assistive-control work usually needs better simulation fidelity and better experiment design than simple PPO script iteration

## What Both Papers Suggest

Taken together, these papers point toward a more serious recipe:

- a proper musculoskeletal or neuromechanical simulator
- mocap or imitation/reference tracking
- domain randomization and robustness work
- carefully designed rewards and assistance objectives
- stronger experiment discipline than one-off script growth

## Where This Repo Fits

This repo is best understood as an exploratory attempt in that broader direction.

It is not:

- a completed reproduction of the Nature paper
- a faithful Exo-Plore reproduction
- a validated exoskeleton-assistance benchmark

It is:

- a record of exploratory engineering work
- a partial RL pipeline on top of a frozen walker
- a reminder that realistic exoskeleton control likely needs a stronger simulator and cleaner pipeline than what survived here
