# Failed Attempts

## PPO-From-Scratch Walking Was Unstable

The repo contains clear evidence that walking-from-scratch PPO experiments were explored, but they did not become a stable foundation for serious exoskeleton work.

Problems with that direction:

- unstable locomotion learning
- too many ad hoc reward and environment tweaks
- weak separation between exploratory scripts and anything close to a maintained pipeline

## DEP-RL/MyoSuite Baseline Was Helpful But Not Sufficient

The local DEP-RL walker baseline was the most useful surviving starting point:

- it could walk and render
- it gave the project a fixed reference controller
- it allowed the stage-2 exoskeleton policy experiment to exist at all

But it was still not enough to claim robust exoskeleton assistance:

- walking reliability was not strong enough for confident assistive-policy conclusions
- the exoskeleton policy layer remained a partial experiment
- artifact presence should not be confused with validated assistive performance

## OpenSim/Moco Was Informative But Too Slow For The RL Direction

OpenSim/Moco helped with:

- torque estimation
- biomechanics intuition
- understanding the gap between simple reward hacking and more realistic assistance modeling

But it was not a practical RL training loop for this repo:

- too slow for large-scale RL iteration
- not preserved here as a clean reproducible training stack
- useful mainly as analysis context and a reminder of what realism requires

## Exo-Plore Build / Reproduction Direction Did Not Land Cleanly

The repo owner noted interest in Exo-Plore-style work, but that path did not become a clean runnable branch here.

Practical issues included:

- dependency and path management problems
- runtime setup friction
- the usual difficulty of jumping from a published simulator stack to an ad hoc local reproduction

## AddBiomechanics Extraction Was Started But Not Finished

The repo includes genuine effort toward AddBiomechanics-based data extraction and conversion, but it did not become a full RL-ready gait dataset pipeline.

What happened:

- inspection scripts were written
- conversion tooling was explored
- reference-cycle artifacts were created

What did not happen:

- a clean, reproducible raw-data-to-training-data pipeline
- a full imitation-learning or mocap-tracking system built on that data
- a trustworthy bridge from AddBiomechanics to robust exoskeleton RL experiments

## Takeaway

The main failure mode was not lack of effort. It was trying to span too many difficult layers at once:

- biomechanics realism
- walking control
- mocap integration
- exoskeleton assistance
- RL stability
- simulator/toolchain maintenance

That combination is exactly why the literature uses more disciplined simulators, cleaner pipelines, and much more deliberate reward/task design than what this repo ultimately achieved.
