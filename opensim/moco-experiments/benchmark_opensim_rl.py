#!/usr/bin/env python3
"""
Benchmark: How fast can OpenSim Manager.integrate() step forward?
This determines whether OpenSim is viable as an RL environment.

Run on cortex:
    conda activate opensim
    cd ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
    python3 benchmark_opensim_rl.py
"""
import opensim as osim
import time
import numpy as np

MODEL_PATH = 'multisubject/P010/P010_scaled_unlocked.osim'

print("=" * 60)
print("Benchmark: OpenSim Manager.integrate() speed")
print("=" * 60)

model = osim.Model(MODEL_PATH)

# Add CoordinateActuators for exo torque (mimics what RL would do)
for side in ['r', 'l']:
    act = osim.CoordinateActuator(f'hip_flexion_{side}')
    act.setName(f'exo_hip_{side}')
    act.setOptimalForce(1.0)
    act.setMinControl(-30.0)
    act.setMaxControl(30.0)
    model.addForce(act)

model.finalizeConnections()
state = model.initSystem()

# Count DOFs and muscles
n_coords = model.getNumCoordinates()
n_muscles = model.getMuscles().getSize()
n_bodies = model.getNumBodies()
print(f"  DOFs:    {n_coords}")
print(f"  Muscles: {n_muscles}")
print(f"  Bodies:  {n_bodies}")

# Set a reasonable initial pose (slight hip flexion)
model.getCoordinateSet().get('hip_flexion_r').setValue(state, 0.2)
model.getCoordinateSet().get('hip_flexion_l').setValue(state, -0.1)
model.getCoordinateSet().get('knee_angle_r').setValue(state, -0.3)
model.getCoordinateSet().get('knee_angle_l').setValue(state, -0.1)

model.realizeVelocity(state)

# Benchmark: integrate for 1 second at 100 Hz (100 steps)
dt = 0.01  # 100 Hz, same as Nature paper
n_steps = 100
manager = osim.Manager(model)
manager.setIntegratorAccuracy(1e-3)

print(f"\nIntegrating {n_steps} steps at dt={dt}s...")
manager.initialize(state)

t0 = time.time()
fell = False
for i in range(n_steps):
    t_target = (i + 1) * dt
    try:
        state = manager.integrate(t_target)
    except Exception as e:
        print(f"  Crashed at step {i}: {e}")
        fell = True
        break

    # Read state (what RL policy would observe)
    hip_r = model.getCoordinateSet().get('hip_flexion_r').getValue(state)
    hip_l = model.getCoordinateSet().get('hip_flexion_l').getValue(state)

wall_time = time.time() - t0
sim_time = state.getTime()

if not fell:
    steps_per_sec = n_steps / wall_time
    realtime_ratio = sim_time / wall_time
else:
    steps_done = max(i, 1)
    steps_per_sec = steps_done / wall_time
    realtime_ratio = sim_time / wall_time
    print(f"\n  Model fell at t={sim_time:.3f}s (expected — no ground contact)")

print(f"\n  Results:")
print(f"    Wall time:        {wall_time:.3f}s")
print(f"    Sim time:         {sim_time:.3f}s")
print(f"    Steps/sec:        {steps_per_sec:.1f}")
print(f"    Real-time ratio:  {realtime_ratio:.2f}x")

# Estimate RL training time
episodes = 1_000_000
steps_per_ep = 200  # ~2 sec episode
total_steps = episodes * steps_per_ep
hours = total_steps / steps_per_sec / 3600
print(f"\n  RL training estimate (1M episodes x 200 steps):")
print(f"    Total steps:      {total_steps:,}")
print(f"    Estimated time:   {hours:.0f} hours ({hours/24:.1f} days)")
print(f"    With 10x speedup (reduced model): ~{hours/10:.0f} hours")

# Verdict
print(f"\n  VERDICT:", end=" ")
if steps_per_sec > 500:
    print("OpenSim RL is feasible as-is")
elif steps_per_sec > 50:
    print("Borderline — need reduced model or fewer episodes")
else:
    print("Too slow — MuJoCo recommended")

print("\nDone.")