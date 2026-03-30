#!/usr/bin/env python3
"""
Benchmark: Reduced OpenSim model for RL feasibility.
Strips model to essential lower-limb DOFs and muscles only.

Run on cortex:
    conda activate opensim
    cd ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
    python3 benchmark_opensim_reduced.py
"""
import opensim as osim
import time
import numpy as np
import xml.etree.ElementTree as ET
import os
import copy

MODEL_PATH = 'multisubject/P010/P010_scaled_unlocked.osim'
OUT_DIR = 'multisubject/P010/mocotrack_output'
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Step 1: Build reduced model via XML surgery ----
print("=" * 60)
print("Building reduced lower-limb model...")
print("=" * 60)

tree = ET.parse(MODEL_PATH)
root = tree.getroot()

# Keep only hip muscles (the ones the exo directly affects)
KEEP_MUSCLES = [
    'glmax1_r', 'glmax2_r', 'glmax3_r',
    'glmed1_r', 'glmed2_r', 'glmed3_r',
    'psoas_r', 'iliacus_r',
    'recfem_r', 'bflh_r', 'bfsh_r',
    'semimem_r', 'semiten_r',
    'glmax1_l', 'glmax2_l', 'glmax3_l',
    'glmed1_l', 'glmed2_l', 'glmed3_l',
    'psoas_l', 'iliacus_l',
    'recfem_l', 'bflh_l', 'bfsh_l',
    'semimem_l', 'semiten_l',
]

# Remove muscles not in keep list
removed_muscles = 0
for forceset in root.iter('ForceSet'):
    objects = forceset.find('objects')
    if objects is not None:
        to_remove = []
        for force in objects:
            name = force.get('name', '')
            # Remove muscles not in our keep list
            if force.tag in ['Millard2012EquilibriumMuscle',
                             'Thelen2003Muscle',
                             'DeGrooteFregly2016Muscle']:
                if name not in KEEP_MUSCLES:
                    to_remove.append(force)
        for f in to_remove:
            objects.remove(f)
            removed_muscles += 1

print(f"  Removed {removed_muscles} muscles, kept {len(KEEP_MUSCLES)}")

# Lock upper body coordinates to reduce DOF
LOCK_COORDS = [
    'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
    'arm_flex_r', 'arm_add_r', 'arm_rot_r',
    'elbow_flex_r', 'pro_sup_r',
    'arm_flex_l', 'arm_add_l', 'arm_rot_l',
    'elbow_flex_l', 'pro_sup_l',
]

locked_count = 0
for coord in root.iter('Coordinate'):
    name = coord.get('name', '')
    if name in LOCK_COORDS:
        locked_el = coord.find('locked')
        if locked_el is None:
            locked_el = ET.SubElement(coord, 'locked')
        locked_el.text = 'true'
        locked_count += 1

# Actually, Moco can't handle locked coords. Use clamped range instead.
# Undo locks, clamp instead
for coord in root.iter('Coordinate'):
    name = coord.get('name', '')
    if name in LOCK_COORDS:
        locked_el = coord.find('locked')
        if locked_el is not None:
            locked_el.text = 'false'
        # Clamp to default value
        default_val = coord.find('default_value')
        dv = float(default_val.text) if default_val is not None else 0.0
        range_el = coord.find('range')
        if range_el is None:
            range_el = ET.SubElement(coord, 'range')
        range_el.text = f'{dv - 0.001} {dv + 0.001}'

print(f"  Clamped {locked_count} upper-body coordinates")

reduced_path = os.path.join(OUT_DIR, 'P010_reduced.osim')
tree.write(reduced_path, xml_declaration=True)
print(f"  Saved: {reduced_path}")

# ---- Step 2: Benchmark the reduced model ----
print("\n" + "=" * 60)
print("Benchmarking reduced model...")
print("=" * 60)

model = osim.Model(reduced_path)

# Add exo actuators
for side in ['r', 'l']:
    act = osim.CoordinateActuator(f'hip_flexion_{side}')
    act.setName(f'exo_hip_{side}')
    act.setOptimalForce(1.0)
    act.setMinControl(-30.0)
    act.setMaxControl(30.0)
    model.addForce(act)

model.finalizeConnections()
state = model.initSystem()

n_coords = model.getNumCoordinates()
n_muscles = model.getMuscles().getSize()
print(f"  DOFs:    {n_coords}")
print(f"  Muscles: {n_muscles}")

# Set initial pose
model.getCoordinateSet().get('hip_flexion_r').setValue(state, 0.2)
model.getCoordinateSet().get('hip_flexion_l').setValue(state, -0.1)
model.getCoordinateSet().get('knee_angle_r').setValue(state, -0.3)
model.getCoordinateSet().get('knee_angle_l').setValue(state, -0.1)
model.realizeVelocity(state)

# Run benchmark
dt = 0.01
n_steps = 100
manager = osim.Manager(model)
manager.setIntegratorAccuracy(1e-3)
manager.initialize(state)

print(f"  Integrating {n_steps} steps at dt={dt}s...")

t0 = time.time()
for i in range(n_steps):
    try:
        state = manager.integrate((i + 1) * dt)
    except Exception as e:
        print(f"  Crashed at step {i}: {e}")
        break

wall_time = time.time() - t0
sim_time = state.getTime()
steps_done = min(i + 1, n_steps)
steps_per_sec = steps_done / wall_time
realtime_ratio = sim_time / wall_time

print(f"\n  FULL MODEL (previous):  1.5 steps/sec")
print(f"  REDUCED MODEL:          {steps_per_sec:.1f} steps/sec")
print(f"  Speedup:                {steps_per_sec / 1.5:.1f}x")
print(f"  Real-time ratio:        {realtime_ratio:.2f}x")

# RL estimates
for label, n_episodes in [("Tiny PoC (1K ep)", 1_000),
                           ("Small (10K ep)", 10_000),
                           ("Medium (100K ep)", 100_000),
                           ("Full (1M ep)", 1_000_000)]:
    steps_per_ep = 200
    total = n_episodes * steps_per_ep
    hours = total / steps_per_sec / 3600
    print(f"  {label:25s}: {hours:8.1f} hours ({hours/24:.1f} days)")

print(f"\n  VERDICT:", end=" ")
if steps_per_sec > 500:
    print("RL feasible as-is")
elif steps_per_sec > 50:
    print("Small PoC feasible, full training needs MuJoCo")
elif steps_per_sec > 5:
    print("Only tiny PoC feasible (~1K episodes)")
else:
    print("Too slow even for PoC — MuJoCo needed")

print("\nDone.")