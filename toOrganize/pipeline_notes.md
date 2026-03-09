# Exoskeleton-Human Model Pipeline

## Overview

Replicate & extend Luo et al. Nature 2024 — RL-trained hip exoskeleton control.
Goal: merge WAWA exoskeleton with Rajagopal musculoskeletal model → run Moco → build RL wrapper.

---

## Pipeline Steps

### 1. Obtain SolidWorks Export (Huang)

- Huang exports WAWA CAD → URDF via SolidWorks URDF Exporter
- Raw files saved to `repos/WAWA/` (backup, do not edit)
- **Ask Huang to verify:** joint axes, joint origins (center of rotation vs surface), actuator torque limits, link masses

### 2. ROS1 → ROS2 Conversion

- Copy URDF + meshes into ROS2 package: `repos/ros2_ws/wawa_description/`
- Update mesh paths: `package://FES_urdf_0306/` → `package://wawa_description/`
- Fix any mass/inertia issues (e.g. J1_L near-zero mass)

### 3. Verify in RViz

- Launch RViz with `joint_state_publisher_gui`
- Huang confirms: joint axes rotate correctly, mesh alignment looks right
- **Do this before OpenSim conversion** — catches axis errors early

### 4. Fix URDF Joint Axes

- SolidWorks exporter produces non-principal axes (e.g. `0.00011 -0.998 -0.058` instead of `0 -1 0`)
- Manually edit axis vectors to nearest principal axis:
  - J1 (hip flexion): `0 -1 0` (R), `0 1 0` (L)
  - J2 (hip adduction): `1 0 0` (R), `1 0 0` (L)
- Re-verify in RViz after editing

### 5. Convert Meshes: STL → VTP

- Script: `repos/exo-assist-pipeline/opensim/convert_stl_to_vtp_simbody.py`
- Input: `.STL` files from `wawa_description/meshes/`
- Output: `.vtp` files for OpenSim

### 6. Convert URDF → .osim

- Script: `repos/exo-assist-pipeline/opensim/convert_urdf_to_osim.py`
- Applies 90° X-axis rotation (URDF Z-up → OpenSim Y-up)
- Output: `.osim` exoskeleton model

### 7. Visualize Alignment

- Script: `repos/exo-assist-pipeline/opensim/visualize_models.py`
- Plots body frame origins of both models in 3D
- Prints bounding boxes + per-body coordinates
- Use to calculate translation offsets before merging

### 8. Merge Exo + Human Model

- Script: `repos/exo-assist-pipeline/opensim/merge_exo_human.py`
- Attaches exo to human via WeldJoint (pelvis) + BushingForces (pelvis, femur_r, femur_l)
- Tunable parameters at top of script: translation offset, rotation, bushing stiffness/damping
- Output: `combined_model.osim`

### 9. Validate in OpenSim GUI

- scp `combined_model.osim` + `.vtp` files + `Geometry/` folder to Mac
- Check: exo visually attached, joints rotate correctly, no initialization errors

### 10. Run MocoInverse (with exoskeleton)

- Use same walking data as baseline (from `example3DWalking/`)
- Compare muscle activations: with vs without exo
- Check reserve actuators — should remain small

### 11. (Future) RL Training Wrapper

- OpenSim Python API + PyTorch (stable-baselines3)
- Switch to MuJoCo only if OpenSim too slow on cortex

---

## Repo Structure

```
repos/
├── WAWA/                          # Raw Huang exports (read-only backup)
│   └── urdf/
├── ros2_ws/
│   └── wawa_description/          # ROS2 package (RViz verification)
│       ├── urdf/
│       └── meshes/
└── exo-assist-pipeline/
    └── opensim/
        ├── convert_urdf_to_osim.py
        ├── convert_stl_to_vtp_simbody.py
        ├── visualize_models.py
        ├── merge_exo_human.py
        ├── combined_model.osim     # merged output
        ├── *.vtp                   # exo meshes
        └── moco-tutorial/
            └── example3DWalking/   # Rajagopal model + walking data
```

---

## Key Gotchas

- **conda env:** always `conda activate opensim` before running scripts
- **No simbody-visualizer in conda OpenSim:** comment out `visualize()` / `solveAndVisualize()` calls
- **X11 through tmux is unreliable:** save plots as PNG, scp to Mac
- **Mesh paths:** VTP files must be in same directory as the .osim that references them
- **Tsinghua pip mirror:** `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <pkg>`
- **GitHub is slow** from cortex; docker.io and Google are blocked

---

## Questions for Huang (Next Visit)

- [ ] Fix joint axes in SolidWorks before re-exporting
- [ ] Confirm J1 = flexion, J2 = adduction
- [ ] Joint origin = center of rotation or outer surface?
- [ ] Actual actuator torque limit (URDF: 10 Nm, Nature paper: 18 Nm)?
- [ ] Why is J1_L mass ~0 in original export?
- [ ] Real joint ROM limits?

---

## References

- Luo et al. Nature 2024 — doi:10.1038/s41586-024-07382-4
- Pseudocode repo — github.com/IntelligentRobotLearning/pseudocode_learning_in_simulation
- Project repo — github.com/wngxp/exo-assist-pipeline
