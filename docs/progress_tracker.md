# WAWA Exoskeleton RL Control — Progress Tracker

## Big Picture
Replicate & extend Luo et al. Nature 2024: train RL policy in simulation → deploy on WAWA bilateral hip exo. Two deployed modules: (1) activity classifier (CNN, Yang's), (2) RL torque policy (ours). OpenSim simulation stack → eventually RL training.

## Pipeline Stages

| Stage | Description | Status | Key Result |
|-------|-------------|--------|------------|
| 1 | Baseline MocoInverse (human only) | ✅ Done | Reference muscle activations |
| 2 | CMA-ES torque optimization | ✅ Done | 12 Nm peak, 27.4% effort reduction |
| 3 | MocoTrack (forward dynamics, single subject) | ✅ Done | 19.7% effort reduction |
| 4 | Multi-subject (AddBiomechanics) | 🔄 In progress | P010 baseline solved, exo-assisted TBD |
| 5 | RL policy training | ⬜ Not started | — |

## Session Log

### Session 1–5 (Feb–early Mar)
- URDF→OSIM conversion, Z-up→Y-up fix, Rodrigues rotation for joint axes
- Combined model: Rajagopal + WAWA via WeldJoint + BushingForce
- Baseline MocoInverse solved (~21 iterations)
- XML surgery pattern established (edit XML before loading, not after initSystem)

### Session 6–8 (Mar 10–14)
- Condition 2 (exo_locked): negligible effect at 2.9 kg — constraint forces absorb mass
- Condition 3 (exo_active): PrescribedController with bell-curve torque
- CMA-ES optimization: 12 Nm peak, onset=0.003, dur=0.59 → 27.4% reduction
- MocoTrack (Stage 3): forward dynamics, 19.7% reduction, 21 min solve

### Session 9 (Mar 19 — Friday)
- Discussed AddBiomechanics dataset, confirmed same Rajagopal model
- Started 389 GB download (wget -c), installed nimblephysics
- Built MocoTrack script, got first successful MocoTrack result
- Navid confirmed 30 Nm hardware max (software-limited to 12 Nm)
- Git pushed Stage 3 results

### Session 10 (Mar 23 — Monday)  
- Unzipped AddBiomechanics (disk full at 77%, deleted zip → 389 GB freed)
- Dataset scan: 731 .b3d files, 210 subjects, 603 with GRF, 397 37-DOF
- explore_addbiomechanics.py → dataset_summary.csv + walking_candidates.csv

### Session 11 (Mar 24 — Tuesday)
- Extracted P010 (54 kg, 1.66 m, Carter2023) from .b3d:
  - P010_scaled.osim, coordinates.sto (absolute state names), grf_walk.mot/xml
- Merged WAWA onto P010 — alignment correct without retuning
- Discovered: AddBiomechanics has locked coords (subtalar, mtp, wrist) → must unlock for Moco
- Discovered: muscle names differ (glmax1 not glut_max1, recfem not rect_fem, bflh not bifemlh)
- MocoTrack on combined model segfaults → switched to MocoInverse approach
- P010 baseline MocoInverse solved (891s)
- P010 exo-assisted MocoInverse launched but may have errored (check ~/P010_exo_inverse.log)

## Key Learnings / Gotchas
- Moco doesn't support locked coordinates → unlock via XML surgery, clamp with tight range
- MocoTrack + combined model (BushingForce) segfaults in CasADi → use MocoInverse on human-only model + PrescribedController instead
- AddBiomechanics muscle naming: glmax1, glmed1, recfem, bflh, semimem, semiten
- Coordinates.sto needs absolute state names (/jointset/hip_r/hip_flexion_r/value)
- TabOpUseAbsoluteStateNames() double-converts if names already absolute → comment out
- conda opensim lacks simbody-visualizer — use matplotlib + scp for plots
- VTP mesh warnings are cosmetic — don't affect simulation
- cortex: 64 cores, 251 GB RAM, RTX A5000, NVMe SSD, network-throttled

## Next Steps
1. Check if P010 exo-assisted MocoInverse finished (~/P010_exo_inverse.log)
2. If not: re-run exo-assisted MocoInverse on P010 (use P010_scaled_unlocked.osim + PrescribedController)
3. Compare baseline vs exo muscle effort for P010
4. Wrap pipeline into reusable script: extract_subject.py → run both MocoInverse → compare
5. Run on 5–10 diverse subjects (vary mass 50–100 kg, height 1.5–1.9 m)
6. Collect results table: subject | mass | height | baseline_effort | exo_effort | reduction%
7. Discuss with Yang: MocoInverse multi-subject data for RL (Path B) vs live RL loop (Path A)

## File Locations (cortex)
- Repo: ~/repos/projects/exo-assist-pipeline/
- Combined model: opensim/combined_model.osim
- Moco experiments: opensim/moco-experiments/
- AddBiomechanics data: data/addbiomechanics/ (731 .b3d files)
- P010 files: opensim/moco-experiments/multisubject/P010/
- Dataset CSVs: data/addbiomechanics/dataset_summary.csv, walking_candidates.csv
