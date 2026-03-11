#!/usr/bin/env python3
"""
Condition 1: Baseline MocoInverse — human only, no exoskeleton.
Reproduces the standard MocoInverse from the tutorial using subject_walk_scaled.osim.

Run from: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/baseline/
Prereqs:  conda activate opensim
"""

import os
import opensim as osim

# ---------------------------------------------------------------------------
# Paths — data files are one level up in moco-experiments/
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_FILE = os.path.join(DATA_DIR, 'subject_walk_scaled.osim')
GRF_FILE = os.path.join(DATA_DIR, 'grf_walk.xml')
COORDS_FILE = os.path.join(DATA_DIR, 'coordinates.sto')
FBPS_FILE = os.path.join(DATA_DIR, '..', 'moco-tutorial', 'example3DWalking',
                         'subject_walk_scaled_FunctionBasedPathSet.xml')

OUTPUT_FILE = 'baseline_MocoInverse_solution.sto'

def solve():
    inverse = osim.MocoInverse()

    # Model setup — same as tutorial
    modelProcessor = osim.ModelProcessor(MODEL_FILE)
    modelProcessor.append(osim.ModOpAddExternalLoads(GRF_FILE))
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    # Use FunctionBasedPaths for speed
    if os.path.exists(FBPS_FILE):
        modelProcessor.append(
            osim.ModOpReplacePathsWithFunctionBasedPaths(FBPS_FILE))
        print(f"Using FunctionBasedPathSet: {FBPS_FILE}")
    else:
        print("WARNING: FunctionBasedPathSet not found, using GeometryPath.")
    modelProcessor.append(osim.ModOpAddReserves(1.0))
    inverse.setModel(modelProcessor)

    # Kinematics
    inverse.setKinematics(osim.TableProcessor(COORDS_FILE))
    inverse.set_initial_time(0.48)
    inverse.set_final_time(1.61)
    inverse.set_mesh_interval(0.02)
    inverse.set_kinematics_allow_extra_columns(True)

    # Solve
    print("Solving baseline MocoInverse (human only)...")
    solution = inverse.solve()
    solution.getMocoSolution().write(OUTPUT_FILE)
    print(f"Solution written to {OUTPUT_FILE}")

    # Generate PDF report
    model = modelProcessor.process()
    report = osim.report.Report(model, OUTPUT_FILE, bilateral=True)
    report.generate()
    print("PDF report generated.")

if __name__ == '__main__':
    solve()
