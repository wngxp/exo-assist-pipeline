#!/usr/bin/env python3
"""
Condition 2: Exo Locked MocoInverse — WAWA exoskeleton attached, joints locked.
Tests the metabolic cost of wearing the exoskeleton as dead weight (~2.9 kg).

The 4 exo coordinates (exo_J1_R_angle, exo_J2_R_angle, exo_J1_L_angle, 
exo_J2_L_angle) are locked at their default values. The exo is connected to the
human via BushingForce elements (pelvis, femur_r, femur_l).

The exo's ground-to-Origin_base WeldJoint means the exo base is fixed to ground.
The bushings transfer forces between the exo and human bodies.

Run from: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/exo_locked/
Prereqs:  conda activate opensim
"""

import os
import opensim as osim

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_FILE = os.path.join(DATA_DIR, 'combined_model.osim')
GRF_FILE = os.path.join(DATA_DIR, 'grf_walk.xml')
COORDS_FILE = os.path.join(DATA_DIR, 'coordinates.sto')

OUTPUT_FILE = 'exo_locked_MocoInverse_solution.sto'

# Exoskeleton coordinate names to lock
EXO_COORDS = [
    'exo_J1_R_angle',
    'exo_J2_R_angle',
    'exo_J1_L_angle',
    'exo_J2_L_angle',
]

def solve():
    grf_file = os.path.abspath(GRF_FILE)
    coords_file = os.path.abspath(COORDS_FILE)
    locked_model = os.path.abspath('combined_model_exo_locked.osim')

    inverse = osim.MocoInverse()
    modelProcessor = osim.ModelProcessor(locked_model)
    modelProcessor.append(osim.ModOpAddExternalLoads(grf_file))
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    modelProcessor.append(osim.ModOpAddReserves(1.0))
    inverse.setModel(modelProcessor)

    inverse.setKinematics(osim.TableProcessor(coords_file))
    inverse.set_initial_time(0.48)
    inverse.set_final_time(1.61)
    inverse.set_mesh_interval(0.02)
    inverse.set_kinematics_allow_extra_columns(True)

    print("Solving MocoInverse with exo locked (dead weight)...")
    solution = inverse.solve()
    solution.getMocoSolution().write(OUTPUT_FILE)
    print(f"Solution written to {OUTPUT_FILE}")

    model = modelProcessor.process()
    report = osim.report.Report(model, OUTPUT_FILE, bilateral=True)
    report.generate()
    print("PDF report generated.")
    
if __name__ == '__main__':
    solve()
