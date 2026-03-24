#!/usr/bin/env python3
"""
Condition 3A: Exo Active (Prescribed Torque) MocoInverse
Applies a prescribed hip assistance torque profile to reduce muscle effort.

Strategy: Read the baseline hip kinematics, generate an assistance torque
profile that assists hip flexion/extension, write it as a .sto controls file,
and apply it via CoordinateActuators added to the model.

The assistance torque follows a simple pattern:
  - Proportional to hip angular velocity (velocity-based assistance)
  - This naturally assists extension during stance and flexion during swing
  - Peak ~8 Nm (consistent with Nature paper values)

Run from: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/exo_active/
Prereqs:  conda activate opensim
"""

import os
import numpy as np
import opensim as osim
import ctypes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_FILE = os.path.join(DATA_DIR, 'subject_walk_scaled.osim')
GRF_FILE = os.path.join(DATA_DIR, 'grf_walk.xml')
COORDS_FILE = os.path.join(DATA_DIR, 'coordinates.sto')
FBPS_FILE = os.path.join(DATA_DIR, '..', 'moco-tutorial', 'example3DWalking',
                         'subject_walk_scaled_FunctionBasedPathSet.xml')

OUTPUT_FILE = 'exo_active_MocoInverse_solution.sto'

# Assistance parameters
PEAK_TORQUE = 8.0  # Nm — peak assistance torque per hip
EXO_MASS_KG = 2.9  # total exo mass to add to human bodies


def read_sto(filepath):
    """Read .sto file, return header count, column names, and data."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.strip() == 'endheader':
                header_lines = i + 1
                break
    data = np.genfromtxt(filepath, skip_header=header_lines,
                         names=True, delimiter='\t', deletechars='')
    return data


def generate_assistance_torque(coords_data):
    """
    Generate hip assistance torque profile from kinematics.
    
    Uses velocity-based assistance: torque = gain * hip_angular_velocity
    This naturally:
      - Assists hip extension during stance (when hip is extending)
      - Assists hip flexion during swing (when hip is flexing)
    
    The gain is scaled so peak torque matches PEAK_TORQUE.
    """
    time = coords_data[coords_data.dtype.names[0]]
    
    # Get hip flexion speeds (from coordinates.sto if available, else differentiate)
    names = coords_data.dtype.names
    
    # Find hip flexion angle columns
    hip_r_col = None
    hip_l_col = None
    for n in names:
        if 'hip_flexion_r' in n and 'speed' not in n and 'value' not in n:
            hip_r_col = n
        elif 'hip_flexion_r' in n and 'value' not in n:
            pass  # skip speed columns for now
    
    # Just use the angle columns and differentiate
    hip_r_angle = None
    hip_l_angle = None
    for n in names:
        if 'hip_flexion_r/value' in n:
            hip_r_angle = coords_data[n]
        elif 'hip_flexion_l/value' in n:
            hip_l_angle = coords_data[n]
    
    if hip_r_angle is None or hip_l_angle is None:
        raise RuntimeError("Could not find hip flexion columns in coordinates.sto")
    
    # Numerical differentiation for angular velocity
    dt = np.diff(time)
    hip_r_vel = np.gradient(hip_r_angle, time)  # rad/s
    hip_l_vel = np.gradient(hip_l_angle, time)  # rad/s
    
    # Scale gain so peak torque = PEAK_TORQUE
    max_vel = max(np.max(np.abs(hip_r_vel)), np.max(np.abs(hip_l_vel)))
    gain = PEAK_TORQUE / max_vel if max_vel > 0 else 0
    
    torque_r = gain * hip_r_vel  # positive = flexion assistance
    torque_l = gain * hip_l_vel
    
    print(f"Assistance torque profile:")
    print(f"  Gain: {gain:.3f} Nm/(rad/s)")
    print(f"  Right hip: peak={np.max(np.abs(torque_r)):.1f} Nm")
    print(f"  Left hip:  peak={np.max(np.abs(torque_l)):.1f} Nm")
    
    return time, torque_r, torque_l


def write_assistance_sto(time, torque_r, torque_l, filename='assistance_torques.sto'):
    """Write assistance torques as an OpenSim .sto file."""
    with open(filename, 'w') as f:
        f.write(f"{filename}\n")
        f.write("version=1\n")
        f.write(f"nRows={len(time)}\n")
        f.write("nColumns=3\n")
        f.write("inDegrees=no\n")
        f.write("endheader\n")
        f.write("time\texo_hip_assist_r\texo_hip_assist_l\n")
        for t, tr, tl in zip(time, torque_r, torque_l):
            f.write(f"{t}\t{tr}\t{tl}\n")
    print(f"Written: {filename}")


def solve():
    model_file = os.path.abspath(MODEL_FILE)
    grf_file = os.path.abspath(GRF_FILE)
    coords_file = os.path.abspath(COORDS_FILE)
    fbps_file = os.path.abspath(FBPS_FILE)
    
    # -----------------------------------------------------------------------
    # Step 1: Generate assistance torque profile from kinematics
    # -----------------------------------------------------------------------
    coords_data = read_sto(coords_file)
    time, torque_r, torque_l = generate_assistance_torque(coords_data)
    write_assistance_sto(time, torque_r, torque_l)
    
    # -----------------------------------------------------------------------
    # Step 2: Build model with exo mass + assistance actuators
    # -----------------------------------------------------------------------
    # Start from human model, add exo mass to bodies
    model = osim.Model(model_file)
    
    # Add exo mass
    pelvis = model.getBodySet().get('pelvis')
    femur_r = model.getBodySet().get('femur_r')
    femur_l = model.getBodySet().get('femur_l')
    pelvis.setMass(pelvis.getMass() + 2.32)
    femur_r.setMass(femur_r.getMass() + 0.29)
    femur_l.setMass(femur_l.getMass() + 0.29)
    print(f"\nAdded exo mass: pelvis +2.32, femur_r +0.29, femur_l +0.29 kg")
    
    # Add CoordinateActuators for hip assistance
    # These will have PRESCRIBED (not optimized) torques
    for side, coord_name in [('r', 'hip_flexion_r'), ('l', 'hip_flexion_l')]:
        actuator = osim.CoordinateActuator()
        actuator.setName(f'exo_hip_assist_{side}')
        actuator.set_coordinate(coord_name)
        actuator.setOptimalForce(PEAK_TORQUE)
        actuator.setMinControl(-1.0)
        actuator.setMaxControl(1.0)
        model.addForce(actuator)
        print(f"Added CoordinateActuator: exo_hip_assist_{side} on {coord_name}")
    
    # Save modified model
    temp_model = os.path.abspath('human_with_exo_assistance.osim')
    model.finalizeConnections()
    model.printToXML(temp_model)
    print(f"Saved model: {temp_model}")
    
    # -----------------------------------------------------------------------
    # Step 3: Run MocoInverse with prescribed exo controls
    # -----------------------------------------------------------------------
    inverse = osim.MocoInverse()
    
    modelProcessor = osim.ModelProcessor(temp_model)
    modelProcessor.append(osim.ModOpAddExternalLoads(grf_file))
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    if os.path.exists(fbps_file):
        modelProcessor.append(
            osim.ModOpReplacePathsWithFunctionBasedPaths(fbps_file))
        print("Using FunctionBasedPathSet")
    modelProcessor.append(osim.ModOpAddReserves(1.0))
    inverse.setModel(modelProcessor)
    
    inverse.setKinematics(osim.TableProcessor(coords_file))
    inverse.set_initial_time(0.48)
    inverse.set_final_time(1.61)
    inverse.set_mesh_interval(0.02)
    inverse.set_kinematics_allow_extra_columns(True)
    
    # Initialize the study to add prescribed control constraint
    study = inverse.initialize()
    problem = study.updProblem()
    
    # Load assistance torque table
    assist_table = osim.TimeSeriesTable('assistance_torques.sto')
    
    # Prescribe the exo actuator controls using the generated torque profile
    # The control value is torque / optimal_force (normalized to [-1, 1])
    prescribedR = osim.MocoControlBoundConstraint()
    prescribedR.setName('prescribe_exo_r')
    prescribedR.addControlPath('/forceset/exo_hip_assist_r')
    
    prescribedL = osim.MocoControlBoundConstraint()
    prescribedL.setName('prescribe_exo_l')
    prescribedL.addControlPath('/forceset/exo_hip_assist_l')
    
    # Create GCVSpline for the normalized torque profiles
    time_vec = assist_table.getIndependentColumn()
    torque_r_col = assist_table.getDependentColumn('exo_hip_assist_r')
    torque_l_col = assist_table.getDependentColumn('exo_hip_assist_l')
    
    # Normalize by optimal force
    t_list = [float(t) for t in time_vec]
    r_list = [float(torque_r_col[i]) / PEAK_TORQUE for i in range(len(time_vec))]
    l_list = [float(torque_l_col[i]) / PEAK_TORQUE for i in range(len(time_vec))]

    spline_r = osim.PiecewiseLinearFunction()
    spline_l = osim.PiecewiseLinearFunction()
    for i in range(len(t_list)):
        spline_r.addPoint(t_list[i], r_list[i])
        spline_l.addPoint(t_list[i], l_list[i])

    prescribedR.setUpperBound(spline_r)
    prescribedR.setLowerBound(spline_r)
    prescribedL.setUpperBound(spline_l)
    prescribedL.setLowerBound(spline_l)
    
    problem.addPathConstraint(prescribedR)
    problem.addPathConstraint(prescribedL)
    
    print("\nSolving MocoInverse with prescribed exo assistance...")
    solution = study.solve()
    solution.write(OUTPUT_FILE)
    print(f"Solution written to {OUTPUT_FILE}")
    
    # Generate PDF report
    model2 = modelProcessor.process()
    report = osim.report.Report(model2, OUTPUT_FILE, bilateral=True)
    report.generate()
    print("PDF report generated.")


if __name__ == '__main__':
    solve()