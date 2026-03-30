#!/usr/bin/env python3
"""
Stage 4: Multi-Subject MocoTrack with Exoskeleton Assistance
=============================================================

Adapted from exo_mocotrack.py for AddBiomechanics subjects.
First test subject: P010 (54 kg, 1.66 m, walking)

Usage:
    cd ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/multisubject/P010/
    conda activate opensim
    python run_mocotrack_P010.py
"""

import os
import sys
import time as timer
import numpy as np
import xml.etree.ElementTree as ET
import opensim as osim

# ===========================================================================
# CONFIGURATION
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files (all in the P010 folder)
MODEL_FILE = os.path.join(BASE_DIR, 'P010_combined.osim')
GRF_XML_FILE = os.path.join(BASE_DIR, 'grf_walk.xml')
COORDS_FILE = os.path.join(BASE_DIR, 'coordinates.sto')

# WAWA VTP meshes (needed for model loading)
WAWA_VTP_DIR = '/home/wxp/repos/projects/exo-assist-pipeline/opensim/'

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'mocotrack_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time window — use ~2 gait cycles from the 8s recording
# Walking at slow speed: ~1.0-1.2s per cycle, so 2s ≈ 2 cycles
# Start a bit into the trial to skip transients
T_START = 1.0
T_END = 2.0

# Solver settings
MESH_INTERVAL = 0.05      # coarse first, refine to 0.02 later
TRACKING_WEIGHT = 10.0
EFFORT_WEIGHT = 1.0

# Exo torque (CMA-ES best from Stage 2)
PEAK_TORQUE = 12.0   # Nm
ONSET_PHASE = 0.003
DURATION = 0.59

# Reserve actuator strength
RESERVE_STRENGTH = 1.0


# ===========================================================================
# HELPER: Build bell-curve torque profile
# ===========================================================================

def build_torque_timeseries(peak, onset, duration, t_start, t_end, dt=0.004):
    """Generate bell-curve hip assistance torque profile."""
    times = np.arange(t_start, t_end + dt, dt)
    gait_duration = t_end - t_start
    n = len(times)

    torque_R = np.zeros(n)
    torque_L = np.zeros(n)

    for i, t in enumerate(times):
        phase = (t - t_start) / gait_duration

        # Right leg
        if onset <= phase <= onset + duration:
            local_phase = (phase - onset) / duration
            torque_R[i] = peak * np.sin(np.pi * local_phase)

        # Left leg — offset by ~50% gait cycle
        phase_L = (phase + 0.5) % 1.0
        if onset <= phase_L <= onset + duration:
            local_phase = (phase_L - onset) / duration
            torque_L[i] = peak * np.sin(np.pi * local_phase)

    return times, torque_R, torque_L


# ===========================================================================
# STEP 1: Prepare model with exo torque
# ===========================================================================

def prepare_model_with_exo_torque(peak, onset, duration):
    """Lock WAWA structural joints, add hip torque actuators + controller."""
    print(f"Preparing model: peak={peak} Nm, onset={onset:.3f}, dur={duration:.3f}")

    # --- XML surgery: lock exo joints ---
    tree = ET.parse(MODEL_FILE)
    root = tree.getroot()

    # Moco doesn't support locked coordinates — unlock ALL of them.
    # The exo structural joints get clamped to their default value via
    # tight coordinate bounds instead. Foot/wrist coords stay near
    # reference values via tracking weight.
    unlocked = []
    for coord_elem in root.iter('Coordinate'):
        name = coord_elem.get('name', '')
        locked = coord_elem.find('locked')
        if locked is not None and locked.text == 'true':
            locked.text = 'false'
            unlocked.append(name)
            # For exo structural joints, clamp range to default value
            if 'exo_' in name:
                default_val = coord_elem.find('default_value')
                dv = float(default_val.text) if default_val is not None else 0.0
                range_elem = coord_elem.find('range')
                if range_elem is None:
                    range_elem = ET.SubElement(coord_elem, 'range')
                range_elem.text = f'{dv - 0.001} {dv + 0.001}'
    locked_count = len(unlocked)
    if unlocked:
        print(f"  Unlocked {len(unlocked)} coordinates: {unlocked}")

    # Also strip VTP geometry references that won't be found
    # (prevents warnings from flooding the log)
    for mesh_elem in root.iter('Mesh'):
        mesh_file = mesh_elem.find('mesh_file')
        if mesh_file is not None and mesh_file.text:
            # Check if VTP exists
            vtp_name = mesh_file.text
            if not os.path.exists(os.path.join(BASE_DIR, vtp_name)):
                if not os.path.exists(os.path.join(WAWA_VTP_DIR, vtp_name)):
                    mesh_file.text = ''  # clear missing mesh references

    temp_model_path = os.path.join(OUTPUT_DIR, 'P010_combined_locked.osim')
    tree.write(temp_model_path, xml_declaration=True)
    print(f"  Processed {locked_count} previously-locked coordinates")

    # --- Add hip torque actuators ---
    model = osim.Model(temp_model_path)

    for side in ['r', 'l']:
        coord_name = f'hip_flexion_{side}'
        actuator = osim.CoordinateActuator(coord_name)
        actuator.setName(f'exo_hip_{side}')
        actuator.setOptimalForce(1.0)
        actuator.setMinControl(-30.0)
        actuator.setMaxControl(30.0)
        model.addForce(actuator)

    # --- Add prescribed controller ---
    times, torque_R, torque_L = build_torque_timeseries(
        peak, onset, duration, T_START, T_END)

    controller = osim.PrescribedController()
    controller.setName('exo_controller')

    spline_R = osim.PiecewiseLinearFunction()
    spline_L = osim.PiecewiseLinearFunction()
    for i in range(len(times)):
        spline_R.addPoint(float(times[i]), float(torque_R[i]))
        spline_L.addPoint(float(times[i]), float(torque_L[i]))

    controller.addActuator(model.getComponent('forceset/exo_hip_r'))
    controller.prescribeControlForActuator('exo_hip_r', spline_R)

    controller.addActuator(model.getComponent('forceset/exo_hip_l'))
    controller.prescribeControlForActuator('exo_hip_l', spline_L)

    model.addController(controller)

    final_path = os.path.join(OUTPUT_DIR, 'P010_track_ready.osim')
    model.finalizeConnections()
    model.printToXML(final_path)
    print(f"  Final model: {final_path}")
    return final_path


# ===========================================================================
# STEP 2: Run MocoTrack
# ===========================================================================

def run_mocotrack(model_path):
    """Run MocoTrack forward dynamics with exo assistance."""
    print("\n" + "=" * 60)
    print("Running MocoTrack (Forward Dynamics) — P010")
    print("=" * 60)

    t0 = timer.time()

    track = osim.MocoTrack()
    track.setName('P010_exo_track')

    # --- Model processor ---
    modelProcessor = osim.ModelProcessor(model_path)
    modelProcessor.append(osim.ModOpAddExternalLoads(
        os.path.abspath(GRF_XML_FILE)))
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    modelProcessor.append(osim.ModOpAddReserves(RESERVE_STRENGTH))

    track.setModel(modelProcessor)

    # --- Reference kinematics ---
    tableProcessor = osim.TableProcessor(os.path.abspath(COORDS_FILE))
    tableProcessor.append(osim.TabOpLowPassFilter(6))
    # tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
    track.setStatesReference(tableProcessor)

    track.set_states_global_tracking_weight(TRACKING_WEIGHT)
    track.set_allow_unused_references(True)

    # --- Time window ---
    track.set_initial_time(T_START)
    track.set_final_time(T_END)
    track.set_mesh_interval(MESH_INTERVAL)

    # --- Initialize and configure ---
    study = track.initialize()
    problem = study.updProblem()

    # Set effort goal weights
    effort = osim.MocoControlGoal.safeDownCast(
        problem.updGoal('control_effort'))
    if effort is not None:
        effort.setWeight(EFFORT_WEIGHT)
        # Don't penalize exo actuators
        effort.setWeightForControl('/forceset/exo_hip_r', 0.0)
        effort.setWeightForControl('/forceset/exo_hip_l', 0.0)

    # Configure solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_optim_convergence_tolerance(1e-2)
    solver.set_optim_constraint_tolerance(1e-2)
    solver.set_optim_max_iterations(1000)

    # --- Solve ---
    print(f"\n  Time window:     {T_START} – {T_END} s ({T_END-T_START:.1f}s)")
    print(f"  Mesh interval:   {MESH_INTERVAL} s")
    print(f"  Tracking weight: {TRACKING_WEIGHT}")
    print(f"  Exo peak torque: {PEAK_TORQUE} Nm")
    print()

    solution = study.solve()
    solve_time = timer.time() - t0
    print(f"\nSolved in {solve_time:.1f}s ({solve_time/60:.1f} min)")

    if solution.isSealed():
        print("WARNING: Solution sealed (solver may have failed). Unsealing...")
        solution.unseal()

    solution_path = os.path.join(OUTPUT_DIR, 'P010_track_solution.sto')
    solution.write(solution_path)
    print(f"Solution: {solution_path}")

    return solution, solution_path, solve_time


# ===========================================================================
# STEP 3: Analyze — extract hip muscle effort
# ===========================================================================

def analyze_results(solution_path):
    """Extract muscle activations and compute effort metric."""
    print("\n" + "=" * 60)
    print("Analyzing Results — P010")
    print("=" * 60)

    HIP_MUSCLES = [
        'glut_max1_r', 'glut_max2_r', 'glut_max3_r',
        'glut_med1_r', 'glut_med2_r', 'glut_med3_r',
        'psoas_r', 'iliacus_r',
        'rect_fem_r', 'semimem_r', 'semiten_r', 'bifemlh_r',
        'glut_max1_l', 'glut_max2_l', 'glut_max3_l',
        'glut_med1_l', 'glut_med2_l', 'glut_med3_l',
        'psoas_l', 'iliacus_l',
        'rect_fem_l', 'semimem_l', 'semiten_l', 'bifemlh_l',
    ]

    track_table = osim.TimeSeriesTable(solution_path)
    track_labels = list(track_table.getColumnLabels())

    total_effort = 0.0
    n_muscles = 0

    print(f"\n  {'Muscle':<25s} {'Mean Act²':>10s}")
    print(f"  {'-'*25} {'-'*10}")

    for muscle in HIP_MUSCLES:
        # Try state format (MocoTrack)
        col_name = f'/forceset/{muscle}/activation'
        if col_name not in track_labels:
            col_name = f'/forceset/{muscle}'
            if col_name not in track_labels:
                continue

        col_data = track_table.getDependentColumn(col_name)
        data = np.array([col_data[i] for i in range(track_table.getNumRows())])
        effort = np.mean(data ** 2)
        total_effort += effort
        n_muscles += 1
        print(f"  {muscle:<25s} {effort:10.6f}")

    if n_muscles > 0:
        avg_effort = total_effort / n_muscles
        print(f"\n  Average hip muscle effort: {avg_effort:.6f}")
        print(f"  Muscles analyzed: {n_muscles}")
        return {'avg_effort': avg_effort, 'n_muscles': n_muscles}
    else:
        print("  WARNING: No matching muscle columns found!")
        print(f"  Available columns (first 15): {track_labels[:15]}")
        return None


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Stage 4: Multi-Subject MocoTrack — P010")
    print(f"  Subject: 54.0 kg, 1.66 m")
    print(f"  Trial: walk_slow_1_segment_0")
    print("=" * 60)

    # Step 1: Prepare model
    model_path = prepare_model_with_exo_torque(
        PEAK_TORQUE, ONSET_PHASE, DURATION)

    # Step 2: Run MocoTrack
    solution, solution_path, solve_time = run_mocotrack(model_path)

    # Step 3: Analyze
    results = analyze_results(solution_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — P010")
    print("=" * 60)
    if results:
        print(f"  Hip muscle effort: {results['avg_effort']:.6f}")
        print(f"  Muscles analyzed:  {results['n_muscles']}")
    print(f"  Solve time:        {solve_time:.1f}s ({solve_time/60:.1f} min)")
    print(f"  Solution:          {solution_path}")
    print()
    print("Next: Run baseline (no exo) MocoTrack on same subject to compare.")
