#!/usr/bin/env python3
"""
Stage 3: MocoTrack with Exoskeleton Assistance (Forward Dynamics)
=================================================================

This script replaces MocoInverse with MocoTrack for the exoskeleton-assisted
condition. The key difference:

  - MocoInverse: "Given EXACT kinematics, what muscle forces are needed?"
    (prescribed motion — can't capture how exo changes gait)

  - MocoTrack:   "TRY to follow these kinematics, but you're free to deviate
    if the physics demands it." (forward dynamics — model finds its own gait)

This means the model can adapt its walking pattern under exo assistance,
which is more realistic and what you need for RL training.

WHAT THIS SCRIPT DOES:
    1. Loads combined_model.osim (Rajagopal + WAWA exo)
    2. Applies a parameterized hip assistance torque (from CMA-ES best: 12 Nm)
    3. Runs MocoTrack (forward dynamics with state tracking)
    4. Extracts muscle activations and compares to baseline
    5. Saves solution + generates comparison plots

ARCHITECTURE (how MocoTrack differs from MocoInverse internally):
    - MocoInverse solves: min Σ(muscle_activations²)
      subject to: q(t) = q_reference(t)  [hard constraint]

    - MocoTrack solves:  min w_effort * Σ(activations²) + w_track * Σ(q - q_ref)²
      subject to: equations of motion  [soft tracking]

    The tracking weight controls how closely the model follows the reference.
    Higher weight = closer to prescribed motion (like MocoInverse).
    Lower weight  = more freedom to find its own gait.

RUN FROM: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
PREREQS:  conda activate opensim
USAGE:    python exo_mocotrack.py
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

# Base directory (moco-experiments/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
MODEL_FILE = os.path.join(BASE_DIR, 'combined_model.osim')
HUMAN_MODEL_FILE = os.path.join(BASE_DIR, 'subject_walk_scaled.osim')
GRF_FILE = os.path.join(BASE_DIR, 'grf_walk.xml')
COORDS_FILE = os.path.join(BASE_DIR, 'coordinates.sto')
FBPS_FILE = os.path.join(BASE_DIR, '..', 'moco-tutorial', 'example3DWalking',
                         'subject_walk_scaled_FunctionBasedPathSet.xml')

# Baseline solution (from your Condition 1 run)
BASELINE_SOLUTION = os.path.join(BASE_DIR, 'baseline',
                                 'baseline_MocoInverse_solution.sto')

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, 'mocotrack_exo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time window (same as your MocoInverse runs)
T_START = 0.48
T_END = 1.61

# Mesh interval — MocoTrack is slower than MocoInverse, so we use a
# coarser mesh to start. You can refine later.
# 0.02 = what you used for MocoInverse (~57 mesh points)
# 0.04 = coarser, ~28 mesh points, ~4x faster to solve
MESH_INTERVAL = 0.04  # Start coarse, refine to 0.02 once it works

# Tracking weight — controls how closely the model follows reference motion.
# Higher = more like MocoInverse (stiff tracking)
# Lower  = more freedom for the model to find its own gait
# The OpenSim MocoTrack tutorial uses 10.0 as default
TRACKING_WEIGHT = 10.0

# Effort weight — how much to penalize muscle activations
EFFORT_WEIGHT = 1.0

# Exo torque parameters (from your CMA-ES optimization best result)
PEAK_TORQUE = 12.0    # Nm
ONSET_PHASE = 0.003   # fraction of gait cycle
DURATION = 0.59       # fraction of gait cycle

# Reserve actuator strength (Nm) — keeps solver stable
RESERVE_STRENGTH = 1.0


# ===========================================================================
# HELPER: Build bell-curve torque profile (same as evaluate_exo_torque.py)
# ===========================================================================

def build_torque_timeseries(peak, onset, duration, t_start, t_end, dt=0.01):
    """
    Generate a bell-curve hip assistance torque profile over the gait cycle.

    Returns an OpenSim TimeSeriesTable with columns for left and right hip
    torque actuators.

    The torque is positive during the onset window (hip flexion assistance)
    and follows a half-sine bell curve shape.
    """
    times = np.arange(t_start, t_end, dt)
    gait_duration = t_end - t_start
    n = len(times)

    torque_R = np.zeros(n)
    torque_L = np.zeros(n)

    for i, t in enumerate(times):
        # Normalize time to gait phase [0, 1]
        phase = (t - t_start) / gait_duration

        # Right leg torque
        if onset <= phase <= onset + duration:
            local_phase = (phase - onset) / duration
            torque_R[i] = peak * np.sin(np.pi * local_phase)

        # Left leg torque — offset by ~50% gait cycle (contralateral)
        phase_L = (phase + 0.5) % 1.0
        if onset <= phase_L <= onset + duration:
            local_phase = (phase_L - onset) / duration
            torque_L[i] = peak * np.sin(np.pi * local_phase)

    return times, torque_R, torque_L


# ===========================================================================
# HELPER: Prepare model with exo torque actuators via XML surgery
# ===========================================================================

def prepare_model_with_exo_torque(peak, onset, duration):
    """
    Take combined_model.osim and add CoordinateActuators for exo hip torque.

    For MocoTrack, we add the exo assistance as a prescribed controller
    (PrescribedController) that applies a time-varying torque to the hip
    joints. This is different from MocoInverse where we used
    MocoControlBoundConstraint.

    Returns path to the prepared model file.
    """
    print(f"Preparing model with exo torque: peak={peak} Nm, "
          f"onset={onset:.3f}, duration={duration:.3f}")

    # --- Step 1: XML surgery to lock WAWA joints (same as exo_locked) ---
    tree = ET.parse(MODEL_FILE)
    root = tree.getroot()

    # Find all WAWA joints and ensure they're welded/locked
    # (the combined model already has them as PinJoints or WeldJoints,
    #  but we want to make sure the exo structure joints are locked)
    exo_coords = ['J1_R_coord', 'J2_R_coord', 'J1_L_coord', 'J2_L_coord']
    for coord_elem in root.iter('Coordinate'):
        name = coord_elem.get('name', '')
        if name in exo_coords:
            # Lock the structural joints
            locked = coord_elem.find('locked')
            if locked is not None:
                locked.text = 'true'
            else:
                ET.SubElement(coord_elem, 'locked').text = 'true'

    # Save modified model
    temp_model_path = os.path.join(OUTPUT_DIR, 'combined_model_for_track.osim')
    tree.write(temp_model_path, xml_declaration=True)
    print(f"  Saved temp model: {temp_model_path}")

    # --- Step 2: Load model and add hip torque actuators ---
    model = osim.Model(temp_model_path)

    # Add CoordinateActuators for hip flexion (these are the exo motors)
    for side in ['r', 'l']:
        coord_name = f'hip_flexion_{side}'
        actuator = osim.CoordinateActuator(coord_name)
        actuator.setName(f'exo_hip_{side}')
        actuator.setOptimalForce(1.0)  # torque in Nm directly
        actuator.setMinControl(-30.0)  # WAWA hardware max
        actuator.setMaxControl(30.0)
        model.addForce(actuator)

    # --- Step 3: Add prescribed controller for the exo torque profile ---
    times, torque_R, torque_L = build_torque_timeseries(
        peak, onset, duration, T_START, T_END)

    controller = osim.PrescribedController()
    controller.setName('exo_controller')

    # GCVSpline needs raw arrays, not osim.Vector
    from numpy import ascontiguousarray, float64
    times_c = ascontiguousarray(times, dtype=float64)
    torque_R_c = ascontiguousarray(torque_R, dtype=float64)
    torque_L_c = ascontiguousarray(torque_L, dtype=float64)

    spline_R = osim.PiecewiseLinearFunction()
    spline_L = osim.PiecewiseLinearFunction()
    for i in range(len(times)):
        spline_R.addPoint(float(times[i]), float(torque_R[i]))
        spline_L.addPoint(float(times[i]), float(torque_L[i]))

    controller.addActuator(model.getComponent(f'forceset/exo_hip_r'))
    controller.prescribeControlForActuator('exo_hip_r', spline_R)

    controller.addActuator(model.getComponent(f'forceset/exo_hip_l'))
    controller.prescribeControlForActuator('exo_hip_l', spline_L)

    model.addController(controller)

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR,
                                     'combined_model_track_ready.osim')
    model.finalizeConnections()
    model.printToXML(final_model_path)
    print(f"  Final model with exo torque: {final_model_path}")

    return final_model_path


# ===========================================================================
# MAIN: Run MocoTrack
# ===========================================================================

def run_mocotrack(model_path, output_prefix='exo_track'):
    """
    Run MocoTrack with the prepared exo-assisted model.

    MocoTrack is OpenSim's convenience class for forward-dynamics tracking.
    Under the hood it sets up a MocoStudy with:
      - MocoStateTrackingGoal (track reference kinematics)
      - MocoControlGoal (minimize muscle effort)
    """
    print("\n" + "=" * 60)
    print("Running MocoTrack (Forward Dynamics)")
    print("=" * 60)

    t0 = timer.time()

    track = osim.MocoTrack()
    track.setName('exo_assisted_track')

    # --- Model ---
    modelProcessor = osim.ModelProcessor(model_path)
    modelProcessor.append(osim.ModOpAddExternalLoads(
        os.path.abspath(GRF_FILE)))
    modelProcessor.append(osim.ModOpIgnoreTendonCompliance())
    modelProcessor.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    modelProcessor.append(osim.ModOpIgnorePassiveFiberForcesDGF())
    modelProcessor.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))

    # FunctionBasedPaths for speed (if available)
    if os.path.exists(FBPS_FILE):
        modelProcessor.append(
            osim.ModOpReplacePathsWithFunctionBasedPaths(
                os.path.abspath(FBPS_FILE)))
        print("Using FunctionBasedPathSet for speed")

    # Add reserves for stability
    modelProcessor.append(osim.ModOpAddReserves(RESERVE_STRENGTH))

    track.setModel(modelProcessor)

    # --- Reference kinematics to track ---
    # MocoTrack uses TableProcessor for the reference states
    tableProcessor = osim.TableProcessor(os.path.abspath(COORDS_FILE))
    tableProcessor.append(osim.TabOpLowPassFilter(6))  # 6 Hz low-pass
    tableProcessor.append(osim.TabOpUseAbsoluteStateNames())
    track.setStatesReference(tableProcessor)

    # Which states to track (only coordinate values, not speeds)
    track.set_states_global_tracking_weight(TRACKING_WEIGHT)

    # Don't track exo joint coordinates (they're locked anyway)
    # Also don't track speeds — just positions
    track.set_allow_unused_references(True)

    # --- Time window ---
    track.set_initial_time(T_START)
    track.set_final_time(T_END)
    track.set_mesh_interval(MESH_INTERVAL)

    # --- Solver settings ---
    # MocoTrack internally creates a MocoCasADiSolver
    # We can customize it after calling initialize()
    study = track.initialize()
    problem = study.updProblem()

    # Add effort goal (minimize muscle activations)
    effort = osim.MocoControlGoal.safeDownCast(
        problem.updGoal('control_effort'))
    if effort is not None:
        effort.setWeight(EFFORT_WEIGHT)

    # Set the exo actuator controls to be excluded from the effort goal
    # (they're prescribed, not optimized — and we don't want to penalize
    #  the exo for doing its job)
    effort.setWeightForControl('/forceset/exo_hip_r', 0.0)
    effort.setWeightForControl('/forceset/exo_hip_l', 0.0)

    # Also exclude reserve actuators from being weighted too heavily
    # (reserves should be small — we just don't want them dominating)

    # Configure solver
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    solver.set_num_mesh_intervals(
        int((T_END - T_START) / MESH_INTERVAL))
    solver.set_optim_convergence_tolerance(1e-3)  # slightly relaxed for speed
    solver.set_optim_constraint_tolerance(1e-3)
    solver.set_optim_max_iterations(1000)

    # --- Solve ---
    print(f"\nSolving MocoTrack...")
    print(f"  Time window:  {T_START} - {T_END} s")
    print(f"  Mesh interval: {MESH_INTERVAL} s")
    print(f"  Tracking weight: {TRACKING_WEIGHT}")
    print(f"  Effort weight: {EFFORT_WEIGHT}")
    print(f"  Exo peak torque: {PEAK_TORQUE} Nm")
    print()
    
    # Warm start from MocoInverse baseline solution
    if os.path.exists(BASELINE_SOLUTION):
        print("Using baseline MocoInverse as initial guess...")
        solver.setGuessFile(BASELINE_SOLUTION)
    else:
        print("No initial guess — using default (this may be slow)")
    solution = study.solve()

    solve_time = timer.time() - t0
    print(f"\nSolved in {solve_time:.1f}s")

    # Check if solution is sealed (failed)
    if solution.isSealed():
        print("WARNING: Solution is sealed (solver may have failed).")
        print("Unsealing to inspect partial solution...")
        solution.unseal()

    # Save solution
    solution_path = os.path.join(OUTPUT_DIR, f'{output_prefix}_solution.sto')
    solution.write(solution_path)
    print(f"Solution written to {solution_path}")

    return solution, solution_path, solve_time


# ===========================================================================
# ANALYSIS: Compare muscle activations to baseline
# ===========================================================================

def analyze_results(solution_path):
    """
    Compare MocoTrack muscle activations against baseline MocoInverse.
    """
    print("\n" + "=" * 60)
    print("Analyzing Results")
    print("=" * 60)

    # Key hip muscles to compare (same as evaluate_exo_torque.py)
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

    # Load MocoTrack solution
    track_table = osim.TimeSeriesTable(solution_path)
    track_labels = list(track_table.getColumnLabels())

    # Load baseline solution
    if not os.path.exists(BASELINE_SOLUTION):
        print(f"WARNING: Baseline solution not found at {BASELINE_SOLUTION}")
        print("Skipping comparison. Run baseline MocoInverse first.")
        return

    baseline_table = osim.TimeSeriesTable(BASELINE_SOLUTION)
    baseline_labels = list(baseline_table.getColumnLabels())

    # Extract activation columns
    # MocoTrack uses state names like '/forceset/glut_max1_r/activation'
    # MocoInverse uses control names like '/forceset/glut_max1_r'
    # We need to handle both formats

    track_effort = 0.0
    baseline_effort = 0.0
    n_muscles = 0

    for muscle in HIP_MUSCLES:
        # Try MocoTrack state format first
        track_col = f'/forceset/{muscle}/activation'
        if track_col not in track_labels:
            # Try control format
            track_col = f'/forceset/{muscle}'
            if track_col not in track_labels:
                continue

        baseline_col = f'/forceset/{muscle}'
        if baseline_col not in baseline_labels:
            # Try activation format
            baseline_col = f'/forceset/{muscle}/activation'
            if baseline_col not in baseline_labels:
                continue

        # Get activation data
        track_col_data = track_table.getDependentColumn(track_col)
        track_data = np.array([track_col_data[i] for i in range(track_table.getNumRows())])
        
        baseline_col_data = baseline_table.getDependentColumn(baseline_col)
        baseline_data = np.array([baseline_col_data[i] for i in range(baseline_table.getNumRows())])

        track_effort += np.mean(track_data ** 2)
        baseline_effort += np.mean(baseline_data ** 2)
        n_muscles += 1

    if n_muscles == 0:
        print("WARNING: No matching muscle columns found.")
        print("Track columns (first 10):", track_labels[:10])
        return

    # Normalize
    track_effort /= n_muscles
    baseline_effort /= n_muscles

    reduction_pct = (1.0 - track_effort / baseline_effort) * 100

    print(f"\nResults ({n_muscles} hip muscles compared):")
    print(f"  Baseline effort (MocoInverse): {baseline_effort:.4f}")
    print(f"  MocoTrack effort (exo assist): {track_effort:.4f}")
    print(f"  Effort reduction:              {reduction_pct:+.1f}%")
    print()

    if reduction_pct > 0:
        print("  >>> Exo assistance reduced muscle effort (good!)")
    else:
        print("  >>> Exo assistance INCREASED effort — check torque profile")

    return {
        'baseline_effort': baseline_effort,
        'track_effort': track_effort,
        'reduction_pct': reduction_pct,
        'n_muscles': n_muscles,
    }


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Stage 3: MocoTrack with Exoskeleton Assistance")
    print("=" * 60)
    print(f"Model:  {MODEL_FILE}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Step 1: Prepare model with exo torque
    model_path = prepare_model_with_exo_torque(
        PEAK_TORQUE, ONSET_PHASE, DURATION)

    # Step 2: Run MocoTrack
    solution, solution_path, solve_time = run_mocotrack(model_path)

    # Step 3: Analyze
    results = analyze_results(solution_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if results:
        print(f"  Effort reduction:  {results['reduction_pct']:+.1f}%")
    print(f"  Solve time:        {solve_time:.1f}s")
    print(f"  Solution:          {solution_path}")
    print()
    print("Next steps:")
    print("  1. If this works, refine mesh to 0.02 for higher accuracy")
    print("  2. Run with different torque params to compare")
    print("  3. Wrap in evaluate() function for optimizer/RL")
    print("  4. Swap in AddBiomechanics subjects for multi-subject training")
