#!/usr/bin/env python3
"""
evaluate_exo_torque.py — RL Evaluation Function for Hip Exoskeleton

PURPOSE:
    This is the "forward pass" of our RL pipeline. Given a set of torque
    parameters, it runs OpenSim MocoInverse and returns a scalar reward
    that measures how much the exoskeleton reduced muscle effort.

    Think of it like a loss function:
        - In classification: model(x) → prediction → loss(prediction, label)
        - Here:              OpenSim(torque_params) → muscle_activations → reward

    An optimizer (CMA-ES, grid search, or eventually PPO) calls this function
    repeatedly with different torque parameters, trying to maximize the reward.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────┐
    │  evaluate(peak, onset, duration, side='bilateral')  │
    │                                                     │
    │  1. Build torque time series from params            │
    │  2. Build OpenSim model (human + exo mass)          │
    │  3. Run MocoInverse with prescribed torque          │
    │  4. Extract muscle activations from solution        │
    │  5. Compare to baseline (Condition 1)               │
    │  6. Return scalar reward                            │
    └─────────────────────────────────────────────────────┘

USAGE:
    # Single evaluation
    python evaluate_exo_torque.py

    # As a module (for optimizer integration)
    from evaluate_exo_torque import evaluate
    reward, info = evaluate(peak_torque=8.0, onset_phase=0.1, duration=0.4)

FILE PATHS:
    Assumes you're running from:
        ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
    
    Required files (already exist from your previous work):
        ../subject_walk_scaled.osim          — human musculoskeletal model
        ../grf_walk.xml                      — ground reaction forces
        ../coordinates.sto                   — walking kinematics (IK solution)
        ../moco-tutorial/example3DWalking/subject_walk_scaled_FunctionBasedPathSet.xml
        baseline/baseline_MocoInverse_solution.sto  — Condition 1 solution

Run from: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
Prereqs:  conda activate opensim
"""

import os
import sys
import time as timer
import numpy as np
import opensim as osim


# ===========================================================================
# CONFIGURATION — adjust these paths to match your cortex directory layout
# ===========================================================================

# Base directory (parent of exo_active/, baseline/, etc.)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input files
MODEL_FILE = os.path.join(BASE_DIR, 'subject_walk_scaled.osim')
GRF_FILE = os.path.join(BASE_DIR, 'grf_walk.xml')
COORDS_FILE = os.path.join(BASE_DIR, 'coordinates.sto')
FBPS_FILE = os.path.join(BASE_DIR, '..', 'moco-tutorial', 'example3DWalking',
                         'subject_walk_scaled_FunctionBasedPathSet.xml')

# Baseline solution (Condition 1 — no exo). Used to compute relative improvement.
BASELINE_SOLUTION = os.path.join(BASE_DIR, 'baseline',
                                 'baseline_MocoInverse_solution.sto')

# Exo mass distribution (kg) — from your WAWA URDF (total 2.9 kg)
EXO_MASS = {'pelvis': 2.32, 'femur_r': 0.29, 'femur_l': 0.29}

# Time window for the gait cycle (from your previous Moco runs)
T_START = 0.48
T_END = 1.61

# Hip muscles to track for reward computation
# These are the muscles most affected by hip exoskeleton assistance
HIP_MUSCLES = [
    'glut_max1_r', 'glut_max2_r', 'glut_max3_r',   # gluteus maximus (extension)
    'glut_max1_l', 'glut_max2_l', 'glut_max3_l',
    'psoas_r', 'iliacus_r',                          # iliopsoas (flexion)
    'psoas_l', 'iliacus_l',
    'rect_fem_r', 'rect_fem_l',                      # rectus femoris (flexion)
    'semimem_r', 'semiten_r', 'bifemlh_r',           # hamstrings (extension)
    'semimem_l', 'semiten_l', 'bifemlh_l',
]

# Reward weights
W_MUSCLE = 1.0      # weight for muscle effort reduction (main objective)
W_RESERVE = 0.3     # weight for reserve actuator penalty
W_SMOOTH = 0.1      # weight for torque smoothness penalty


# ===========================================================================
# STEP 1: Build torque profile from parameters
# ===========================================================================

def build_torque_profile(time_array, peak_torque, onset_phase, duration,
                         side='bilateral'):
    """
    Build a torque time series from 3 parameters.

    This is the "action" in RL terms. Instead of the full neural network
    outputting torque at every timestep (that comes later in Stage 3+),
    we parameterize the torque as a simple bell-shaped pulse.

    Parameters
    ----------
    time_array : np.array
        Time points from the kinematics file (e.g., 0.48 to 1.61 s)
    peak_torque : float
        Maximum assistance torque in Nm (e.g., 8.0)
    onset_phase : float
        When in the gait cycle the assistance starts (0.0 = start, 1.0 = end)
    duration : float
        How long the assistance lasts as fraction of gait cycle (e.g., 0.4 = 40%)
    side : str
        'bilateral' (same on both sides), 'right', or 'left'

    Returns
    -------
    torque_r, torque_l : np.array
        Right and left hip assistance torques at each time point

    Why a bell curve?
    -----------------
    The Nature paper found that smooth torque profiles are critical for
    comfort and effectiveness (they penalize jerk in their reward function,
    Eq. 14). A bell curve is smooth by construction — no sharp edges.
    Later (Stage 3), the RL policy will learn arbitrary smooth profiles,
    but for Stage 2 optimization, 3 parameters is enough to find the
    ballpark of good assistance timing and magnitude.
    """
    # Normalize time to gait phase (0 to 1)
    t_start = time_array[0]
    t_end = time_array[-1]
    gait_cycle = t_end - t_start
    phase = (time_array - t_start) / gait_cycle  # 0 → 1

    # Build a smooth bell curve (raised cosine)
    # This is zero outside [onset, onset+duration] and peaks at the midpoint
    center = onset_phase + duration / 2.0
    torque = np.zeros_like(time_array)

    for i, phi in enumerate(phase):
        # Distance from center, normalized by half-duration
        dist = abs(phi - center) / (duration / 2.0) if duration > 0 else 999
        if dist <= 1.0:
            # Raised cosine: smooth bump from 0 to peak and back
            torque[i] = peak_torque * 0.5 * (1 + np.cos(np.pi * dist))

    # Assign to sides
    if side == 'bilateral':
        torque_r = torque.copy()
        # Shift left leg by ~50% of gait cycle (contralateral timing)
        # In walking, left and right hips are ~180° out of phase
        phase_shifted = (phase + 0.5) % 1.0
        torque_l = np.zeros_like(time_array)
        for i, phi in enumerate(phase_shifted):
            dist = abs(phi - center) / (duration / 2.0) if duration > 0 else 999
            if dist <= 1.0:
                torque_l[i] = peak_torque * 0.5 * (1 + np.cos(np.pi * dist))
    elif side == 'right':
        torque_r = torque.copy()
        torque_l = np.zeros_like(time_array)
    elif side == 'left':
        torque_r = np.zeros_like(time_array)
        torque_l = torque.copy()
    else:
        raise ValueError(f"side must be 'bilateral', 'right', or 'left', got {side}")

    return torque_r, torque_l


# ===========================================================================
# STEP 2: Read .sto files (reused from your existing code)
# ===========================================================================

def read_sto(filepath):
    """Read an OpenSim .sto file into a dict of numpy arrays."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.strip() == 'endheader':
                header_lines = i + 1
                break
    data = np.genfromtxt(filepath, skip_header=header_lines,
                         names=True, delimiter='\t', deletechars='')
    return data


def write_assistance_sto(time, torque_r, torque_l,
                         filename='_temp_assistance_torques.sto'):
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
    return filename


# ===========================================================================
# STEP 3: Build model and run MocoInverse (adapted from your exo_active.py)
# ===========================================================================

def run_moco_inverse(torque_r, torque_l, time_array, peak_torque,
                     output_file='_temp_eval_solution.sto', verbose=True):
    """
    Run MocoInverse with prescribed exo assistance torques.

    This is your existing Condition 3 code, restructured as a function.
    The key difference: it RETURNS the solution path instead of just saving it.

    Parameters
    ----------
    torque_r, torque_l : np.array
        Hip assistance torque profiles (Nm) at each time point
    time_array : np.array
        Time points matching the torque arrays
    peak_torque : float
        Used to set optimal force for the CoordinateActuator
    output_file : str
        Where to save the MocoInverse solution
    verbose : bool
        Print progress messages

    Returns
    -------
    solution_path : str or None
        Path to the solution .sto file, or None if solve failed
    """
    model_file = os.path.abspath(MODEL_FILE)
    grf_file = os.path.abspath(GRF_FILE)
    coords_file = os.path.abspath(COORDS_FILE)
    fbps_file = os.path.abspath(FBPS_FILE)

    # --- Write torque profile to .sto ---
    assist_sto = write_assistance_sto(time_array, torque_r, torque_l)

    # --- Build model with exo mass + actuators ---
    model = osim.Model(model_file)

    # Add exo mass to body segments
    for body_name, mass in EXO_MASS.items():
        body = model.getBodySet().get(body_name)
        body.setMass(body.getMass() + mass)

    if verbose:
        total_exo = sum(EXO_MASS.values())
        print(f"Added exo mass: {total_exo:.2f} kg total")

    # Add CoordinateActuators for hip assistance
    # optimal_force sets the scale — control values are normalized to [-1, 1]
    optimal_force = max(peak_torque, 1.0)  # avoid division by zero
    for side, coord_name in [('r', 'hip_flexion_r'), ('l', 'hip_flexion_l')]:
        actuator = osim.CoordinateActuator()
        actuator.setName(f'exo_hip_assist_{side}')
        actuator.set_coordinate(coord_name)
        actuator.setOptimalForce(optimal_force)
        actuator.setMinControl(-1.0)
        actuator.setMaxControl(1.0)
        model.addForce(actuator)

    # Save temporary model
    temp_model = os.path.abspath('_temp_model_with_exo.osim')
    model.finalizeConnections()
    model.printToXML(temp_model)

    # --- Set up MocoInverse ---
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
    modelProcessor.append(osim.ModOpAddReserves(1.0))
    inverse.setModel(modelProcessor)

    inverse.setKinematics(osim.TableProcessor(coords_file))
    inverse.set_initial_time(T_START)
    inverse.set_final_time(T_END)
    inverse.set_mesh_interval(0.02)
    inverse.set_kinematics_allow_extra_columns(True)

    # --- Initialize and add prescribed torque constraints ---
    study = inverse.initialize()
    problem = study.updProblem()

    # Load the assistance torque table we just wrote
    assist_table = osim.TimeSeriesTable(assist_sto)
    time_vec = assist_table.getIndependentColumn()

    for side_label, col_name in [('r', 'exo_hip_assist_r'),
                                  ('l', 'exo_hip_assist_l')]:
        torque_col = assist_table.getDependentColumn(col_name)

        # Build piecewise linear function (normalized by optimal force)
        spline = osim.PiecewiseLinearFunction()
        for i in range(len(time_vec)):
            normalized = float(torque_col[i]) / optimal_force
            spline.addPoint(float(time_vec[i]), normalized)

        # Prescribe: set both upper and lower bound to same spline
        # This forces the optimizer to use exactly this torque profile
        constraint = osim.MocoControlBoundConstraint()
        constraint.setName(f'prescribe_exo_{side_label}')
        constraint.addControlPath(f'/forceset/exo_hip_assist_{side_label}')
        constraint.setUpperBound(spline)
        constraint.setLowerBound(spline)
        problem.addPathConstraint(constraint)

    # --- Solve ---
    if verbose:
        print(f"Solving MocoInverse (peak={peak_torque:.1f} Nm)...")
    
    t0 = timer.time()
    try:
        solution = study.solve()
        elapsed = timer.time() - t0
        if verbose:
            print(f"Solved in {elapsed:.1f}s")

        output_path = os.path.abspath(output_file)
        solution.write(output_path)
        return output_path

    except Exception as e:
        elapsed = timer.time() - t0
        print(f"MocoInverse FAILED after {elapsed:.1f}s: {e}")
        return None

    finally:
        # Clean up temp files
        for f in [assist_sto, temp_model]:
            if os.path.exists(f):
                os.remove(f)


# ===========================================================================
# STEP 4: Extract muscle activations from solution
# ===========================================================================

def extract_muscle_effort(solution_path, muscle_names=None):
    """
    Extract muscle activation data from a MocoInverse solution.

    In ML terms, this is like extracting the "prediction" from the model
    output so you can compute the loss.

    Parameters
    ----------
    solution_path : str
        Path to the MocoInverse solution .sto file
    muscle_names : list of str, optional
        Which muscles to extract. Defaults to HIP_MUSCLES.

    Returns
    -------
    effort : float
        Sum of squared muscle activations (integrated over the gait cycle).
        Lower = better (less muscle work needed).
    reserve_penalty : float
        Sum of squared reserve actuator values. Should be small if the
        solution is physically realistic.
    muscle_data : dict
        Raw activation arrays for each muscle (for plotting later).
    """
    if muscle_names is None:
        muscle_names = HIP_MUSCLES

    data = read_sto(solution_path)
    col_names = data.dtype.names
    time_col = data[col_names[0]]  # first column is always time

    # --- Muscle activations ---
    # In the .sto solution, muscle activations are columns like:
    #   /forceset/glut_max1_r/activation  (varies by OpenSim version)
    # We search for partial matches
    muscle_data = {}
    total_effort = 0.0

    for muscle in muscle_names:
        # Find the column that contains this muscle name + 'activation'
        matched_col = None
        for cn in col_names:
            if muscle in cn and 'activation' in cn:
                matched_col = cn
                break

        if matched_col is not None:
            activation = data[matched_col]
            muscle_data[muscle] = activation
            # Effort = integral of activation^2 over time (trapezoidal rule)
            effort_i = np.trapz(activation ** 2, time_col)
            total_effort += effort_i
        # If muscle not found, skip silently (some models may differ)

    # --- Reserve actuator usage ---
    # Reserves are columns containing 'reserve' in the name
    reserve_penalty = 0.0
    for cn in col_names:
        if 'reserve' in cn.lower():
            reserve_vals = data[cn]
            reserve_penalty += np.trapz(reserve_vals ** 2, time_col)

    return total_effort, reserve_penalty, muscle_data


# ===========================================================================
# STEP 5: Load baseline and compute reward
# ===========================================================================

def load_baseline_effort():
    """
    Load muscle effort from the Condition 1 (no exo) baseline solution.
    
    This only needs to run once — we cache the result.
    """
    if not os.path.exists(BASELINE_SOLUTION):
        raise FileNotFoundError(
            f"Baseline solution not found at {BASELINE_SOLUTION}\n"
            f"Run your Condition 1 (baseline) MocoInverse first, then update "
            f"the BASELINE_SOLUTION path."
        )
    effort, reserve, _ = extract_muscle_effort(BASELINE_SOLUTION)
    return effort, reserve


# Cache baseline so we don't re-read the file every evaluation
_baseline_cache = None


def get_baseline():
    """Get cached baseline effort values."""
    global _baseline_cache
    if _baseline_cache is None:
        _baseline_cache = load_baseline_effort()
    return _baseline_cache


# ===========================================================================
# STEP 6: The main evaluation function — this is what the optimizer calls
# ===========================================================================

def evaluate(peak_torque, onset_phase, duration, side='bilateral',
             verbose=True):
    """
    THE CORE FUNCTION.

    Given torque parameters, run OpenSim and return a reward.

    This is analogous to:
        - In supervised learning:  loss = criterion(model(x), y)
        - Here:                    reward = -cost(opensim(torque_params), baseline)

    Parameters
    ----------
    peak_torque : float
        Peak assistance torque in Nm (typical range: 2–15 Nm)
    onset_phase : float
        When assistance starts in gait cycle (0.0–1.0)
    duration : float
        Duration of assistance as fraction of gait cycle (0.1–0.6)
    side : str
        'bilateral', 'right', or 'left'
    verbose : bool
        Print progress

    Returns
    -------
    reward : float
        Scalar reward. Higher = better.
        Positive means the exo reduced muscle effort vs baseline.
        Negative means the exo made things worse.
    info : dict
        Detailed breakdown for debugging:
        - 'muscle_effort': total squared muscle activation
        - 'baseline_effort': baseline (no exo) effort
        - 'effort_reduction_pct': percentage reduction
        - 'reserve_penalty': reserve actuator usage
        - 'smoothness_penalty': torque profile jerkiness
        - 'solve_time': how long MocoInverse took
        - 'success': whether the solve completed
    """
    info = {
        'peak_torque': peak_torque,
        'onset_phase': onset_phase,
        'duration': duration,
        'side': side,
        'success': False,
    }

    # --- Sanity checks ---
    if peak_torque < 0:
        print("Warning: negative peak_torque, clipping to 0")
        peak_torque = 0.0
    if not (0 <= onset_phase <= 1):
        print(f"Warning: onset_phase={onset_phase} out of [0,1], clipping")
        onset_phase = np.clip(onset_phase, 0, 1)
    if not (0.05 <= duration <= 0.8):
        print(f"Warning: duration={duration} out of [0.05, 0.8], clipping")
        duration = np.clip(duration, 0.05, 0.8)

    # --- Step 1: Build torque profile ---
    coords_data = read_sto(COORDS_FILE)
    time_array = coords_data[coords_data.dtype.names[0]]

    torque_r, torque_l = build_torque_profile(
        time_array, peak_torque, onset_phase, duration, side)

    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATE: peak={peak_torque:.1f} Nm, "
              f"onset={onset_phase:.2f}, dur={duration:.2f}")
        print(f"  Right torque: max={np.max(torque_r):.1f} Nm")
        print(f"  Left torque:  max={np.max(torque_l):.1f} Nm")
        print(f"{'='*60}")

    # --- Steps 2-3: Run MocoInverse ---
    t0 = timer.time()
    solution_path = run_moco_inverse(
        torque_r, torque_l, time_array, peak_torque, verbose=verbose)
    solve_time = timer.time() - t0
    info['solve_time'] = solve_time

    if solution_path is None:
        # Solver failed — return large negative reward
        info['reward'] = -100.0
        print(f"RESULT: solve failed → reward = -100")
        return -100.0, info

    # --- Step 4: Extract muscle effort ---
    muscle_effort, reserve_penalty, muscle_data = extract_muscle_effort(
        solution_path)
    info['muscle_effort'] = muscle_effort
    info['reserve_penalty'] = reserve_penalty
    info['muscle_data'] = muscle_data

    # --- Step 5: Compare to baseline ---
    baseline_effort, baseline_reserve = get_baseline()
    info['baseline_effort'] = baseline_effort
    info['baseline_reserve'] = baseline_reserve

    # Effort reduction as percentage
    if baseline_effort > 0:
        reduction_pct = (baseline_effort - muscle_effort) / baseline_effort * 100
    else:
        reduction_pct = 0.0
    info['effort_reduction_pct'] = reduction_pct

    # --- Smoothness penalty ---
    # Penalize rapid changes in torque (jerk)
    # This matches the paper's r_as reward term (Eq. 14)
    dt = np.diff(time_array)
    jerk_r = np.diff(torque_r, n=2)  # second derivative
    jerk_l = np.diff(torque_l, n=2)
    smoothness_penalty = np.mean(jerk_r**2) + np.mean(jerk_l**2)
    info['smoothness_penalty'] = smoothness_penalty

    # --- Step 6: Compute reward ---
    # Reward = effort_reduction - penalties
    # Positive reward = exo is helping
    # Negative reward = exo is making things worse
    reward = (
        W_MUSCLE * (baseline_effort - muscle_effort)  # positive if effort reduced
        - W_RESERVE * reserve_penalty                   # penalize unrealistic reserves
        - W_SMOOTH * smoothness_penalty                 # penalize jerky torques
    )
    info['reward'] = reward
    info['success'] = True

    if verbose:
        print(f"\nRESULT:")
        print(f"  Muscle effort:     {muscle_effort:.4f} "
              f"(baseline: {baseline_effort:.4f})")
        print(f"  Effort reduction:  {reduction_pct:+.1f}%")
        print(f"  Reserve penalty:   {reserve_penalty:.4f}")
        print(f"  Smoothness pen:    {smoothness_penalty:.4f}")
        print(f"  >>> REWARD:        {reward:+.4f}")
        print(f"  Solve time:        {solve_time:.1f}s")

    # Clean up solution file
    if os.path.exists(solution_path):
        os.remove(solution_path)

    return reward, info


# ===========================================================================
# DEMO: Run a few evaluations to test the pipeline
# ===========================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Exoskeleton Torque Evaluation Function — Test Run")
    print("=" * 60)

    # First, verify baseline exists
    try:
        bl_effort, bl_reserve = get_baseline()
        print(f"\nBaseline loaded:")
        print(f"  Muscle effort:  {bl_effort:.4f}")
        print(f"  Reserve usage:  {bl_reserve:.4f}")
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nYou need to update the BASELINE_SOLUTION path first.")
        print("It should point to your Condition 1 (no exo) solution .sto file.")
        sys.exit(1)

    # --- Test 1: Run with parameters similar to the Nature paper ---
    print("\n\n" + "=" * 60)
    print("TEST 1: Nature-paper-like parameters")
    print("  peak=8 Nm, onset=0.10, duration=0.40")
    print("=" * 60)
    reward1, info1 = evaluate(
        peak_torque=8.0, onset_phase=0.10, duration=0.40)

    # --- Test 2: Try a different timing ---
    print("\n\n" + "=" * 60)
    print("TEST 2: Later onset, shorter burst")
    print("  peak=8 Nm, onset=0.30, duration=0.25")
    print("=" * 60)
    reward2, info2 = evaluate(
        peak_torque=8.0, onset_phase=0.30, duration=0.25)

    # --- Test 3: Intentionally bad parameters (should give negative reward) ---
    print("\n\n" + "=" * 60)
    print("TEST 3: Very low torque (should barely help)")
    print("  peak=1 Nm, onset=0.10, duration=0.40")
    print("=" * 60)
    reward3, info3 = evaluate(
        peak_torque=1.0, onset_phase=0.10, duration=0.40)

    # --- Summary ---
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (8Nm, onset=0.10, dur=0.40): reward={reward1:+.4f}, "
          f"reduction={info1.get('effort_reduction_pct', 0):+.1f}%")
    print(f"  Test 2 (8Nm, onset=0.30, dur=0.25): reward={reward2:+.4f}, "
          f"reduction={info2.get('effort_reduction_pct', 0):+.1f}%")
    print(f"  Test 3 (1Nm, onset=0.10, dur=0.40): reward={reward3:+.4f}, "
          f"reduction={info3.get('effort_reduction_pct', 0):+.1f}%")
    print()
    print("If Test 1 > Test 3, the reward function is working correctly —")
    print("more assistance at the right time produces higher reward.")
    print()
    print("Next step: plug this into CMA-ES to find optimal parameters.")
    print("See: optimize_torque_params.py (Stage 2)")
