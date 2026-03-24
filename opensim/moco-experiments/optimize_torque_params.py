#!/usr/bin/env python3
"""
optimize_torque_params.py — Stage 2: Low-Dimensional Policy Search

PURPOSE:
    Find the best torque profile parameters (peak, onset, duration) by
    calling evaluate() many times with different parameter combinations.

    This is NOT neural-network RL yet. Think of it as:
        - Grid search, but smarter (CMA-ES explores efficiently)
        - The "sanity check" before committing to full RL
        - Equivalent to hyperparameter tuning in ML

    If this works (i.e., the optimizer finds parameters that reduce muscle
    effort), it validates that:
        1. The evaluation function returns meaningful rewards
        2. The OpenSim model responds correctly to exo torques
        3. The reward function captures what we want

    If this DOESN'T work, we debug here before adding neural networks.

HOW IT WORKS:
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a
    derivative-free optimizer. It:
        1. Starts with an initial guess (e.g., peak=8, onset=0.1, dur=0.4)
        2. Generates a "population" of ~10 parameter vectors (slight variations)
        3. Evaluates each one by calling evaluate()
        4. Keeps the best ones, adjusts the search distribution
        5. Repeats for N generations

    It's like natural selection applied to torque parameters.

INSTALL:
    pip install cma --break-system-packages
    (on cortex, use: pip install cma -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages)

USAGE:
    python optimize_torque_params.py

    Expected runtime: ~30 min to a few hours depending on population size
    and number of generations (each evaluate() call takes ~30-120s for MocoInverse)

Run from: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
Prereqs:  conda activate opensim
"""

import os
import json
import time
import numpy as np

# We import our evaluation function
from evaluate_exo_torque import evaluate, get_baseline


# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Search bounds for the 3 parameters
PARAM_BOUNDS = {
    'peak_torque': (2.0, 12.0),    # Nm — WAWA can do up to 18 Nm
    'onset_phase': (0.0, 0.6),     # early-to-mid gait cycle
    'duration':    (0.15, 0.60),   # 15% to 60% of gait cycle
}

# Initial guess (based on Nature paper values)
X0 = [7.0, 0.10, 0.40]   # [peak, onset, duration]

# CMA-ES settings
SIGMA0 = 2.0       # initial step size (how far to explore from X0)
POPSIZE = 6        # population size per generation (keep small — each is a Moco solve)
MAX_GENERATIONS = 20  # total generations (total evals ≈ POPSIZE * MAX_GENERATIONS)

# Output
RESULTS_FILE = 'optimization_results.json'
LOG_FILE = 'optimization_log.csv'


# ===========================================================================
# OPTION A: CMA-ES (recommended — smarter search)
# ===========================================================================

def optimize_cmaes():
    """
    Run CMA-ES optimization over torque parameters.

    CMA-ES is the go-to optimizer for low-dimensional continuous problems
    where you can't compute gradients (which is our case — OpenSim is a
    black box, we can't backprop through it).
    """
    try:
        import cma
    except ImportError:
        print("CMA-ES not installed. Install with:")
        print("  pip install cma -i https://pypi.tuna.tsinghua.edu.cn/simple "
              "--break-system-packages")
        print("\nFalling back to grid search...")
        return optimize_grid()

    print("=" * 60)
    print("CMA-ES Optimization of Exoskeleton Torque Parameters")
    print("=" * 60)
    print(f"Initial guess:  peak={X0[0]}, onset={X0[1]}, dur={X0[2]}")
    print(f"Population:     {POPSIZE} per generation")
    print(f"Max generations: {MAX_GENERATIONS}")
    print(f"Estimated total evaluations: ~{POPSIZE * MAX_GENERATIONS}")
    print()

    # Verify baseline first
    bl_effort, bl_reserve = get_baseline()
    print(f"Baseline effort: {bl_effort:.4f}")
    print()

    # Set up logging
    log_entries = []
    best_reward = -float('inf')
    best_params = None

    def objective(x):
        """
        CMA-ES minimizes, but we want to maximize reward.
        So we return negative reward.
        """
        peak, onset, dur = x

        # Clip to bounds
        peak = np.clip(peak, *PARAM_BOUNDS['peak_torque'])
        onset = np.clip(onset, *PARAM_BOUNDS['onset_phase'])
        dur = np.clip(dur, *PARAM_BOUNDS['duration'])

        reward, info = evaluate(peak, onset, dur, verbose=False)

        # Log this evaluation
        entry = {
            'peak_torque': float(peak),
            'onset_phase': float(onset),
            'duration': float(dur),
            'reward': float(reward),
            'effort_reduction_pct': info.get('effort_reduction_pct', 0),
            'solve_time': info.get('solve_time', 0),
            'success': info.get('success', False),
        }
        log_entries.append(entry)

        nonlocal best_reward, best_params
        if reward > best_reward:
            best_reward = reward
            best_params = [float(peak), float(onset), float(dur)]
            print(f"  ★ NEW BEST: reward={reward:+.4f}, "
                  f"peak={peak:.1f}, onset={onset:.2f}, dur={dur:.2f}, "
                  f"reduction={info.get('effort_reduction_pct', 0):+.1f}%")

        # CMA-ES minimizes, so negate
        return -reward

    # Run CMA-ES
    bounds_lower = [PARAM_BOUNDS['peak_torque'][0],
                    PARAM_BOUNDS['onset_phase'][0],
                    PARAM_BOUNDS['duration'][0]]
    bounds_upper = [PARAM_BOUNDS['peak_torque'][1],
                    PARAM_BOUNDS['onset_phase'][1],
                    PARAM_BOUNDS['duration'][1]]

    opts = {
        'popsize': POPSIZE,
        'maxiter': MAX_GENERATIONS,
        'bounds': [bounds_lower, bounds_upper],
        'verb_disp': 1,      # print every generation
        'seed': 42,
    }

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(X0, SIGMA0, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)
        es.disp()

    total_time = time.time() - t0

    # --- Results ---
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best parameters found:")
    print(f"  peak_torque:  {best_params[0]:.2f} Nm")
    print(f"  onset_phase:  {best_params[1]:.3f}")
    print(f"  duration:     {best_params[2]:.3f}")
    print(f"  reward:       {best_reward:+.4f}")
    print(f"Total evaluations: {len(log_entries)}")
    print(f"Total time:        {total_time/60:.1f} min")
    print(f"Avg time/eval:     {total_time/max(len(log_entries),1):.1f}s")

    # Save results
    results = {
        'best_params': {
            'peak_torque': best_params[0],
            'onset_phase': best_params[1],
            'duration': best_params[2],
        },
        'best_reward': best_reward,
        'total_evaluations': len(log_entries),
        'total_time_min': total_time / 60,
        'log': log_entries,
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Save CSV log for easy plotting
    with open(LOG_FILE, 'w') as f:
        f.write("eval,peak_torque,onset_phase,duration,reward,"
                "effort_reduction_pct,solve_time\n")
        for i, entry in enumerate(log_entries):
            f.write(f"{i},{entry['peak_torque']:.3f},"
                    f"{entry['onset_phase']:.3f},{entry['duration']:.3f},"
                    f"{entry['reward']:.4f},"
                    f"{entry['effort_reduction_pct']:.2f},"
                    f"{entry['solve_time']:.1f}\n")
    print(f"Log saved to {LOG_FILE}")

    return best_params, best_reward


# ===========================================================================
# OPTION B: Grid search (fallback if CMA-ES not installed)
# ===========================================================================

def optimize_grid():
    """
    Simple grid search over torque parameters.

    Less efficient than CMA-ES, but requires no extra packages.
    Good for a first test with just a few parameter combinations.
    """
    print("=" * 60)
    print("Grid Search over Torque Parameters")
    print("=" * 60)

    # Define grid (keep it small — each evaluation takes ~1 min)
    peaks = [4.0, 8.0, 12.0]
    onsets = [0.05, 0.15, 0.30]
    durations = [0.25, 0.40]

    total = len(peaks) * len(onsets) * len(durations)
    print(f"Grid: {len(peaks)} peaks × {len(onsets)} onsets × "
          f"{len(durations)} durations = {total} evaluations")
    print(f"Estimated time: ~{total * 1.5:.0f} min\n")

    results = []
    best_reward = -float('inf')
    best_params = None

    for i, peak in enumerate(peaks):
        for j, onset in enumerate(onsets):
            for k, dur in enumerate(durations):
                idx = i * len(onsets) * len(durations) + j * len(durations) + k
                print(f"\n[{idx+1}/{total}] peak={peak}, onset={onset}, "
                      f"dur={dur}")

                reward, info = evaluate(peak, onset, dur, verbose=False)

                results.append({
                    'peak_torque': peak,
                    'onset_phase': onset,
                    'duration': dur,
                    'reward': reward,
                    'effort_reduction_pct': info.get('effort_reduction_pct', 0),
                })

                if reward > best_reward:
                    best_reward = reward
                    best_params = [peak, onset, dur]
                    print(f"  ★ NEW BEST: reward={reward:+.4f}")
                else:
                    print(f"  reward={reward:+.4f}")

    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best: peak={best_params[0]}, onset={best_params[1]}, "
          f"dur={best_params[2]}")
    print(f"Reward: {best_reward:+.4f}")

    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump({'best_params': dict(zip(
            ['peak_torque', 'onset_phase', 'duration'], best_params)),
            'best_reward': best_reward,
            'grid_results': results}, f, indent=2)

    return best_params, best_reward


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == '__main__':
    print("\nWhich optimizer to use?")
    print("  1. CMA-ES (recommended, needs: pip install cma)")
    print("  2. Grid search (no extra packages)")
    print()

    # Try CMA-ES first, fall back to grid
    try:
        import cma
        best_params, best_reward = optimize_cmaes()
    except ImportError:
        best_params, best_reward = optimize_grid()

    print("\n\nDone! Next steps:")
    print("  1. Check optimization_results.json for the best parameters")
    print("  2. Run evaluate() with the best params to generate a full report")
    print("  3. Compare muscle activations vs baseline using matplotlib")
    print("  4. If results look good → Stage 3 (RL with neural network policy)")
