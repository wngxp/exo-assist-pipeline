#!/usr/bin/env python3
"""
Compare muscle activations across conditions:
  - Baseline (human only)
  - Exo locked (dead weight)
  - Exo active (future)

Reads .sto solution files from each condition subfolder and plots key hip
muscles side by side.

Run from: ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
Prereqs:  conda activate opensim (or just needs numpy + matplotlib)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONDITIONS = {
    'Baseline (no exo)': 'baseline/baseline_MocoInverse_solution.sto',
    'Exo locked (dead weight)': 'exo_locked/exo_locked_MocoInverse_solution.sto',
    # 'Exo active (assistance)': 'exo_active/exo_active_MocoInverse_solution.sto',
}

# Key hip muscles to compare — these are most affected by the exoskeleton
MUSCLES = [
    '/forceset/iliopsoas_r',
    '/forceset/iliopsoas_l',
    '/forceset/glmax1_r',
    '/forceset/glmax1_l',
    '/forceset/glmed1_r',
    '/forceset/glmed1_l',
    '/forceset/recfem_r',
    '/forceset/recfem_l',
    '/forceset/bflh_r',
    '/forceset/bflh_l',
]

# Short display names
MUSCLE_LABELS = {
    '/forceset/iliopsoas_r': 'Iliopsoas R',
    '/forceset/iliopsoas_l': 'Iliopsoas L',
    '/forceset/glmax1_r': 'Glut Max R',
    '/forceset/glmax1_l': 'Glut Max L',
    '/forceset/glmed1_r': 'Glut Med R',
    '/forceset/glmed1_l': 'Glut Med L',
    '/forceset/recfem_r': 'Rec Fem R',
    '/forceset/recfem_l': 'Rec Fem L',
    '/forceset/bflh_r': 'BF Long Head R',
    '/forceset/bflh_l': 'BF Long Head L',
}

COLORS = ['#2196F3', '#F44336', '#4CAF50']


def read_sto(filepath):
    """Read an OpenSim .sto file, return (time, column_names, data_array)."""
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.strip() == 'endheader':
                header_lines = i + 1
                break

    data = np.genfromtxt(filepath, skip_header=header_lines,
                         names=True, delimiter='\t', deletechars='')
    time = data[data.dtype.names[0]]
    return time, data.dtype.names, data


def normalize_to_gait_pct(time):
    """Convert time array to 0-100% gait cycle."""
    return (time - time[0]) / (time[-1] - time[0]) * 100


def main():
    # Load all condition data
    condition_data = {}
    for label, filepath in CONDITIONS.items():
        if os.path.exists(filepath):
            time, names, data = read_sto(filepath)
            gait_pct = normalize_to_gait_pct(time)
            condition_data[label] = (gait_pct, names, data)
            print(f"Loaded: {label} ({len(time)} frames)")
        else:
            print(f"SKIPPED (file not found): {label} -> {filepath}")

    if not condition_data:
        print("No data loaded. Run the MocoInverse scripts first.")
        return

    # Find which muscles are available in the solution files
    # MocoInverse outputs controls as /forceset/muscle_name
    available_muscles = []
    first_names = list(condition_data.values())[0][1]
    for m in MUSCLES:
        if m in first_names:
            available_muscles.append(m)
        else:
            print(f"  Column not found: {m}")

    if not available_muscles:
        print("No matching muscle columns found. Available columns:")
        for n in first_names[:20]:
            print(f"  {n}")
        return

    # Plot
    n_muscles = len(available_muscles)
    n_cols = 2
    n_rows = (n_muscles + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows),
                             sharex=True)
    axes = axes.flatten()

    for idx, muscle in enumerate(available_muscles):
        ax = axes[idx]
        for ci, (label, (gait_pct, names, data)) in enumerate(
                condition_data.items()):
            if muscle in names:
                ax.plot(gait_pct, data[muscle], label=label,
                        color=COLORS[ci], linewidth=1.5)
        ax.set_title(MUSCLE_LABELS.get(muscle, muscle), fontsize=10)
        ax.set_ylabel('Activation')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    for idx in range(n_muscles, len(axes)):
        axes[idx].set_visible(False)

    # Common x-label
    for ax in axes[-n_cols:]:
        ax.set_xlabel('Gait Cycle (%)')

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(condition_data),
               fontsize=11, frameon=False)

    plt.suptitle('Hip Muscle Activations: Baseline vs Exoskeleton Conditions',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    outfile = 'muscle_activation_comparison.png'
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    print(f"Saved: {outfile}")

    # Also compute and print summary stats
    print("\n--- Mean Activation Summary (hip muscles) ---")
    for muscle in available_muscles:
        print(f"\n{MUSCLE_LABELS.get(muscle, muscle)}:")
        for label, (gait_pct, names, data) in condition_data.items():
            if muscle in names:
                mean_act = np.mean(data[muscle])
                peak_act = np.max(data[muscle])
                print(f"  {label:30s}  mean={mean_act:.4f}  peak={peak_act:.4f}")


if __name__ == '__main__':
    main()
