from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROLLOUT_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage2_eval_rollout.npz")
FIGURE_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage2_torque_vs_gait_v2.png")
SUMMARY_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage2_torque_vs_gait_v2_summary.json")

MIN_CYCLE_STEPS = 8
MAX_CYCLE_STEPS = 80
PHASE_GRID = np.linspace(0.0, 1.0, 101)


def print_available_keys(data):
    print("Available rollout keys:")
    for key in data.files:
        arr = np.asarray(data[key])
        print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")


def first_matching_key(data, candidates):
    for key in candidates:
        if key in data.files:
            return key, np.asarray(data[key])
    return None, None


def require_same_length(name_a, array_a, name_b, array_b):
    if len(array_a) != len(array_b):
        raise ValueError(
            f"Array length mismatch: {name_a} has {len(array_a)} samples but "
            f"{name_b} has {len(array_b)} samples."
        )


def extract_torque(data):
    right_key, tau_r = first_matching_key(data, ["tau_r"])
    left_key, tau_l = first_matching_key(data, ["tau_l"])

    if tau_r is not None and tau_l is not None:
        print(f"Using torque keys: {right_key}, {left_key}")
        return tau_r.astype(float), tau_l.astype(float)

    actions_key, actions = first_matching_key(data, ["actions"])
    if actions is not None:
        if actions.ndim != 2 or actions.shape[1] < 2:
            raise KeyError(
                f"Key '{actions_key}' exists but does not have shape [N, 2+]. "
                "Cannot extract left/right torque from it."
            )
        print(f"Using torque columns from key: {actions_key}")
        return actions[:, 0].astype(float), actions[:, 1].astype(float)

    raise KeyError(
        "Missing torque arrays. Expected 'tau_r'/'tau_l' or an 'actions' array with two columns."
    )


def extract_hip_velocity(data):
    right_key, hipd_r = first_matching_key(data, ["hipd_r"])
    left_key, hipd_l = first_matching_key(data, ["hipd_l"])

    if hipd_r is None or hipd_l is None:
        raise KeyError(
            "Missing hip velocity arrays. Expected both 'hipd_r' and 'hipd_l' in the rollout."
        )

    print(f"Using hip-velocity keys: {right_key}, {left_key}")
    return hipd_r.astype(float), hipd_l.astype(float)


def detect_cycle_starts(hipd_r):
    centered_hipd_r = hipd_r - np.mean(hipd_r)
    starts = []

    for idx in range(1, len(centered_hipd_r)):
        prev_value = centered_hipd_r[idx - 1]
        curr_value = centered_hipd_r[idx]
        if prev_value < 0.0 and curr_value >= 0.0:
            starts.append(idx)

    return starts, centered_hipd_r


def build_valid_cycles(cycle_starts):
    valid_cycles = []
    cycle_lengths = []

    for start_idx, stop_idx in zip(cycle_starts[:-1], cycle_starts[1:]):
        cycle_length = int(stop_idx - start_idx)
        if cycle_length < MIN_CYCLE_STEPS or cycle_length > MAX_CYCLE_STEPS:
            continue

        valid_cycles.append((start_idx, stop_idx))
        cycle_lengths.append(cycle_length)

    return valid_cycles, cycle_lengths


def interpolate_cycle(values, start_idx, stop_idx):
    cycle_values = np.asarray(values[start_idx:stop_idx], dtype=float)
    if len(cycle_values) < 2:
        raise ValueError(f"Cycle [{start_idx}, {stop_idx}) has fewer than 2 samples.")

    source_phase = np.linspace(0.0, 1.0, len(cycle_values), endpoint=True)
    return np.interp(PHASE_GRID, source_phase, cycle_values)


def save_summary(num_samples, cycle_starts, cycle_lengths, tau_r, tau_l):
    summary = {
        "rollout_path": str(ROLLOUT_PATH),
        "number_of_samples": int(num_samples),
        "number_of_detected_cycle_starts": int(len(cycle_starts)),
        "number_of_cycles_used": int(len(cycle_lengths)),
        "cycle_lengths_in_steps": [int(length) for length in cycle_lengths],
        "mean_cycle_length": float(np.mean(cycle_lengths)),
        "min_cycle_length": int(np.min(cycle_lengths)),
        "max_cycle_length": int(np.max(cycle_lengths)),
        "mean_abs_torque_right": float(np.mean(np.abs(tau_r))),
        "mean_abs_torque_left": float(np.mean(np.abs(tau_l))),
        "peak_abs_torque_right": float(np.max(np.abs(tau_r))),
        "peak_abs_torque_left": float(np.max(np.abs(tau_l))),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary to: {SUMMARY_PATH}")


def main():
    if not ROLLOUT_PATH.exists():
        raise FileNotFoundError(f"Rollout file not found: {ROLLOUT_PATH}")

    data = np.load(ROLLOUT_PATH, allow_pickle=True)
    print_available_keys(data)

    tau_r, tau_l = extract_torque(data)
    hipd_r, hipd_l = extract_hip_velocity(data)
    sample_index = np.arange(len(tau_r))

    require_same_length("tau_r", tau_r, "tau_l", tau_l)
    require_same_length("tau_r", tau_r, "hipd_r", hipd_r)
    require_same_length("tau_r", tau_r, "hipd_l", hipd_l)

    cycle_starts, _ = detect_cycle_starts(hipd_r)
    valid_cycles, cycle_lengths = build_valid_cycles(cycle_starts)

    print(f"Detected cycle starts: {cycle_starts}")
    print(f"Detected cycle lengths in steps: {cycle_lengths}")
    print("Cycle detection used centered right hip angular velocity to remove DC offset.")

    if not cycle_starts:
        raise RuntimeError(
            "No right-leg negative-to-positive hip velocity zero crossings were detected."
        )
    if not valid_cycles:
        raise RuntimeError(
            "No valid gait cycles remained after filtering. "
            f"Expected cycle lengths between {MIN_CYCLE_STEPS} and {MAX_CYCLE_STEPS} steps."
        )

    right_cycle_traces = np.stack(
        [interpolate_cycle(tau_r, start_idx, stop_idx) for start_idx, stop_idx in valid_cycles]
    )
    left_cycle_traces = np.stack(
        [interpolate_cycle(tau_l, start_idx, stop_idx) for start_idx, stop_idx in valid_cycles]
    )

    mean_right_trace = np.mean(right_cycle_traces, axis=0)
    mean_left_trace = np.mean(left_cycle_traces, axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(sample_index, tau_r, color="#c44e52", linewidth=1.5, label="Right torque")
    axes[0].plot(sample_index, tau_l, color="#4c72b0", linewidth=1.5, label="Left torque")
    for idx, start_idx in enumerate(cycle_starts):
        label = "Detected cycle start" if idx == 0 else None
        axes[0].axvline(
            start_idx,
            color="0.45",
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label=label,
        )
    axes[0].set_title("Exoskeleton Torque vs Rollout Sample")
    axes[0].set_xlabel("Rollout sample index")
    axes[0].set_ylabel("Torque")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    for trace in right_cycle_traces:
        axes[1].plot(PHASE_GRID, trace, color="#f1a3a3", linewidth=1.0, alpha=0.45)
    for trace in left_cycle_traces:
        axes[1].plot(PHASE_GRID, trace, color="#a8c5f0", linewidth=1.0, alpha=0.45)
    axes[1].plot(
        PHASE_GRID,
        mean_right_trace,
        color="#8c1d40",
        linewidth=2.8,
        label="Mean right torque",
    )
    axes[1].plot(
        PHASE_GRID,
        mean_left_trace,
        color="#1f4e79",
        linewidth=2.8,
        label="Mean left torque",
    )
    axes[1].set_title("Exoskeleton Torque vs Right-Leg Gait Phase")
    axes[1].set_xlabel("Normalized right-leg gait phase")
    axes[1].set_ylabel("Torque")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)

    save_summary(len(tau_r), cycle_starts, cycle_lengths, tau_r, tau_l)

    print(f"Rollout length: {len(tau_r)}")
    print(
        "Torque stats:"
        f" mean_abs_right={np.mean(np.abs(tau_r)):.3f},"
        f" mean_abs_left={np.mean(np.abs(tau_l)):.3f},"
        f" peak_abs_right={np.max(np.abs(tau_r)):.3f},"
        f" peak_abs_left={np.max(np.abs(tau_l)):.3f}"
    )
    print(f"Saved figure to: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
