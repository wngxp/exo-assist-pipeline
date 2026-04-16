from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROLLOUT_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage2_eval_rollout.npz")
OUTPUT_PATH = Path("/Users/wxp/dev/exo-assist-pipeline/rl/rl_output/stage2_torque_vs_gait.png")
PHASE_BINS = 100
SMOOTH_WINDOW = 7


def print_available_keys(data):
    print("Available rollout keys:")
    for key in data.files:
        arr = data[key]
        print(f"  - {key}: shape={arr.shape}, dtype={arr.dtype}")


def first_matching_key(data, candidates):
    for key in candidates:
        if key in data.files:
            return key, np.asarray(data[key])
    return None, None


def extract_torque(data):
    right_key, tau_r = first_matching_key(
        data,
        ["tau_r", "torque_r", "right_torque", "action_r"],
    )
    left_key, tau_l = first_matching_key(
        data,
        ["tau_l", "torque_l", "left_torque", "action_l"],
    )

    if tau_r is not None and tau_l is not None:
        print(f"Using torque keys: {right_key}, {left_key}")
        return tau_r.astype(float), tau_l.astype(float)

    array_key, actions = first_matching_key(data, ["actions", "torque", "action"])
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 2:
        print(f"Using torque columns from key: {array_key}")
        return actions[:, 0].astype(float), actions[:, 1].astype(float)

    raise KeyError("Could not find left/right torque arrays in rollout. See available keys above.")


def extract_hip_velocity(data):
    right_key, hipd_r = first_matching_key(
        data,
        ["hipd_r", "hip_r_vel", "right_hip_velocity", "hip_velocity_r"],
    )
    left_key, hipd_l = first_matching_key(
        data,
        ["hipd_l", "hip_l_vel", "left_hip_velocity", "hip_velocity_l"],
    )

    if hipd_r is not None and hipd_l is not None:
        print(f"Using hip-velocity keys: {right_key}, {left_key}")
        return hipd_r.astype(float), hipd_l.astype(float)

    print("Hip angular velocity keys not found; falling back to normalized time within each episode.")
    return None, None


def moving_average(values, window):
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def phase_from_zero_crossings(episode_ids, hip_velocity):
    phase = np.full(len(hip_velocity), np.nan, dtype=float)

    for episode in np.unique(episode_ids):
        mask = episode_ids == episode
        idx = np.flatnonzero(mask)
        vel = hip_velocity[idx]

        crossings = []
        for i in range(1, len(vel)):
            if vel[i - 1] <= 0.0 < vel[i]:
                crossings.append(i)

        if len(crossings) >= 2:
            for start, stop in zip(crossings[:-1], crossings[1:]):
                segment = np.arange(start, stop + 1)
                if len(segment) <= 1:
                    continue
                phase[idx[segment]] = np.linspace(0.0, 1.0, len(segment), endpoint=True)
        else:
            phase[idx] = np.linspace(0.0, 1.0, len(idx), endpoint=False)

        missing = np.isnan(phase[idx])
        if np.any(missing):
            phase[idx[missing]] = np.linspace(0.0, 1.0, missing.sum(), endpoint=False)

    return phase


def phase_from_episode_time(episode_ids):
    phase = np.zeros(len(episode_ids), dtype=float)
    for episode in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == episode)
        if len(idx) == 0:
            continue
        phase[idx] = np.linspace(0.0, 1.0, len(idx), endpoint=False)
    return phase


def binned_phase_curve(phase, values, n_bins=PHASE_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    digitized = np.digitize(phase, bins) - 1

    means = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        mask = digitized == i
        if np.any(mask):
            means[i] = np.mean(values[mask])

    valid = np.isfinite(means)
    if np.any(valid):
        means[valid] = moving_average(means[valid], SMOOTH_WINDOW)
    return centers, means


def add_episode_boundaries(ax, episode_ids):
    change_points = np.flatnonzero(np.diff(episode_ids)) + 0.5
    for boundary in change_points:
        ax.axvline(boundary, color="0.85", linewidth=0.8, linestyle="--")


def main():
    if not ROLLOUT_PATH.exists():
        raise FileNotFoundError(f"Rollout file not found: {ROLLOUT_PATH}")

    data = np.load(ROLLOUT_PATH, allow_pickle=True)
    print_available_keys(data)

    tau_r, tau_l = extract_torque(data)
    hipd_r, hipd_l = extract_hip_velocity(data)

    episode_ids = np.asarray(data["episode"]) if "episode" in data.files else np.zeros(len(tau_r), dtype=int)
    sample_axis = np.arange(len(tau_r))

    if hipd_r is not None:
        phase = phase_from_zero_crossings(episode_ids, hipd_r)
        print("Estimated gait phase from right hip angular velocity zero-crossings.")
    else:
        phase = phase_from_episode_time(episode_ids)
        print("Estimated gait phase from normalized episode time.")

    phase_centers, phase_tau_r = binned_phase_curve(phase, tau_r)
    _, phase_tau_l = binned_phase_curve(phase, tau_l)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    axes[0].plot(sample_axis, tau_r, label="Right torque", color="#c44e52", linewidth=1.5)
    axes[0].plot(sample_axis, tau_l, label="Left torque", color="#4c72b0", linewidth=1.5)
    add_episode_boundaries(axes[0], episode_ids)
    axes[0].set_title("Exoskeleton Torque vs Time")
    axes[0].set_xlabel("Rollout sample")
    axes[0].set_ylabel("Torque")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].scatter(phase, tau_r, s=14, alpha=0.2, color="#c44e52", label="Right torque samples")
    axes[1].scatter(phase, tau_l, s=14, alpha=0.2, color="#4c72b0", label="Left torque samples")
    axes[1].plot(phase_centers, phase_tau_r, color="#8c1d40", linewidth=2.5, label="Right torque mean")
    axes[1].plot(phase_centers, phase_tau_l, color="#1f4e79", linewidth=2.5, label="Left torque mean")
    axes[1].axvspan(0.0, 0.5, color="0.95", alpha=0.8)
    axes[1].axvspan(0.5, 1.0, color="0.9", alpha=0.4)
    axes[1].set_title("Exoskeleton Torque vs Gait Phase")
    axes[1].set_xlabel("Gait phase")
    axes[1].set_ylabel("Torque")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)

    print(f"Rollout length: {len(tau_r)}")
    print(
        "Torque stats:"
        f" mean_right={np.mean(tau_r):.3f}, mean_left={np.mean(tau_l):.3f},"
        f" max_abs_right={np.max(np.abs(tau_r)):.3f}, max_abs_left={np.max(np.abs(tau_l)):.3f}"
    )
    print(f"Saved figure to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
