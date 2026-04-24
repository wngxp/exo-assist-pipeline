"""Simple tracking-reward helpers for the surviving Stage 2 environment."""

from __future__ import annotations

import numpy as np

TRACKED_DOF_NAMES = [
    "hip_flexion_r",
    "knee_angle_r",
    "ankle_angle_r",
    "hip_flexion_l",
    "knee_angle_l",
    "ankle_angle_l",
]


def build_tracking_indices(ref_pos_cols):
    indices = []
    for name in TRACKED_DOF_NAMES:
        key = f"pos_{name}"
        if key not in ref_pos_cols:
            raise ValueError(f"Missing reference column: {key}")
        indices.append(ref_pos_cols.index(key))
    return indices


def compute_tracking_reward(q, dq, q_ref, dq_ref, track_idx):
    q = np.asarray(q, dtype=np.float64)
    dq = np.asarray(dq, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    dq_ref = np.asarray(dq_ref, dtype=np.float64)
    track_idx = np.asarray(track_idx, dtype=np.int64)

    q_err = q[track_idx] - q_ref[track_idx]
    dq_err = dq[track_idx] - dq_ref[track_idx]

    r_pos = -np.sum(q_err**2)
    r_vel = -0.1 * np.sum(dq_err**2)
    r_track = r_pos + r_vel

    return {
        "r_pos": float(r_pos),
        "r_vel": float(r_vel),
        "r_track": float(r_track),
        "q_err_norm": float(np.linalg.norm(q_err)),
        "dq_err_norm": float(np.linalg.norm(dq_err)),
    }
