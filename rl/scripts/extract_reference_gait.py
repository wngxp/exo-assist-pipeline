from __future__ import annotations

from pathlib import Path
import json

import nimblephysics as nimble
import numpy as np

B3D_PATH = Path("data/addbiomechanics/train/No_Arm/vanderZee2022_Formatted_No_Arm/p1/p1.b3d")
OUT_DIR = Path("rl/rl_output/stage1/reference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRIAL = 0

IDX = {
    "hip_flexion_r": 6,
    "hip_flexion_l": 13,
    "knee_angle_r": 9,
    "knee_angle_l": 16,
    "ankle_angle_r": 10,
    "ankle_angle_l": 17,
}

def main() -> None:
    dataset = nimble.biomechanics.SubjectOnDisk(str(B3D_PATH))
    num_passes = dataset.getNumProcessingPasses()
    num_frames = dataset.getTrialLength(TRIAL)

    times = []
    pos = []
    vel = []

    for t in range(num_frames):
        frame = dataset.readFrames(TRIAL, t, 1, num_passes - 1)[0]
        pp = frame.processingPasses[num_passes - 1]
        times.append(frame.t)
        pos.append(np.array(pp.pos, dtype=float))
        vel.append(np.array(pp.vel, dtype=float))

    times = np.array(times)
    pos = np.vstack(pos)
    vel = np.vstack(vel)

    out = {
        "time": times.tolist(),
        "hip_flexion_r": pos[:, IDX["hip_flexion_r"]].tolist(),
        "hip_flexion_l": pos[:, IDX["hip_flexion_l"]].tolist(),
        "hip_flexion_r_vel": vel[:, IDX["hip_flexion_r"]].tolist(),
        "hip_flexion_l_vel": vel[:, IDX["hip_flexion_l"]].tolist(),
        "knee_angle_r": pos[:, IDX["knee_angle_r"]].tolist(),
        "knee_angle_l": pos[:, IDX["knee_angle_l"]].tolist(),
        "ankle_angle_r": pos[:, IDX["ankle_angle_r"]].tolist(),
        "ankle_angle_l": pos[:, IDX["ankle_angle_l"]].tolist(),
    }

    out_path = OUT_DIR / "p1_trial0_reference_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Saved reference data to: {out_path.resolve()}")
    print(f"Frames: {num_frames}")
    print(f"Duration: {times[-1] - times[0]:.3f} s")

if __name__ == "__main__":
    main()