import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

path = Path("rl/rl_output/stage1/reference/p1_trial0_reference_raw.json")
with open(path, "r") as f:
    data = json.load(f)

x = np.arange(len(data["hip_flexion_r"]))

plt.figure(figsize=(10, 6))
plt.plot(x, data["hip_flexion_r"], label="hip_flexion_r")
plt.plot(x, data["hip_flexion_l"], label="hip_flexion_l")
plt.legend()
plt.tight_layout()

out1 = "rl/rl_output/stage1/reference/hip_angles.png"
plt.savefig(out1)
print("Saved:", out1)
plt.close()


plt.figure(figsize=(10, 6))
plt.plot(x, data["hip_flexion_r_vel"], label="hip_flexion_r_vel")
plt.plot(x, data["hip_flexion_l_vel"], label="hip_flexion_l_vel")
plt.legend()
plt.tight_layout()

out2 = "rl/rl_output/stage1/reference/hip_velocities.png"
plt.savefig(out2)
print("Saved:", out2)
plt.close()