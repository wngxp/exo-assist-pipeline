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
plt.xlabel("sample")
plt.ylabel("angle")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, data["hip_flexion_r_vel"], label="hip_flexion_r_vel")
plt.plot(x, data["hip_flexion_l_vel"], label="hip_flexion_l_vel")
plt.xlabel("sample")
plt.ylabel("velocity")
plt.legend()
plt.tight_layout()
plt.show()