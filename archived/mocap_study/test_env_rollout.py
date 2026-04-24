import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl.envs.exo_with_walker_env import ExoWithWalkerSB3

# Create env
env = ExoWithWalkerSB3()
obs, _ = env.reset()

# Log storage
log = []

# Rollout
for i in range(100):
    action = np.zeros(2)

    obs, reward, terminated, truncated, info = env.step(action)

    # Store values each step
    log.append({
        "r_track": info.get("r_track", 0.0),
        "r_pos": info.get("r_pos", 0.0),
        "r_vel": info.get("r_vel", 0.0),
        "q_err": info.get("q_err_norm", 0.0),
        "dq_err": info.get("dq_err_norm", 0.0),
        "phase": info.get("phase", 0.0),
    })

    print(
        f"step={i} "
        f"phase={info.get('phase', 'MISSING'):.3f} "
        f"r_track={info.get('r_track', 0.0):.4f} "
        f"q_err={info.get('q_err_norm', 0.0):.4f} "
        f"dq_err={info.get('dq_err_norm', 0.0):.4f}"
    )

    if terminated or truncated:
        print("Episode ended early")
        break

# Convert to DataFrame
df = pd.DataFrame(log)

# Plot tracking reward
plt.figure()
plt.plot(df["r_track"])
plt.title("Tracking Reward")
plt.xlabel("Step")
plt.ylabel("r_track")

# Plot errors
plt.figure()
plt.plot(df["q_err"], label="q_err")
plt.plot(df["dq_err"], label="dq_err")
plt.legend()
plt.title("Tracking Errors")

# Plot phase
plt.figure()
plt.plot(df["phase"])
plt.title("Phase Progression")

plt.show()

env.close()