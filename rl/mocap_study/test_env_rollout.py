import numpy as np
from rl.envs.exo_with_walker_env import ExoWithWalkerSB3

env = ExoWithWalkerSB3()

obs, _ = env.reset()

for i in range(50):
    action = np.zeros(2)  # no exo torque yet

    obs, reward, terminated, truncated, info = env.step(action)

    print("info keys:", sorted(info.keys()))
    print(
        f"step={i} "
        f"phase={info.get('phase', 'MISSING')} "
        f"ref_idx={info.get('ref_index', 'MISSING')} "
        f"r_track={info.get('r_track', 'MISSING')} "
        f"q_err={info.get('q_err_norm', 'MISSING')} "
        f"dq_err={info.get('dq_err_norm', 'MISSING')}"
    )

    if terminated or truncated:
        print("Episode ended early")
        break

env.close()