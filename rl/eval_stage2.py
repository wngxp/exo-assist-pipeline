import os
import numpy as np
from stable_baselines3 import PPO

from two_stage_exo import ExoWithWalkerSB3, OUT_DIR

def run_eval(walker_path, exo_path, n_episodes=5, max_steps=300):
    env = ExoWithWalkerSB3(walker_path, max_steps=max_steps)
    exo = PPO.load(exo_path)

    rollout = {
        "episode": [],
        "step": [],
        "reward": [],
        "effort": [],
        "tau_r": [],
        "tau_l": [],
        "hip_r": [],
        "hip_l": [],
        "hipd_r": [],
        "hipd_l": [],
        "pelvis_vx": [],
        "torso_pitch": [],
        "sin_phi_r": [],
        "cos_phi_r": [],
        "sin_phi_l": [],
        "cos_phi_l": [],
        "termination_source": [],
        "termination_reason": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            action, _ = exo.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            rollout["episode"].append(ep)
            rollout["step"].append(step)
            rollout["reward"].append(float(reward))
            rollout["effort"].append(float(info["current_effort"]))
            rollout["tau_r"].append(float(action[0]))
            rollout["tau_l"].append(float(action[1]))
            rollout["sin_phi_r"].append(float(obs[0]))
            rollout["cos_phi_r"].append(float(obs[1]))
            rollout["sin_phi_l"].append(float(obs[2]))
            rollout["cos_phi_l"].append(float(obs[3]))
            rollout["hip_r"].append(float(obs[4]))
            rollout["hip_l"].append(float(obs[5]))
            rollout["hipd_r"].append(float(obs[6]))
            rollout["hipd_l"].append(float(obs[7]))
            rollout["pelvis_vx"].append(float(obs[8]))
            rollout["torso_pitch"].append(float(obs[9]))

            obs = next_obs

            if terminated or truncated:
                term_source = info.get("termination_source")
                term_reason = info.get("termination_reason")

                rollout["termination_source"].append(term_source)
                rollout["termination_reason"].append(term_reason)

                print(
                    f"[EVAL END] ep={ep} step={step} "
                    f"source={term_source} reason={term_reason} "
                    f"effort={info.get('current_effort'):.6f}"
                )
                break

        # pad termination info for steps without termination
        while len(rollout["termination_source"]) < len(rollout["episode"]):
            rollout["termination_source"].append(None)
            rollout["termination_reason"].append(None)

    env.close()
    out_path = os.path.join(OUT_DIR, "stage2_eval_rollout.npz")
    np.savez(out_path, **{k: np.array(v) for k, v in rollout.items()})
    print(f"Saved rollout to: {out_path}")

if __name__ == "__main__":
    walker_path = os.path.join(OUT_DIR, "walker_policy")
    exo_path = os.path.join(OUT_DIR, "exo_policy")
    run_eval(walker_path, exo_path)