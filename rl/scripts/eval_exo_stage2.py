import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.baselines.load_reference_walker import load_reference_walker
from rl.envs.exo_with_walker_env import ExoWithWalkerSB3

OUT_DIR = "rl_output"
RL_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT = RL_DIR / OUT_DIR


def _resolve_rl_path(path):
    if os.path.isabs(path):
        return path
    return str((RL_DIR / path).resolve())


def _normalize_step(result):
    if len(result) == 5:
        return result
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), False, info
    raise ValueError(f"Unexpected step result length: {len(result)}")


def _current_effort(sim):
    act = sim.data.act
    if act is not None and len(act) > 0:
        return float(np.mean(act**2))
    return 0.0


def _init_rollout():
    return {
        "condition": [],
        "episode": [],
        "step": [],
        "reward": [],
        "effort": [],
        "mean_torque": [],
        "termination_source": [],
        "termination_reason": [],
    }


def _append_rollout(rollout, condition, episode, step, reward, effort, mean_torque, termination_source, termination_reason):
    rollout["condition"].append(condition)
    rollout["episode"].append(episode)
    rollout["step"].append(step)
    rollout["reward"].append(float(reward))
    rollout["effort"].append(float(effort))
    rollout["mean_torque"].append(float(mean_torque))
    rollout["termination_source"].append(termination_source)
    rollout["termination_reason"].append(termination_reason)


def _print_step(label, episode, step, reward, effort, mean_torque, termination_reason):
    print(
        f"[{label}] ep={episode} step={step} "
        f"reward={reward:.3f} effort={effort:.6f} "
        f"|tau|={mean_torque:.3f} term={termination_reason}"
    )


def _print_summary(label, rewards, efforts, torques, episode_lengths):
    print("\n" + "=" * 60)
    print(label)
    print(f"  mean reward: {np.mean(rewards):.3f}")
    print(f"  mean effort: {np.mean(efforts):.6f}")
    print(f"  mean torque: {np.mean(torques):.3f}")
    print(f"  episode length: {np.mean(episode_lengths):.2f}")
    print("=" * 60)


def _save_rollout(rollout):
    os.makedirs(OUT_ROOT, exist_ok=True)
    out_path = OUT_ROOT / "stage2_eval_rollout.npz"
    np.savez(out_path, **{key: np.array(value) for key, value in rollout.items()})
    print(f"Saved rollout to: {out_path}")


def _eval_walker_only(walker_path, n_episodes, max_steps, rollout):
    from myosuite.utils import gym as myogym

    env = myogym.make("myoLegWalk-v0")
    walker = load_reference_walker(walker_path)

    rewards = []
    efforts = []
    torques = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        for step in range(1, max_steps + 1):
            action, _ = walker.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = _normalize_step(env.step(action))

            effort = _current_effort(env.unwrapped.sim)
            mean_torque = 0.0

            termination_source = None
            termination_reason = None
            if terminated:
                termination_source = "env_done"
                termination_reason = info.get("termination_reason") or "env_done"
            elif truncated or step >= max_steps:
                termination_source = "time_limit"
                termination_reason = info.get("termination_reason") or "max_steps"

            _append_rollout(
                rollout,
                "walker_only",
                ep,
                step,
                reward,
                effort,
                mean_torque,
                termination_source,
                termination_reason,
            )
            _print_step(
                "WALKER",
                ep,
                step,
                float(reward),
                effort,
                mean_torque,
                termination_reason,
            )

            rewards.append(float(reward))
            efforts.append(effort)
            torques.append(mean_torque)

            if terminated or truncated or step >= max_steps:
                episode_lengths.append(step)
                break

    env.close()
    _print_summary("Walker-only", rewards, efforts, torques, episode_lengths)


def _eval_walker_plus_exo(walker_path, exo_path, n_episodes, max_steps, rollout):
    env = ExoWithWalkerSB3(walker_path, max_steps=max_steps)
    exo = PPO.load(exo_path, device="cpu")

    rewards = []
    efforts = []
    torques = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for step in range(1, max_steps + 1):
            action, _ = exo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            effort = float(info["current_effort"])
            mean_torque = float(info["mean_abs_torque"])
            termination_source = info.get("termination_source")
            termination_reason = info.get("termination_reason")
            if truncated and termination_reason is None:
                termination_reason = "max_steps"

            _append_rollout(
                rollout,
                "walker_plus_exo",
                ep,
                step,
                reward,
                effort,
                mean_torque,
                termination_source,
                termination_reason,
            )
            _print_step(
                "EXO",
                ep,
                step,
                float(reward),
                effort,
                mean_torque,
                termination_reason,
            )

            rewards.append(float(reward))
            efforts.append(effort)
            torques.append(mean_torque)

            if terminated or truncated:
                episode_lengths.append(step)
                break

    env.close()
    _print_summary("Walker + exo", rewards, efforts, torques, episode_lengths)


def run_eval(walker_path, exo_path, n_episodes=5, max_steps=300):
    walker_path = _resolve_rl_path(walker_path)
    exo_path = _resolve_rl_path(exo_path)

    rollout = _init_rollout()
    _eval_walker_only(walker_path, n_episodes, max_steps, rollout)
    _eval_walker_plus_exo(walker_path, exo_path, n_episodes, max_steps, rollout)
    _save_rollout(rollout)


if __name__ == "__main__":
    walker_path = os.path.join(OUT_DIR, "walker_policy")
    exo_path = os.path.join(OUT_DIR, "exo_policy")
    run_eval(walker_path, exo_path)
