#!/usr/bin/env python3
"""
Two-stage hip exoskeleton RL training.

Deprecated: Stage 2 has been refactored into the modular RL pipeline.
See `rl/scripts/train_exo_stage2.py` and `rl/scripts/eval_exo_stage2.py`.
"""

import os
import sys
import time
from pathlib import Path

import torch.nn as nn

OUT_DIR = "rl_output"
RL_DIR = Path(__file__).resolve().parent
OUT_ROOT = RL_DIR / OUT_DIR
os.makedirs(OUT_ROOT, exist_ok=True)


def stage1_train_walker(total_timesteps=500_000):
    """
    Train myoLeg to walk using PPO from stable-baselines3.
    This gives us a "human" that can walk and whose muscles
    respond naturally to perturbations (including exo torque).
    """
    from myosuite.utils import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env

    print("=" * 60)
    print("Stage 1: Training myoLeg Walking Policy")
    print(f"  Timesteps: {total_timesteps:,}")
    print("=" * 60)

    # Create vectorized environment for parallel rollouts
    def make_env():
        from myosuite.utils import gym as g

        return g.make("myoLegWalk-v0")

    env = make_vec_env(make_env, n_envs=4)

    # Eval env for periodic evaluation
    eval_env = make_vec_env(make_env, n_envs=1)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
        ),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(OUT_ROOT / "walker_checkpoints"),
        name_prefix="walker",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(OUT_ROOT / "walker_best"),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    print("\nTraining...")
    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
    )
    train_time = time.time() - t0

    save_path = OUT_ROOT / "walker_policy"
    model.save(str(save_path))

    print(f"\n  Stage 1 complete in {train_time/60:.1f} min")
    print(f"  Saved: {save_path}.zip")

    env.close()
    eval_env.close()
    return str(save_path)


def _print_stage2_deprecation():
    print("\nDeprecated: refactored into modular pipeline")
    print("  Train Stage 2: python rl/scripts/train_exo_stage2.py")
    print("  Eval Stage 2 : python rl/scripts/eval_exo_stage2.py")


if __name__ == "__main__":
    stage = "1"
    if "--stage" in sys.argv:
        idx = sys.argv.index("--stage")
        stage = sys.argv[idx + 1]

    if stage in ["1", "walker"]:
        stage1_train_walker(total_timesteps=500_000)
    else:
        _print_stage2_deprecation()
