from __future__ import annotations

import os
from pathlib import Path

import myosuite  # noqa: F401  # required for env registration
import numpy as np
from myosuite.utils import gym as myogym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

OUT_DIR = Path("rl_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_TIMESTEPS = 1_500_000
ENV_ID = "myoLegWalk-v0"
N_EVAL_EPISODES = 5
SEED = 42


def make_env() -> Monitor:
    """Create one monitored MyoSuite walking env."""
    env = myogym.make(ENV_ID)
    env = Monitor(env)
    return env


def build_vecnormalize(training: bool) -> VecNormalize:
    """Create a normalized vectorized env for SB3 PPO."""
    venv = DummyVecEnv([make_env])
    venv.seed(SEED)
    return VecNormalize(
        venv,
        training=training,
        norm_obs=True,
        norm_reward=training,
        clip_obs=10.0,
        gamma=0.99,
    )


def train_stage1_v2() -> None:
    train_env = build_vecnormalize(training=True)
    eval_env = build_vecnormalize(training=False)
    eval_env.obs_rms = train_env.obs_rms

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(OUT_DIR / "walker_stage1_v2_checkpoints"),
        name_prefix="walker_stage1_v2",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(OUT_DIR / "walker_stage1_v2_best"),
        log_path=str(OUT_DIR / "walker_stage1_v2_eval"),
        eval_freq=25_000,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
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
        tensorboard_log=str(OUT_DIR / "tb_stage1_v2"),
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.Tanh,
        ),
        seed=SEED,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    model_path = OUT_DIR / "walker_policy_stage1_v2"
    vecnorm_path = OUT_DIR / "walker_policy_stage1_v2_vecnormalize.pkl"
    model.save(str(model_path))
    train_env.save(str(vecnorm_path))

    print(f"Saved model to {model_path}.zip")
    print(f"Saved VecNormalize stats to {vecnorm_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 v2: PPO walker training with VecNormalize")
    print(f"  Env: {ENV_ID}")
    print(f"  Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Output dir: {OUT_DIR.resolve()}")
    print("=" * 60)
    train_stage1_v2()
