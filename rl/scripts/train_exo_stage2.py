import os
import sys
import time
from pathlib import Path

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.baselines.load_deprl_reference import BASELINE_DIR
from rl.envs.exo_with_walker_env import ExoWithWalkerSB3

OUT_DIR = "rl_output"
RL_DIR = Path(__file__).resolve().parents[1]
OUT_ROOT = RL_DIR / OUT_DIR


def _resolve_rl_path(path):
    if path is None:
        return str(BASELINE_DIR.resolve())
    if os.path.isabs(path):
        return path
    return str((RL_DIR / path).resolve())


def train_exo(walker_path=None, total_timesteps=1_000_000):
    walker_path = _resolve_rl_path(walker_path)
    os.makedirs(OUT_ROOT, exist_ok=True)

    print("\n" + "=" * 60)
    print("Stage 2: Training Exo Policy (SB3 PPO)")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Walker baseline: {walker_path}")
    print("=" * 60)

    train_env = Monitor(ExoWithWalkerSB3(walker_path))
    eval_env = Monitor(ExoWithWalkerSB3(walker_path))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.00,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(OUT_ROOT / "tb_exo"),
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.Tanh,
        ),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=str(OUT_ROOT / "exo_checkpoints"),
        name_prefix="exo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(OUT_ROOT / "exo_best"),
        log_path=str(OUT_ROOT / "exo_eval"),
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )
    train_time = time.time() - t0

    save_path = OUT_ROOT / "exo_policy"
    model.save(str(save_path))

    print(f"\n  Stage 2 complete in {train_time/60:.1f} min")
    print(f"  Saved: {save_path}.zip")

    train_env.close()
    eval_env.close()
    return str(save_path)


if __name__ == "__main__":
    train_exo()
