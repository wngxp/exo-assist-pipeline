import os
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList

from rl.mocap_study.envs import MocapWalkerEnv


RUN_NAME = "mocap_stage1_walker_ppo_v1"
BASE_DIR = Path("rl/mocap_study/output/training") / RUN_NAME
LOG_DIR = BASE_DIR / "logs"
CKPT_DIR = BASE_DIR / "checkpoints"
BEST_DIR = BASE_DIR / "best_model"
FINAL_MODEL_PATH = BASE_DIR / "final_model"


def make_env():
    env = MocapWalkerEnv()
    env = Monitor(env)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(BEST_DIR, exist_ok=True)

    train_env = make_env()
    eval_env = make_env()

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(CKPT_DIR),
        name_prefix="ppo_mocap_stage1_walker",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(BASE_DIR / "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    total_timesteps = 200_000

    print(f"Starting training: {RUN_NAME}")
    print(f"Output dir: {BASE_DIR}")
    print(f"Total timesteps: {total_timesteps}")
    print("Environment: MocapWalkerEnv (walker-only, no exo, no DEP-RL)")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=RUN_NAME,
    )

    model.save(str(FINAL_MODEL_PATH))
    print(f"Saved final model to: {FINAL_MODEL_PATH}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()