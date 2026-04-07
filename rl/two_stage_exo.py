#!/usr/bin/env python3
"""
Two-stage hip exoskeleton RL training.

Stage 1: Train a walking policy (muscles learn to walk)
         ~30-60 min, saves to rl_output/walker_policy.zip
         
Stage 2: Freeze the walker, train exo policy on top
         The walker's muscles now RESPOND to exo torque because
         the walker policy reacts to changed dynamics.
         ~30-60 min, saves to rl_output/exo_policy_final.pt

Run on cortex:
    conda activate myosuite
    cd ~/repos/projects/exo-assist-pipeline/rl/
    
    # Stage 1 only (~30 min):
    python3 -u two_stage_exo.py --stage 1 2>&1 | tee ~/walker_train.log
    
    # Stage 2 only (after stage 1 finishes):
    python3 -u two_stage_exo.py --stage 2 2>&1 | tee ~/exo_train2.log
    
    # Both stages back-to-back (overnight):
    python3 -u two_stage_exo.py --stage both 2>&1 | tee ~/full_train.log
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
import sys
import json
from collections import deque

OUT_DIR = 'rl_output'
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# STAGE 1: Train walking policy with stable-baselines3
# =====================================================================
def stage1_train_walker(total_timesteps=500_000):
    """
    Train myoLeg to walk using PPO from stable-baselines3.
    This gives us a "human" that can walk and whose muscles
    respond naturally to perturbations (including exo torque).
    """
    from myosuite.utils import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback)

    print("=" * 60)
    print("Stage 1: Training myoLeg Walking Policy")
    print(f"  Timesteps: {total_timesteps:,}")
    print("=" * 60)

    # Create vectorized environment for parallel rollouts
    def make_env():
        from myosuite.utils import gym as g
        return g.make('myoLegWalk-v0')

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

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(OUT_DIR, 'walker_checkpoints'),
        name_prefix='walker',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(OUT_DIR, 'walker_best'),
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

    # Save final model
    save_path = os.path.join(OUT_DIR, 'walker_policy')
    model.save(save_path)

    print(f"\n  Stage 1 complete in {train_time/60:.1f} min")
    print(f"  Saved: {save_path}.zip")

    env.close()
    eval_env.close()
    return save_path


# =====================================================================
# STAGE 2: Train exo policy with SB3 on top of trained walker
# =====================================================================
import gymnasium as gym
from gymnasium import spaces

class ExoWithWalkerSB3(gym.Env):
    """
    Stage-1 bootstrap environment:
    - frozen walker supplies nominal locomotion
    - exo policy adds 2 hip torques
    - reward favors walking quality + lower effort proxy + smooth torque

    This is NOT the final co-adaptive architecture.
    It is a practical first assistance-training setup.
    """
    metadata = {"render_modes": []}
    MAX_TORQUE = 12.0

    def __init__(self, walker_path, max_steps=300):
        super().__init__()

        from myosuite.utils import gym as myogym
        from stable_baselines3 import PPO

        self.env = myogym.make("myoLegWalk-v0")
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.walker = PPO.load(walker_path, device="cpu")
        self.sim = self.env.unwrapped.sim
        self.model_mj = self.sim.model

        self.max_steps = max_steps
        self.step_count = 0

        self.last_done = False
        self.last_custom_terminated = False
        self.last_termination_reason = None
        self.last_termination_snapshot = {}

        self.prev_torque = np.zeros(2, dtype=np.float32)
        self.walker_obs = obs

        self._map_joints()
        self._map_root()
        self.baseline_effort = self._measure_baseline()

        # obs = [sin_phi_r, cos_phi_r, sin_phi_l, cos_phi_l,
        #        hip_r, hip_l, hipd_r, hipd_l,
        #        pelvis_vx, torso_pitch, prev_tau_r, prev_tau_l]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-self.MAX_TORQUE, high=self.MAX_TORQUE, shape=(2,), dtype=np.float32
        )

    def _map_joints(self):
        self.hip_r_qpos = None
        self.hip_l_qpos = None
        self.hip_r_qvel = None
        self.hip_l_qvel = None
        self.knee_r_qpos = None
        self.knee_l_qpos = None

        for i in range(self.model_mj.njnt):
            name = self.model_mj.joint(i).name.lower()
            if "hip_flexion" in name and "_r" in name:
                self.hip_r_qpos = self.model_mj.jnt_qposadr[i]
                self.hip_r_qvel = self.model_mj.jnt_dofadr[i]
            elif "hip_flexion" in name and "_l" in name:
                self.hip_l_qpos = self.model_mj.jnt_qposadr[i]
                self.hip_l_qvel = self.model_mj.jnt_dofadr[i]
            elif "knee" in name and "_r" in name:
                self.knee_r_qpos = self.model_mj.jnt_qposadr[i]
            elif "knee" in name and "_l" in name:
                self.knee_l_qpos = self.model_mj.jnt_qposadr[i]

        found = sum(
            x is not None for x in
            [self.hip_r_qpos, self.hip_l_qpos, self.hip_r_qvel, self.hip_l_qvel]
        )
        print(f"  Joint mapping: {found}/4 hip indices found")

    def _map_root(self):
        self.root_x_qvel = 0
        self.torso_pitch_qpos = None
        self.pelvis_height_qpos = None

        # try to find useful torso/root joints
        for i in range(self.model_mj.njnt):
            name = self.model_mj.joint(i).name.lower()
            if self.torso_pitch_qpos is None and ("pelvis_tilt" in name or "torso" in name):
                self.torso_pitch_qpos = self.model_mj.jnt_qposadr[i]
            if self.pelvis_height_qpos is None and (
                "pelvis_ty" in name or "root_ty" in name or "pelvis_y" in name
            ):
                self.pelvis_height_qpos = self.model_mj.jnt_qposadr[i]
            if "root_tx" in name or "pelvis_tx" in name:
                self.root_x_qvel = self.model_mj.jnt_dofadr[i]

    def _measure_baseline(self, n_episodes=3):
        print("  Measuring baseline effort (no exo)...")
        efforts = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]
            ep_efforts = []

            for _ in range(200):
                action, _ = self.walker.predict(obs, deterministic=True)
                result = self.env.step(action)
                obs = result[0]
                done = result[2]

                act = self.env.unwrapped.sim.data.act
                if act is not None and len(act) > 0:
                    ep_efforts.append(float(np.mean(act ** 2)))
                if done:
                    break

            if ep_efforts:
                efforts.append(np.mean(ep_efforts))

        baseline = float(np.mean(efforts)) if efforts else 0.01
        print(f"  Baseline muscle effort: {baseline:.6f}")
        return baseline

    def _estimate_phase(self):
        """
        Cheap bilateral phase proxy for Stage 1:
        derive a right-leg phase from right hip angle/velocity,
        then approximate the left leg as half a cycle out of phase.
        This is still only a proxy, but it gives the policy explicit
        left/right timing information.
        """
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        hip_r = qpos[self.hip_r_qpos] if self.hip_r_qpos is not None else 0.0
        hipd_r = qvel[self.hip_r_qvel] if self.hip_r_qvel is not None else 0.0

        phi_r = np.arctan2(hipd_r, hip_r)   # [-pi, pi]
        phi_l = phi_r + np.pi

        return (
            np.sin(phi_r), np.cos(phi_r),
            np.sin(phi_l), np.cos(phi_l),
        )

    def _get_obs(self):
        obs = np.zeros(12, dtype=np.float32)
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        sin_phi_r, cos_phi_r, sin_phi_l, cos_phi_l = self._estimate_phase()
        obs[0] = sin_phi_r
        obs[1] = cos_phi_r
        obs[2] = sin_phi_l
        obs[3] = cos_phi_l

        if self.hip_r_qpos is not None:
            obs[4] = qpos[self.hip_r_qpos]
        if self.hip_l_qpos is not None:
            obs[5] = qpos[self.hip_l_qpos]
        if self.hip_r_qvel is not None:
            obs[6] = qvel[self.hip_r_qvel]
        if self.hip_l_qvel is not None:
            obs[7] = qvel[self.hip_l_qvel]

        if self.root_x_qvel is not None and self.root_x_qvel < len(qvel):
            obs[8] = qvel[self.root_x_qvel]

        if self.torso_pitch_qpos is not None:
            obs[9] = qpos[self.torso_pitch_qpos]

        obs[10] = self.prev_torque[0] / self.MAX_TORQUE
        obs[11] = self.prev_torque[1] / self.MAX_TORQUE
        return obs

    def _compute_reward(self, action, base_reward, current_effort):
        # rough effort reduction proxy
        effort_reduction = (self.baseline_effort - current_effort) / max(self.baseline_effort, 1e-6)
        effort_bonus = 3.0 * effort_reduction

        # keep existing walking quality from MyoSuite
        walk_reward = float(base_reward)

        # encourage smaller torques
        energy_penalty = 0.003 * float(np.sum(action ** 2))

        # encourage smooth torques
        jerk_penalty = 0.01 * float(np.sum((action - self.prev_torque) ** 2))

        return walk_reward + effort_bonus - energy_penalty - jerk_penalty

    def _terminated(self):
        qpos = self.sim.data.qpos

        torso_pitch = 0.0
        if self.torso_pitch_qpos is not None:
            torso_pitch = float(qpos[self.torso_pitch_qpos])

        pelvis_h = 1.0
        if self.pelvis_height_qpos is not None:
            pelvis_h = float(qpos[self.pelvis_height_qpos])

        reason = None
        if abs(torso_pitch) > 1.2:
            reason = "torso_pitch"
        elif pelvis_h < 0.65:
            reason = "pelvis_height"

        snapshot = {
            "torso_pitch": torso_pitch,
            "pelvis_height": pelvis_h,
        }
        return reason is not None, reason, snapshot

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.walker_obs = obs
        self.sim = self.env.unwrapped.sim
        self.prev_torque[:] = 0.0
        self.step_count = 0

        self.last_done = False
        self.last_custom_terminated = False
        self.last_termination_reason = None
        self.last_termination_snapshot = {}

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE)

        # clear previously applied generalized forces
        self.sim.data.qfrc_applied[:] = 0.0

        # apply exo torques
        if self.hip_r_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_r_qvel] = float(action[0])
        if self.hip_l_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_l_qvel] = float(action[1])

        # walker chooses muscle action from the current last simulator state
        muscle_action, _ = self.walker.predict(self.walker_obs, deterministic=True)

        result = self.env.step(muscle_action)
        self.walker_obs = result[0]
        base_reward = float(result[1])
        done = bool(result[2])

        self.step_count += 1

        act = self.sim.data.act
        if act is not None and len(act) > 0:
            current_effort = float(np.mean(act ** 2))
        else:
            current_effort = self.baseline_effort

        reward = self._compute_reward(action, base_reward, current_effort)

        custom_terminated, custom_reason, custom_snapshot = self._terminated()
        terminated = done or custom_terminated
        truncated = self.step_count >= self.max_steps

        self.last_done = done
        self.last_custom_terminated = custom_terminated
        self.last_termination_reason = custom_reason
        self.last_termination_snapshot = custom_snapshot

        if truncated and not terminated:
            termination_source = "time_limit"
        elif done and custom_terminated:
            termination_source = "env+custom"
        elif done:
            termination_source = "env_done"
        elif custom_terminated:
            termination_source = "custom"
        else:
            termination_source = None

        info = {
            "current_effort": current_effort,
            "baseline_effort": self.baseline_effort,
            "effort_reduction_pct": 100.0 * (
                (self.baseline_effort - current_effort) / max(self.baseline_effort, 1e-6)
            ),
            "mean_abs_torque": float(np.mean(np.abs(action))),
            "env_done": bool(done),
            "custom_terminated": bool(custom_terminated),
            "termination_source": termination_source,
            "termination_reason": custom_reason,
            "torso_pitch": float(custom_snapshot["torso_pitch"]),
            "pelvis_height": float(custom_snapshot["pelvis_height"]),
            "step_count": int(self.step_count),
        }

        if terminated or truncated:
            print(
                "[EP END] "
                f"source={termination_source} "
                f"reason={custom_reason} "
                f"step={self.step_count} "
                f"base_reward={base_reward:.3f} "
                f"effort={current_effort:.6f} "
                f"|tau|={float(np.mean(np.abs(action))):.3f} "
                f"torso={float(custom_snapshot['torso_pitch']):.3f} "
                f"pelvis_h={float(custom_snapshot['pelvis_height']):.3f}"
            )

        self.prev_torque = action.copy()
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def stage2_train_exo(walker_path, total_timesteps=1_000_000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor

    print("\n" + "=" * 60)
    print("Stage 2: Training Exo Policy (SB3 PPO)")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Walker:    {walker_path}")
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
        tensorboard_log=os.path.join(OUT_DIR, "tb_exo"),
        device="cpu",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.Tanh,
        ),
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(OUT_DIR, "exo_checkpoints"),
        name_prefix="exo",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(OUT_DIR, "exo_best"),
        log_path=os.path.join(OUT_DIR, "exo_eval"),
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

    save_path = os.path.join(OUT_DIR, "exo_policy")
    model.save(save_path)

    print(f"\n  Stage 2 complete in {train_time/60:.1f} min")
    print(f"  Saved: {save_path}.zip")

    train_env.close()
    eval_env.close()


def sanity_check_exo_env(walker_path, n_steps=200):
    env = ExoWithWalkerSB3(walker_path)
    obs, _ = env.reset()

    rewards = []
    efforts = []
    torques = []
    termination_sources = []
    termination_reasons = []

    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(reward)
        efforts.append(info["current_effort"])
        torques.append(info["mean_abs_torque"])
        if terminated or truncated:
            termination_sources.append(info.get("termination_source"))
            termination_reasons.append(info.get("termination_reason"))

        if not np.all(np.isfinite(obs)):
            raise RuntimeError("Non-finite observation encountered")
        if not np.isfinite(reward):
            raise RuntimeError("Non-finite reward encountered")

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    print("\nSanity check passed")
    print(f"  mean reward: {np.mean(rewards):.3f}")
    print(f"  mean effort: {np.mean(efforts):.6f}")
    print(f"  mean |tau| : {np.mean(torques):.3f} Nm")
    if termination_sources:
        print(f"  termination sources: {termination_sources}")
        print(f"  termination reasons: {termination_reasons}")


def baseline_check_walker(walker_path, n_episodes=5, max_steps=200):
    """
    Run the trained walker alone on the raw MyoSuite env and report
    when the base environment signals done. This helps distinguish
    base-task horizon issues from exo-wrapper issues.
    """
    from myosuite.utils import gym as myogym
    from stable_baselines3 import PPO

    print("\n" + "=" * 60)
    print("Baseline walker-only diagnostic")
    print(f"  Episodes: {n_episodes}")
    print(f"  Max steps: {max_steps}")
    print("=" * 60)

    env = myogym.make("myoLegWalk-v0")
    walker = PPO.load(walker_path, device="cpu")

    lengths = []
    end_rewards = []

    for ep in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        done = False
        final_reward = None
        step_idx = 0

        for step_idx in range(1, max_steps + 1):
            action, _ = walker.predict(obs, deterministic=True)
            result = env.step(action)
            obs = result[0]
            final_reward = float(result[1])
            done = bool(result[2])

            if done:
                print(
                    f"[WALKER ONLY END] ep={ep} step={step_idx} "
                    f"reward={final_reward:.3f}"
                )
                break

        lengths.append(step_idx)
        end_rewards.append(final_reward if final_reward is not None else float("nan"))

    env.close()

    print("\nWalker-only summary")
    print(f"  mean episode length: {np.mean(lengths):.2f}")
    print(f"  std episode length : {np.std(lengths):.2f}")
    print(f"  lengths            : {lengths}")
    print(f"  end rewards        : {[round(x, 3) if np.isfinite(x) else x for x in end_rewards]}")

# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    stage = 'both'
    if '--stage' in sys.argv:
        idx = sys.argv.index('--stage')
        stage = sys.argv[idx + 1]

    walker_path = os.path.join(OUT_DIR, 'walker_policy')

    if stage in ['1', 'both']:
        stage1_train_walker(total_timesteps=500_000)

    if stage == 'check':
        if not os.path.exists(walker_path + '.zip'):
            print(f"ERROR: Walker not found at {walker_path}.zip")
            sys.exit(1)
        sanity_check_exo_env(walker_path, n_steps=200)

    if stage in ['2', 'both']:
        if not os.path.exists(walker_path + '.zip'):
            print(f"ERROR: Walker not found at {walker_path}.zip")
            print("  Run with --stage 1 first")
            sys.exit(1)

        sanity_check_exo_env(walker_path, n_steps=200)
        stage2_train_exo(walker_path, total_timesteps=1_000_000)

    print("\nAll done!")
    if stage == 'baseline':
        if not os.path.exists(walker_path + '.zip'):
            print(f"ERROR: Walker not found at {walker_path}.zip")
            sys.exit(1)
        baseline_check_walker(walker_path, n_episodes=5, max_steps=200)