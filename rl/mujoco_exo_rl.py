#!/usr/bin/env python3
"""
MuJoCo/MyoSuite Hip Exoskeleton RL — Setup & Benchmark

Step 1: Install MyoSuite
    conda create --name myosuite python=3.9
    conda activate myosuite
    pip install myosuite torch stable-baselines3 -i https://pypi.tuna.tsinghua.edu.cn/simple

Step 2: Run this script to:
    a) Verify myoLeg loads and walks
    b) Benchmark simulation speed (compare to OpenSim's 8 steps/sec)
    c) Run a quick PPO training on the built-in walking task
    d) Add exo torque actuators and train with exo assistance

Run on cortex:
    conda activate myosuite
    python3 mujoco_exo_rl.py
"""
import time
import numpy as np

# =====================================================================
# PHASE 1: Verify MyoSuite installation and benchmark speed
# =====================================================================
def phase1_benchmark():
    """Load myoLeg, run random actions, measure steps/sec."""
    from myosuite.utils import gym

    print("=" * 60)
    print("Phase 1: MyoSuite Benchmark")
    print("=" * 60)

    env = gym.make('myoLegWalk-v0')
    obs = env.reset()

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space:      {env.action_space.shape}")
    print(f"  Action range:      [{env.action_space.low[0]:.1f}, {env.action_space.high[0]:.1f}]")

    # Benchmark: 10,000 random steps
    n_steps = 10_000
    t0 = time.time()
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            obs, info = env.reset()
    wall = time.time() - t0

    steps_per_sec = n_steps / wall
    print(f"\n  {n_steps} steps in {wall:.2f}s")
    print(f"  Steps/sec: {steps_per_sec:.0f}")
    print(f"  vs OpenSim: {steps_per_sec / 8.3:.0f}x faster")

    # RL training time estimates
    for label, n_ep in [("10K episodes", 10_000),
                        ("100K episodes", 100_000),
                        ("1M episodes", 1_000_000)]:
        total = n_ep * 200
        hours = total / steps_per_sec / 3600
        print(f"  {label:20s}: {hours:.1f} hours")

    env.close()
    return steps_per_sec


# =====================================================================
# PHASE 2: Train walking policy with PPO (no exo, baseline)
# =====================================================================
def phase2_baseline_ppo(total_timesteps=100_000):
    """Train a baseline walking policy using stable-baselines3 PPO."""
    from myosuite.utils import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import os

    print("\n" + "=" * 60)
    print("Phase 2: Baseline PPO (no exo)")
    print(f"  Timesteps: {total_timesteps:,}")
    print("=" * 60)

    # Vectorized envs for parallel data collection
    n_envs = 4
    def make_env():
        def _init():
            from myosuite.utils import gym as g
            return g.make('myoLegWalk-v0')
        return _init

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./rl_output/tb_logs/",
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps)
    train_time = time.time() - t0

    os.makedirs("rl_output", exist_ok=True)
    model.save("rl_output/myoleg_baseline_ppo")

    print(f"\n  Training time: {train_time/60:.1f} min")
    print(f"  Model saved: rl_output/myoleg_baseline_ppo.zip")

    env.close()
    return model


# =====================================================================
# PHASE 3: Custom exo environment — add WAWA hip torque actuators
# =====================================================================
def phase3_exo_env():
    """
    Create a custom Gym env that wraps myoLeg + adds exo torque.

    The exo policy observes hip angles/velocities (like real IMU)
    and outputs bilateral hip torques. The reward is:
      - Forward walking velocity (keep walking)
      - Negative muscle activation (reduce human effort)
      - Smoothness penalty (comfortable torque profile)
    """
    import gymnasium
    from gymnasium import spaces
    from myosuite.utils import gym

    class HipExoEnv(gymnasium.Env):
        """
        MyoLeg + bilateral hip exoskeleton.

        Observation (what real WAWA sensors provide):
            hip_angle_r, hip_angle_l         (from motor encoders)
            hip_velocity_r, hip_velocity_l   (from motor encoders)
            knee_angle_r, knee_angle_l       (bonus, not on real hw)
            prev_torque_r, prev_torque_l     (from controller)

        Action:
            exo_torque_r, exo_torque_l       (applied to hip flexion)
        """

        MAX_TORQUE = 12.0  # Nm, WAWA software limit

        def __init__(self):
            super().__init__()

            # Wrap the myoLeg walking environment
            self.base_env = gym.make('myoLegWalk-v0')

            # Our observation: 8-dim (like real WAWA sensors)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(8,), dtype=np.float32)

            # Our action: bilateral hip torque
            self.action_space = spaces.Box(
                low=-self.MAX_TORQUE, high=self.MAX_TORQUE,
                shape=(2,), dtype=np.float32)

            self.prev_torque = np.zeros(2)

            # Find hip actuator indices in the myoLeg model
            # These will need to be identified from the model's
            # actuator list — hip flexor/extensor muscle indices
            self._find_hip_indices()

        def _find_hip_indices(self):
            """Identify hip-related joint and actuator indices."""
            sim = self.base_env.sim
            model = sim.model

            # Print all joint names to find hip joints
            self.joint_names = [
                model.joint(i).name for i in range(model.njnt)]
            self.actuator_names = [
                model.actuator(i).name for i in range(model.nu)]

            # Find hip flexion joint indices (for observation)
            self.hip_r_qpos_idx = None
            self.hip_l_qpos_idx = None
            self.hip_r_qvel_idx = None
            self.hip_l_qvel_idx = None
            self.knee_r_qpos_idx = None
            self.knee_l_qpos_idx = None

            for i, name in enumerate(self.joint_names):
                name_lower = name.lower()
                if 'hip_flexion' in name_lower and '_r' in name_lower:
                    self.hip_r_qpos_idx = model.joint(i).qposadr
                    self.hip_r_qvel_idx = model.joint(i).dofadr
                elif 'hip_flexion' in name_lower and '_l' in name_lower:
                    self.hip_l_qpos_idx = model.joint(i).qposadr
                    self.hip_l_qvel_idx = model.joint(i).dofadr
                elif 'knee' in name_lower and '_r' in name_lower:
                    self.knee_r_qpos_idx = model.joint(i).qposadr
                elif 'knee' in name_lower and '_l' in name_lower:
                    self.knee_l_qpos_idx = model.joint(i).qposadr

            print(f"  Hip R qpos idx: {self.hip_r_qpos_idx}")
            print(f"  Hip L qpos idx: {self.hip_l_qpos_idx}")
            print(f"  Joints: {len(self.joint_names)}")
            print(f"  Actuators: {len(self.actuator_names)}")

        def reset(self, **kwargs):
            self.base_env.reset()
            self.prev_torque = np.zeros(2)
            return self._get_obs(), {}

        def step(self, action):
            action = np.clip(action, -self.MAX_TORQUE, self.MAX_TORQUE)

            # Get the base env's action (muscle activations)
            # For now, use zeros (no voluntary muscle activation)
            # In full version: a separate motion-imitation policy
            # drives the muscles while the exo policy adds torque
            base_action = np.zeros(self.base_env.action_space.shape)

            # Apply exo torque directly to hip joints via xfrc_applied
            # or by modifying the muscle activations
            # NOTE: This is a placeholder — the exact mechanism depends
            # on how myoLeg exposes joint torques. Options:
            #   1. Use mujoco's ctrl to add torque actuators to the XML
            #   2. Use xfrc_applied on hip bodies
            #   3. Add CoordinateActuators in the MJCF XML

            obs_base, reward_base, done, truncated, info = \
                self.base_env.step(base_action)

            # Apply exo torque via qfrc_applied (direct joint torque)
            sim = self.base_env.sim
            if self.hip_r_qvel_idx is not None:
                sim.data.qfrc_applied[self.hip_r_qvel_idx] = action[0]
            if self.hip_l_qvel_idx is not None:
                sim.data.qfrc_applied[self.hip_l_qvel_idx] = action[1]

            # Custom reward
            reward = self._compute_reward(action, reward_base)

            self.prev_torque = action.copy()
            return self._get_obs(), reward, done, truncated, info

        def _get_obs(self):
            """Extract WAWA-compatible observation."""
            sim = self.base_env.sim
            obs = np.zeros(8, dtype=np.float32)
            if self.hip_r_qpos_idx is not None:
                obs[0] = sim.data.qpos[self.hip_r_qpos_idx]
                obs[1] = sim.data.qpos[self.hip_l_qpos_idx]
                obs[2] = sim.data.qvel[self.hip_r_qvel_idx]
                obs[3] = sim.data.qvel[self.hip_l_qvel_idx]
            if self.knee_r_qpos_idx is not None:
                obs[4] = sim.data.qpos[self.knee_r_qpos_idx]
                obs[5] = sim.data.qpos[self.knee_l_qpos_idx]
            obs[6] = self.prev_torque[0]
            obs[7] = self.prev_torque[1]
            return obs

        def _compute_reward(self, action, base_reward):
            """
            Reward = walking reward + muscle effort reduction
                     - energy penalty - smoothness penalty
            """
            sim = self.base_env.sim

            # Muscle effort: sum of squared activations
            muscle_act = sim.data.act.copy() if sim.data.act is not None \
                else np.zeros(1)
            muscle_effort = np.mean(muscle_act ** 2)

            # Walking reward from base env (forward velocity)
            walk_reward = base_reward

            # Energy penalty
            energy_penalty = 0.001 * np.sum(action ** 2)

            # Smoothness penalty
            smooth_penalty = 0.01 * np.sum(
                (action - self.prev_torque) ** 2)

            reward = (walk_reward
                      - 0.5 * muscle_effort
                      - energy_penalty
                      - smooth_penalty)
            return reward

        def close(self):
            self.base_env.close()

    print("\n" + "=" * 60)
    print("Phase 3: Custom HipExoEnv")
    print("=" * 60)

    env = HipExoEnv()
    obs, _ = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs: {obs}")

    # Quick benchmark
    n = 1000
    t0 = time.time()
    for _ in range(n):
        action = env.action_space.sample()
        obs, r, done, trunc, info = env.step(action)
        if done:
            obs, _ = env.reset()
    wall = time.time() - t0
    print(f"  {n} steps in {wall:.2f}s ({n/wall:.0f} steps/sec)")

    env.close()
    return HipExoEnv


# =====================================================================
# MAIN
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("MuJoCo/MyoSuite Hip Exoskeleton RL")
    print("=" * 60)

    # Phase 1: Benchmark
    speed = phase1_benchmark()

    if speed < 100:
        print("\nMyoSuite too slow — something is wrong with install")
        exit(1)

    # Phase 2: Baseline walking (optional, takes ~10 min)
    import sys
    if '--baseline' in sys.argv:
        phase2_baseline_ppo(total_timesteps=100_000)

    # Phase 3: Custom exo environment
    if '--exo' in sys.argv:
        ExoEnvClass = phase3_exo_env()

    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("""
    1. Run Phase 1 to confirm speed:
         python3 mujoco_exo_rl.py

    2. Train baseline walking policy (~10 min):
         python3 mujoco_exo_rl.py --baseline

    3. Test custom exo environment:
         python3 mujoco_exo_rl.py --exo

    4. Once exo env works, train full exo policy:
         - Add WAWA URDF to myoLeg MJCF model
         - Train PPO with domain randomization
         - Compare muscle effort with vs without exo

    5. Deploy: Export policy MLP weights to Raspberry Pi
    """)