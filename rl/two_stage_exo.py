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
# STAGE 2: Train exo policy with frozen walker
# =====================================================================

# --- Exo environment that uses the trained walker ---
class ExoWithWalker:
    """
    Environment where:
    - A trained PPO walker controls the 80 muscles (frozen)
    - The exo policy controls 2 hip torques
    - Muscles naturally respond to exo because the walker policy
      observes the changed state and adjusts its activations
    """

    MAX_TORQUE = 12.0  # WAWA limit

    def __init__(self, walker_path):
        from myosuite.utils import gym
        from stable_baselines3 import PPO

        self.env = gym.make('myoLegWalk-v0')
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Load frozen walker
        self.walker = PPO.load(walker_path)
        print(f"  Loaded walker from {walker_path}")

        self.sim = self.env.sim
        self.model_mj = self.sim.model

        # Map hip joint indices
        self._map_joints()

        self.prev_torque = np.zeros(2)
        self.step_count = 0

        # Measure baseline effort (walker without exo)
        self.baseline_effort = self._measure_baseline()

    def _map_joints(self):
        """Find hip and knee joint indices."""
        self.hip_r_qpos = None
        self.hip_l_qpos = None
        self.hip_r_qvel = None
        self.hip_l_qvel = None
        self.knee_r_qpos = None
        self.knee_l_qpos = None

        for i in range(self.model_mj.njnt):
            name = self.model_mj.joint(i).name.lower()
            if 'hip_flexion' in name and '_r' in name:
                self.hip_r_qpos = self.model_mj.jnt_qposadr[i]
                self.hip_r_qvel = self.model_mj.jnt_dofadr[i]
            elif 'hip_flexion' in name and '_l' in name:
                self.hip_l_qpos = self.model_mj.jnt_qposadr[i]
                self.hip_l_qvel = self.model_mj.jnt_dofadr[i]
            elif 'knee' in name and '_r' in name:
                self.knee_r_qpos = self.model_mj.jnt_qposadr[i]
            elif 'knee' in name and '_l' in name:
                self.knee_l_qpos = self.model_mj.jnt_qposadr[i]

        found = sum(x is not None for x in [
            self.hip_r_qpos, self.hip_l_qpos,
            self.hip_r_qvel, self.hip_l_qvel])
        print(f"  Joint mapping: {found}/4 hip indices found")

    def _measure_baseline(self, n_episodes=5):
        """Measure average muscle effort without exo assistance."""
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

                act = self.env.sim.data.act
                if act is not None and len(act) > 0:
                    ep_efforts.append(float(np.mean(act ** 2)))
                if done:
                    break
            if ep_efforts:
                efforts.append(np.mean(ep_efforts))

        baseline = np.mean(efforts) if efforts else 0.01
        print(f"  Baseline muscle effort: {baseline:.6f}")
        return baseline

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        self.walker_obs = obs
        self.sim = self.env.sim
        self.prev_torque = np.zeros(2)
        self.step_count = 0
        return self._get_exo_obs()

    def step(self, exo_action):
        """
        1. Apply exo torque to hip joints
        2. Walker policy generates muscle activations (reacting to exo)
        3. Step simulation
        4. Compute reward focused on effort reduction
        """
        exo_action = np.clip(exo_action, -self.MAX_TORQUE, self.MAX_TORQUE)

        # Apply exo torque BEFORE the walker decides muscle activations
        # This way the walker "feels" the exo and can reduce its effort
        if self.hip_r_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_r_qvel] = exo_action[0]
        if self.hip_l_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_l_qvel] = exo_action[1]

        # Walker decides muscle activations based on current state
        # (which includes the effect of exo torque from previous step)
        muscle_action, _ = self.walker.predict(
            self.walker_obs, deterministic=True)

        # Step simulation with muscle activations
        result = self.env.step(muscle_action)
        self.walker_obs = result[0]  # full obs for walker
        base_reward = float(result[1])
        done = result[2]

        self.step_count += 1

        # Measure muscle effort THIS step
        act = self.sim.data.act
        if act is not None and len(act) > 0:
            current_effort = float(np.mean(act ** 2))
        else:
            current_effort = self.baseline_effort

        # Reward: encourage effort reduction + maintain walking
        reward = self._compute_reward(
            exo_action, base_reward, current_effort)

        self.prev_torque = exo_action.copy()
        return self._get_exo_obs(), reward, done, current_effort

    def _get_exo_obs(self):
        """8-dim observation matching real WAWA sensors."""
        obs = np.zeros(8, dtype=np.float32)
        try:
            qpos = self.sim.data.qpos
            qvel = self.sim.data.qvel
            if self.hip_r_qpos is not None:
                obs[0] = qpos[self.hip_r_qpos]
            if self.hip_l_qpos is not None:
                obs[1] = qpos[self.hip_l_qpos]
            if self.hip_r_qvel is not None:
                obs[2] = qvel[self.hip_r_qvel]
            if self.hip_l_qvel is not None:
                obs[3] = qvel[self.hip_l_qvel]
            if self.knee_r_qpos is not None:
                obs[4] = qpos[self.knee_r_qpos]
            if self.knee_l_qpos is not None:
                obs[5] = qpos[self.knee_l_qpos]
            obs[6] = self.prev_torque[0] / self.MAX_TORQUE
            obs[7] = self.prev_torque[1] / self.MAX_TORQUE
        except Exception:
            pass
        return obs

    def _compute_reward(self, action, base_reward, current_effort):
        """
        Reward = walking_reward
               + effort_reduction_bonus (key metric!)
               - energy_penalty
               - smoothness_penalty
        """
        # Effort reduction relative to baseline
        effort_reduction = (self.baseline_effort - current_effort) / \
            max(self.baseline_effort, 1e-6)
        # Bonus for reducing effort (can be negative if effort increases)
        effort_bonus = 5.0 * effort_reduction

        # Walking reward (maintain gait quality)
        walk_reward = base_reward

        # Exo energy penalty
        energy_penalty = 0.005 * float(np.sum(action ** 2))

        # Smoothness
        jerk_penalty = 0.02 * float(np.sum(
            (action - self.prev_torque) ** 2))

        reward = walk_reward + effort_bonus - energy_penalty - jerk_penalty
        return reward

    def close(self):
        self.env.close()


# --- Exo policy network ---
class ExoPolicy(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * -1.0)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor_mean(h), self.critic(h)

    def get_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, value = self.forward(state_t)
            std = torch.exp(self.actor_log_std)
            dist = Normal(mean, std)
            raw = dist.sample()
            log_prob = dist.log_prob(raw).sum(-1)
        action = torch.tanh(raw) * 12.0  # WAWA limit
        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def evaluate(self, states, actions):
        mean, values = self.forward(states)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)
        raw = torch.atanh(torch.clamp(actions / 12.0, -0.999, 0.999))
        log_probs = dist.log_prob(raw).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy


# --- PPO update ---
def ppo_update(policy, optimizer, batch):
    states = torch.FloatTensor(np.array(batch['states']))
    actions = torch.FloatTensor(np.array(batch['actions']))
    old_lp = torch.FloatTensor(np.array(batch['log_probs']))
    returns = torch.FloatTensor(np.array(batch['returns']))
    advs = torch.FloatTensor(np.array(batch['advantages']))

    if len(advs) > 1:
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    for _ in range(4):
        lp, vals, ent = policy.evaluate(states, actions)
        ratio = torch.exp(lp - old_lp)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        p_loss = -torch.min(ratio * advs, clipped * advs).mean()
        v_loss = 0.5 * (returns - vals).pow(2).mean()
        loss = p_loss + v_loss - 0.01 * ent.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    return p_loss.item(), v_loss.item()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advs, rets = [], []
    gae, nv = 0, 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            nv, gae = 0, 0
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * gae
        advs.insert(0, gae)
        rets.insert(0, gae + values[t])
        nv = values[t]
    return rets, advs


# --- Stage 2 training loop ---
def stage2_train_exo(walker_path, n_episodes=50_000):
    print("\n" + "=" * 60)
    print("Stage 2: Training Exo Policy (with trained walker)")
    print(f"  Episodes:   {n_episodes:,}")
    print(f"  Walker:     {walker_path}")
    print("=" * 60)

    env = ExoWithWalker(walker_path)
    policy = ExoPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    reward_hist = deque(maxlen=100)
    effort_hist = deque(maxlen=100)
    torque_hist = deque(maxlen=100)
    length_hist = deque(maxlen=100)
    reduction_hist = deque(maxlen=100)

    batch = {k: [] for k in
             ['states', 'actions', 'log_probs',
              'rewards', 'values', 'dones']}

    BATCH_SIZE = 32
    best_reward = -float('inf')
    t0 = time.time()

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0
        ep_efforts = []
        ep_torques = []

        for step in range(200):
            action, lp, val = policy.get_action(obs)
            next_obs, reward, done, effort = env.step(action)

            batch['states'].append(obs)
            batch['actions'].append(action)
            batch['log_probs'].append(lp)
            batch['rewards'].append(reward)
            batch['values'].append(val)
            batch['dones'].append(done)

            ep_reward += reward
            ep_efforts.append(effort)
            ep_torques.append(np.abs(action).mean())
            obs = next_obs
            if done:
                break

        reward_hist.append(ep_reward)
        length_hist.append(step + 1)
        torque_hist.append(np.mean(ep_torques))
        avg_effort = np.mean(ep_efforts) if ep_efforts else 0
        effort_hist.append(avg_effort)
        reduction = (env.baseline_effort - avg_effort) / \
            max(env.baseline_effort, 1e-6) * 100
        reduction_hist.append(reduction)

        # PPO update
        if (ep + 1) % BATCH_SIZE == 0:
            rets, advs = compute_gae(
                batch['rewards'], batch['values'], batch['dones'])
            batch['returns'] = rets
            batch['advantages'] = advs
            ppo_update(policy, optimizer, batch)
            batch = {k: [] for k in batch}

        # Log
        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = (n_episodes - ep - 1) / ((ep + 1) / elapsed) / 60

            avg_r = np.mean(reward_hist)
            avg_t = np.mean(torque_hist)
            avg_e = np.mean(effort_hist)
            avg_red = np.mean(reduction_hist)
            avg_len = np.mean(length_hist)

            print(f"  Ep {ep+1:6d}/{n_episodes} | "
                  f"R={avg_r:7.1f} | "
                  f"Len={avg_len:5.1f} | "
                  f"τ={avg_t:4.1f}Nm | "
                  f"Effort={avg_e:.5f} | "
                  f"Reduction={avg_red:+.1f}% | "
                  f"ETA={eta:.0f}min")

            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(policy.state_dict(),
                           os.path.join(OUT_DIR, 'best_exo_policy.pt'))

        if (ep + 1) % 5000 == 0:
            torch.save(policy.state_dict(),
                       os.path.join(OUT_DIR, f'exo_ep{ep+1}.pt'))

    # Final save
    total_time = time.time() - t0
    torch.save(policy.state_dict(),
               os.path.join(OUT_DIR, 'exo_policy_final.pt'))

    final_reduction = float(np.mean(reduction_hist))
    log = {
        'n_episodes': n_episodes,
        'total_time_min': float(total_time / 60),
        'baseline_effort': float(env.baseline_effort),
        'final_avg_effort': float(np.mean(effort_hist)),
        'final_avg_reduction_pct': final_reduction,
        'final_avg_torque_nm': float(np.mean(torque_hist)),
        'final_avg_reward': float(np.mean(reward_hist)),
        'final_avg_length': float(np.mean(length_hist)),
        'best_reward': float(best_reward),
    }
    with open(os.path.join(OUT_DIR, 'exo_training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Stage 2 complete in {total_time/60:.1f} min")
    print("=" * 60)
    print(f"  Baseline effort:    {log['baseline_effort']:.6f}")
    print(f"  Final effort:       {log['final_avg_effort']:.6f}")
    print(f"  EFFORT REDUCTION:   {final_reduction:+.1f}%")
    print(f"  Avg torque:         {log['final_avg_torque_nm']:.1f} Nm")
    print(f"  Avg ep length:      {log['final_avg_length']:.0f} steps")
    print(f"  Saved: {OUT_DIR}/exo_policy_final.pt")

    env.close()


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

    if stage in ['2', 'both']:
        if not os.path.exists(walker_path + '.zip'):
            print(f"ERROR: Walker not found at {walker_path}.zip")
            print("  Run with --stage 1 first")
            sys.exit(1)
        stage2_train_exo(walker_path, n_episodes=50_000)

    print("\nAll done!")