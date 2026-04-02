#!/usr/bin/env python3
"""
Hip Exoskeleton RL Training on MyoSuite myoLeg.

Architecture:
  - myoLeg walks using its built-in muscle controller (random baseline)
  - Exo policy observes hip kinematics → outputs bilateral hip torques
  - Reward = walking velocity - muscle effort - energy cost - jerk penalty

The exo torque is applied via qfrc_applied at each timestep (100 Hz),
so the policy learns a state-dependent torque profile, not a constant.

Run on cortex:
    conda activate myosuite
    cd ~/repos/projects/exo-assist-pipeline/rl/
    tmux new -s exo_train
    python3 train_exo_myoleg.py 2>&1 | tee ~/exo_train.log

Expected: ~1-2 hours for 10K episodes at 528 steps/sec.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
import json
from collections import deque

# =====================================================================
# CONFIG
# =====================================================================
N_EPISODES = 10_000
MAX_STEPS_PER_EP = 200        # ~2 sec at 100 Hz
MAX_TORQUE = 12.0             # Nm, WAWA software limit
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
PPO_CLIP = 0.2
BATCH_SIZE = 32               # episodes per PPO update
LR = 3e-4
OUT_DIR = 'rl_output'
os.makedirs(OUT_DIR, exist_ok=True)

# Observation: hip angles (2) + hip velocities (2) + knee angles (2)
#              + prev torque (2) = 8 dims
# This matches what real WAWA IMU + encoders provide
STATE_DIM = 8
ACTION_DIM = 2


# =====================================================================
# ENVIRONMENT WRAPPER
# =====================================================================
class HipExoMyoLeg:
    """
    Wraps myoLeg walking env. Adds exo hip torque on top of
    whatever the muscles are doing.

    The muscle activations are driven by a simple sinusoidal
    pattern (crude CPG) so the model walks. The exo policy
    then learns to assist on top of that.
    """

    def __init__(self):
        from myosuite.utils import gym
        self.env = gym.make('myoLegWalk-v0')
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.sim = self.env.sim
        self.model = self.sim.model

        # Discover joint indices
        self._map_joints()

        # For CPG-based muscle drive
        self.phase = 0.0
        self.dt = self.model.opt.timestep
        self.gait_freq = 1.0  # Hz, ~1 stride/sec

        self.prev_torque = np.zeros(2)
        self.step_count = 0

        # Baseline muscle effort (measured without exo)
        self.baseline_effort = None

    def _map_joints(self):
        """Find hip and knee joint indices in MuJoCo model."""
        self.hip_r_qpos = None
        self.hip_l_qpos = None
        self.hip_r_qvel = None
        self.hip_l_qvel = None
        self.knee_r_qpos = None
        self.knee_l_qpos = None

        for i in range(self.model.njnt):
            name = self.model.joint(i).name.lower()
            if 'hip_flexion' in name and '_r' in name:
                self.hip_r_qpos = self.model.jnt_qposadr[i]
                self.hip_r_qvel = self.model.jnt_dofadr[i]
            elif 'hip_flexion' in name and '_l' in name:
                self.hip_l_qpos = self.model.jnt_qposadr[i]
                self.hip_l_qvel = self.model.jnt_dofadr[i]
            elif 'knee' in name and '_r' in name and 'angle' in name:
                self.knee_r_qpos = self.model.jnt_qposadr[i]
            elif 'knee' in name and '_l' in name and 'angle' in name:
                self.knee_l_qpos = self.model.jnt_qposadr[i]

        # Fallback: try without 'angle' for knee
        if self.knee_r_qpos is None:
            for i in range(self.model.njnt):
                name = self.model.joint(i).name.lower()
                if 'knee' in name and '_r' in name:
                    self.knee_r_qpos = self.model.jnt_qposadr[i]
                elif 'knee' in name and '_l' in name:
                    self.knee_l_qpos = self.model.jnt_qposadr[i]

        found = sum(x is not None for x in [
            self.hip_r_qpos, self.hip_l_qpos,
            self.hip_r_qvel, self.hip_l_qvel])
        print(f"  Joint mapping: {found}/4 hip indices found")

        if found < 4:
            # Print all joints for debugging
            print("  All joints:")
            for i in range(self.model.njnt):
                print(f"    {i}: {self.model.joint(i).name}")

    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.prev_torque = np.zeros(2)
        self.step_count = 0
        self.phase = 0.0
        self.sim = self.env.sim  # refresh sim reference

        return self._get_obs()

    def step(self, exo_action):
        """
        1. Generate crude muscle activations (CPG pattern)
        2. Apply exo torque to hip joints
        3. Step simulation
        4. Compute reward
        """
        exo_action = np.clip(exo_action, -MAX_TORQUE, MAX_TORQUE)

        # --- Muscle activations: simple sinusoidal CPG ---
        # This is a placeholder — a real setup would use a
        # pre-trained walking policy or motion imitation network
        n_muscles = self.env.action_space.shape[0]
        muscle_act = np.zeros(n_muscles)

        # Apply a crude alternating pattern to hip flexors/extensors
        # Just enough to keep the model somewhat upright
        self.phase += self.dt * self.gait_freq * 2 * np.pi
        for i in range(n_muscles):
            # Small baseline activation + sinusoidal modulation
            muscle_act[i] = 0.1 + 0.05 * np.sin(
                self.phase + i * 0.3)
        muscle_act = np.clip(muscle_act, 0, 1)

        # --- Apply exo torque directly to hip DOFs ---
        if self.hip_r_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_r_qvel] = exo_action[0]
        if self.hip_l_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_l_qvel] = exo_action[1]

        # --- Step the simulation ---
        result = self.env.step(muscle_act)
        obs_full = result[0]
        base_reward = result[1]
        done = result[2]

        self.step_count += 1

        # --- Compute custom reward ---
        reward = self._compute_reward(exo_action, base_reward)

        self.prev_torque = exo_action.copy()
        return self._get_obs(), reward, done

    def _get_obs(self):
        """8-dim observation matching real WAWA sensors."""
        obs = np.zeros(STATE_DIM, dtype=np.float32)
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
            obs[6] = self.prev_torque[0] / MAX_TORQUE  # normalized
            obs[7] = self.prev_torque[1] / MAX_TORQUE
        except Exception:
            pass
        return obs

    def _compute_reward(self, action, base_reward):
        """
        Reward components:
          + walking reward (from base env — forward velocity)
          - muscle effort (sum of activation^2)
          - exo energy cost (torque^2)
          - jerk penalty (torque change rate)
        """
        # Muscle effort
        act = self.sim.data.act
        if act is not None and len(act) > 0:
            muscle_effort = float(np.mean(act ** 2))
        else:
            muscle_effort = 0.0

        # Walking reward (base env rewards forward velocity)
        walk_reward = float(base_reward)

        # Exo energy cost: penalize large torques
        energy_cost = 0.002 * float(np.sum(action ** 2))

        # Smoothness: penalize torque rate of change
        jerk_cost = 0.02 * float(np.sum(
            (action - self.prev_torque) ** 2))

        reward = (walk_reward
                  - 0.5 * muscle_effort
                  - energy_cost
                  - jerk_cost)

        return reward

    def close(self):
        self.env.close()


# =====================================================================
# POLICY (same Actor-Critic MLP as the OpenSim version)
# =====================================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * -0.5)  # start conservative
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
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(-1)
        # Tanh squash → scale to torque range
        action = torch.tanh(raw_action) * MAX_TORQUE
        return action.squeeze(0).numpy(), log_prob.item(), value.item()

    def evaluate(self, states, actions):
        mean, values = self.forward(states)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)
        # Unsquash for log_prob
        raw = torch.atanh(torch.clamp(actions / MAX_TORQUE, -0.999, 0.999))
        log_probs = dist.log_prob(raw).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values.squeeze(-1), entropy


# =====================================================================
# PPO UPDATE
# =====================================================================
def ppo_update(policy, optimizer, batch):
    states = torch.FloatTensor(np.array(batch['states']))
    actions = torch.FloatTensor(np.array(batch['actions']))
    old_log_probs = torch.FloatTensor(np.array(batch['log_probs']))
    returns = torch.FloatTensor(np.array(batch['returns']))
    advantages = torch.FloatTensor(np.array(batch['advantages']))

    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8)

    total_p_loss = 0
    total_v_loss = 0

    for _ in range(PPO_EPOCHS):
        log_probs, values, entropy = policy.evaluate(states, actions)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)

        p_loss = -torch.min(ratio * advantages,
                            clipped * advantages).mean()
        v_loss = 0.5 * (returns - values).pow(2).mean()
        e_loss = -0.01 * entropy.mean()

        loss = p_loss + v_loss + e_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        total_p_loss += p_loss.item()
        total_v_loss += v_loss.item()

    return total_p_loss / PPO_EPOCHS, total_v_loss / PPO_EPOCHS


def compute_gae(rewards, values, dones):
    advantages = []
    returns = []
    gae = 0
    next_val = 0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_val = 0
            gae = 0
        delta = rewards[t] + GAMMA * next_val - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_val = values[t]
    return returns, advantages


# =====================================================================
# MAIN TRAINING LOOP
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Hip Exo RL Training — MyoSuite myoLeg")
    print(f"  Episodes:     {N_EPISODES:,}")
    print(f"  Steps/ep:     {MAX_STEPS_PER_EP}")
    print(f"  Max torque:   {MAX_TORQUE} Nm")
    print(f"  Batch size:   {BATCH_SIZE} episodes")
    print("=" * 60)

    env = HipExoMyoLeg()
    policy = ActorCritic(STATE_DIM, ACTION_DIM, hidden=128)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Logging
    reward_history = deque(maxlen=100)
    length_history = deque(maxlen=100)
    torque_history = deque(maxlen=100)
    effort_history = deque(maxlen=100)

    batch = {
        'states': [], 'actions': [], 'log_probs': [],
        'rewards': [], 'values': [], 'dones': []
    }

    best_reward = -float('inf')
    t_start = time.time()

    for ep in range(N_EPISODES):
        obs = env.reset()
        ep_reward = 0
        ep_torques = []
        ep_efforts = []

        for step in range(MAX_STEPS_PER_EP):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, done = env.step(action)

            batch['states'].append(obs)
            batch['actions'].append(action)
            batch['log_probs'].append(log_prob)
            batch['rewards'].append(reward)
            batch['values'].append(value)
            batch['dones'].append(done)

            ep_reward += reward
            ep_torques.append(np.abs(action).mean())

            # Track muscle effort
            act = env.sim.data.act
            if act is not None and len(act) > 0:
                ep_efforts.append(float(np.mean(act ** 2)))

            obs = next_obs
            if done:
                break

        reward_history.append(ep_reward)
        length_history.append(step + 1)
        torque_history.append(np.mean(ep_torques))
        if ep_efforts:
            effort_history.append(np.mean(ep_efforts))

        # PPO update
        if (ep + 1) % BATCH_SIZE == 0:
            returns, advantages = compute_gae(
                batch['rewards'], batch['values'], batch['dones'])
            batch['returns'] = returns
            batch['advantages'] = advantages
            p_loss, v_loss = ppo_update(policy, optimizer, batch)
            batch = {k: [] for k in batch}

        # Logging
        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t_start
            eps_sec = (ep + 1) / elapsed
            eta = (N_EPISODES - ep - 1) / eps_sec / 60

            avg_r = np.mean(reward_history)
            avg_len = np.mean(length_history)
            avg_torque = np.mean(torque_history)
            avg_effort = np.mean(effort_history) if effort_history else 0

            print(f"  Ep {ep+1:5d}/{N_EPISODES} | "
                  f"R={avg_r:7.1f} | "
                  f"Len={avg_len:5.1f} | "
                  f"τ={avg_torque:4.1f}Nm | "
                  f"Effort={avg_effort:.4f} | "
                  f"ETA={eta:.0f}min")

            # Save best model
            if avg_r > best_reward:
                best_reward = avg_r
                torch.save(policy.state_dict(),
                           os.path.join(OUT_DIR, 'best_exo_policy.pt'))

        # Periodic save
        if (ep + 1) % 1000 == 0:
            torch.save(policy.state_dict(),
                       os.path.join(OUT_DIR, f'exo_policy_ep{ep+1}.pt'))

    # ---- Final save ----
    total_time = time.time() - t_start
    torch.save(policy.state_dict(),
               os.path.join(OUT_DIR, 'exo_policy_final.pt'))

    print("\n" + "=" * 60)
    print(f"Training complete: {total_time/60:.1f} min")
    print("=" * 60)

    # Convert numpy types for JSON
    log_data = {
        'n_episodes': N_EPISODES,
        'total_time_sec': float(total_time),
        'final_avg_reward': float(np.mean(reward_history)),
        'final_avg_torque': float(np.mean(torque_history)),
        'final_avg_effort': float(np.mean(effort_history)) if effort_history else 0,
        'final_avg_length': float(np.mean(length_history)),
        'best_reward': float(best_reward),
    }
    with open(os.path.join(OUT_DIR, 'training_log.json'), 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"  Avg reward:  {log_data['final_avg_reward']:.1f}")
    print(f"  Avg torque:  {log_data['final_avg_torque']:.1f} Nm")
    print(f"  Avg effort:  {log_data['final_avg_effort']:.4f}")
    print(f"  Avg length:  {log_data['final_avg_length']:.0f} steps")
    print(f"  Best reward: {log_data['best_reward']:.1f}")
    print(f"\n  Saved to: {OUT_DIR}/")
    print(f"    best_exo_policy.pt")
    print(f"    exo_policy_final.pt")
    print(f"    training_log.json")