#!/usr/bin/env python3
"""
Toy RL Proof-of-Concept: PPO on OpenSim hip exoskeleton.

This is a minimal end-to-end demo showing:
  state (hip angles/velocities) → policy (MLP) → action (exo torque) → 
  simulate (OpenSim Manager) → reward (negative muscle effort) → PPO update

NOT expected to learn a good policy with only 1K episodes.
The point is to prove the pipeline works before migrating to MuJoCo.

Run on cortex:
    conda activate opensim
    pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages
    cd ~/repos/projects/exo-assist-pipeline/opensim/moco-experiments/
    python3 toy_rl_opensim.py 2>&1 | tee ~/toy_rl.log

Expected runtime: ~6-8 hours for 1K episodes on cortex.
"""
import opensim as osim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
import json

# ==========================================================================
# CONFIG
# ==========================================================================
MODEL_PATH = 'multisubject/P010/mocotrack_output/P010_reduced.osim'
OUT_DIR = 'rl_output'
os.makedirs(OUT_DIR, exist_ok=True)

N_EPISODES = 1000
STEPS_PER_EPISODE = 100       # 1 sec at 100 Hz
DT = 0.01                     # 100 Hz control loop
MAX_TORQUE = 12.0             # Nm, software limit from Navid
GAMMA = 0.99                  # discount factor
GAE_LAMBDA = 0.95             # GAE parameter
PPO_EPOCHS = 4                # PPO update epochs per batch
PPO_CLIP = 0.2                # PPO clipping parameter
BATCH_SIZE = 20               # episodes per PPO update
LR = 3e-4                     # learning rate
INTEGRATOR_ACCURACY = 1e-2    # looser for speed (was 1e-3)

# State: hip_angle_r, hip_angle_l, hip_vel_r, hip_vel_l,
#        knee_angle_r, knee_angle_l, prev_torque_r, prev_torque_l
STATE_DIM = 8
# Action: exo_torque_r, exo_torque_l
ACTION_DIM = 2

# Hip muscles to compute effort reward
HIP_MUSCLES = [
    'glmax1_r', 'glmax2_r', 'glmax3_r',
    'psoas_r', 'iliacus_r', 'recfem_r',
    'bflh_r', 'semimem_r', 'semiten_r',
    'glmax1_l', 'glmax2_l', 'glmax3_l',
    'psoas_l', 'iliacus_l', 'recfem_l',
    'bflh_l', 'semimem_l', 'semiten_l',
]


# ==========================================================================
# OPENSIM ENVIRONMENT
# ==========================================================================
class ExoEnv:
    """Minimal OpenSim environment for hip exoskeleton RL."""

    def __init__(self, model_path):
        self.model = osim.Model(model_path)

        # Add exo actuators
        for side in ['r', 'l']:
            act = osim.CoordinateActuator(f'hip_flexion_{side}')
            act.setName(f'exo_hip_{side}')
            act.setOptimalForce(1.0)
            act.setMinControl(-MAX_TORQUE)
            act.setMaxControl(MAX_TORQUE)
            self.model.addForce(act)

        self.model.finalizeConnections()
        self.state = self.model.initSystem()

        # Cache coordinate references
        cs = self.model.getCoordinateSet()
        self.hip_r = cs.get('hip_flexion_r')
        self.hip_l = cs.get('hip_flexion_l')
        self.knee_r = cs.get('knee_angle_r')
        self.knee_l = cs.get('knee_angle_l')

        # Cache muscle references
        self.muscles = []
        ms = self.model.getMuscles()
        for name in HIP_MUSCLES:
            try:
                m = ms.get(name)
                self.muscles.append(m)
            except:
                pass  # muscle not in reduced model

        # Cache actuator references
        fs = self.model.getForceSet()
        self.exo_r = osim.CoordinateActuator.safeDownCast(fs.get('exo_hip_r'))
        self.exo_l = osim.CoordinateActuator.safeDownCast(fs.get('exo_hip_l'))

        self.prev_torque = np.zeros(2)
        self.manager = None

    def reset(self):
        """Reset to a slightly randomized standing pose."""
        self.state = self.model.initSystem()

        # Small random perturbation around neutral standing
        noise = np.random.uniform(-0.05, 0.05, 4)
        self.hip_r.setValue(self.state, 0.1 + noise[0])
        self.hip_l.setValue(self.state, -0.05 + noise[1])
        self.knee_r.setValue(self.state, -0.2 + noise[2])
        self.knee_l.setValue(self.state, -0.1 + noise[3])

        self.model.realizeVelocity(self.state)

        self.manager = osim.Manager(self.model)
        self.manager.setIntegratorAccuracy(INTEGRATOR_ACCURACY)
        self.manager.initialize(self.state)

        self.prev_torque = np.zeros(2)
        self.step_count = 0

        return self._get_obs()

    def step(self, action):
        """
        Apply exo torque and step simulation forward by dt.
        action: [torque_r, torque_l] in [-MAX_TORQUE, MAX_TORQUE]
        """
        # Clamp action
        action = np.clip(action, -MAX_TORQUE, MAX_TORQUE)

        # Set exo torque via override
        self.exo_r.setOverrideActuation(self.state, float(action[0]))
        self.exo_l.setOverrideActuation(self.state, float(action[1]))

        # Integrate one step
        self.step_count += 1
        t_target = self.step_count * DT
        done = False

        try:
            self.state = self.manager.integrate(t_target)
        except Exception:
            # Simulation crashed (model fell, etc.)
            done = True
            return self._get_obs(), -10.0, done

        # Compute reward
        reward = self._compute_reward(action)

        # Check if model is in a bad state (fell over, etc.)
        pelvis_y = self.model.getBodySet().get('pelvis').getPositionInGround(
            self.state).get(1)
        if pelvis_y < 0.3:  # pelvis too low = fell
            done = True
            reward = -10.0

        self.prev_torque = action.copy()
        return self._get_obs(), reward, done

    def _get_obs(self):
        """Extract state vector from OpenSim."""
        try:
            obs = np.array([
                self.hip_r.getValue(self.state),
                self.hip_l.getValue(self.state),
                self.hip_r.getSpeedValue(self.state),
                self.hip_l.getSpeedValue(self.state),
                self.knee_r.getValue(self.state),
                self.knee_l.getValue(self.state),
                self.prev_torque[0],
                self.prev_torque[1],
            ], dtype=np.float32)
        except Exception:
            obs = np.zeros(STATE_DIM, dtype=np.float32)
        return obs

    def _compute_reward(self, action):
        """
        Reward = negative muscle effort + smoothness bonus + energy penalty.
        This is simplified — real version would weight muscles properly.
        """
        # Muscle effort: sum of activation^2
        try:
            self.model.realizeAcceleration(self.state)
            muscle_effort = 0.0
            for m in self.muscles:
                a = m.getActivation(self.state)
                muscle_effort += a * a
            muscle_effort /= max(len(self.muscles), 1)
        except Exception:
            muscle_effort = 1.0  # worst case

        # Smoothness penalty (penalize jerky torque changes)
        torque_change = np.sum((action - self.prev_torque) ** 2)
        smoothness_penalty = 0.01 * torque_change

        # Energy penalty (don't waste motor energy)
        energy_penalty = 0.001 * np.sum(action ** 2)

        reward = -muscle_effort - smoothness_penalty - energy_penalty
        return float(reward)


# ==========================================================================
# POLICY NETWORK (Actor-Critic)
# ==========================================================================
class PolicyNetwork(nn.Module):
    """Simple MLP actor-critic for PPO."""

    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Actor head (mean of Gaussian policy)
        self.actor_mean = nn.Linear(hidden, action_dim)
        # Log std as learnable parameter
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action(self, state):
        """Sample action from policy."""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            mean, value = self.forward(state_t)
            std = torch.exp(self.actor_log_std)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)

        # Scale action to torque range
        action_np = action.squeeze(0).numpy() * MAX_TORQUE
        return action_np, log_prob.item(), value.item()

    def evaluate(self, states, actions):
        """Evaluate actions for PPO update."""
        mean, values = self.forward(states)
        std = torch.exp(self.actor_log_std)
        dist = Normal(mean, std)

        # Unscale actions back
        actions_unscaled = actions / MAX_TORQUE
        log_probs = dist.log_prob(actions_unscaled).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_probs, values.squeeze(-1), entropy


# ==========================================================================
# PPO UPDATE
# ==========================================================================
def ppo_update(policy, optimizer, states, actions, old_log_probs,
               returns, advantages):
    """Standard PPO clipped objective update."""
    states_t = torch.FloatTensor(np.array(states))
    actions_t = torch.FloatTensor(np.array(actions))
    old_log_probs_t = torch.FloatTensor(np.array(old_log_probs))
    returns_t = torch.FloatTensor(np.array(returns))
    advantages_t = torch.FloatTensor(np.array(advantages))

    # Normalize advantages
    if len(advantages_t) > 1:
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        log_probs, values, entropy = policy.evaluate(states_t, actions_t)

        # PPO clipped objective
        ratio = torch.exp(log_probs - old_log_probs_t)
        clipped = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP)
        policy_loss = -torch.min(ratio * advantages_t,
                                 clipped * advantages_t).mean()

        # Value loss
        value_loss = 0.5 * (returns_t - values).pow(2).mean()

        # Entropy bonus (encourage exploration)
        entropy_loss = -0.01 * entropy.mean()

        loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

    return policy_loss.item(), value_loss.item()


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=GAE_LAMBDA):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    returns = []
    gae = 0
    next_value = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            next_value = 0
            gae = 0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])
        next_value = values[t]

    return returns, advantages


# ==========================================================================
# MAIN TRAINING LOOP
# ==========================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Toy RL: PPO on OpenSim Hip Exoskeleton")
    print(f"  Episodes:    {N_EPISODES}")
    print(f"  Steps/ep:    {STEPS_PER_EPISODE}")
    print(f"  Max torque:  {MAX_TORQUE} Nm")
    print(f"  State dim:   {STATE_DIM}")
    print(f"  Action dim:  {ACTION_DIM}")
    print("=" * 60)

    # Check if actuator override is available
    env = ExoEnv(MODEL_PATH)
    test_state = env.reset()

    # Verify override works
    try:
        env.exo_r.overrideActuation(env.state, True)
        env.exo_l.overrideActuation(env.state, True)
        print("  Actuator override: OK")
    except Exception as e:
        print(f"  WARNING: Override failed ({e})")
        print("  Falling back to setOverrideActuation only")

    policy = PolicyNetwork(STATE_DIM, ACTION_DIM, hidden=64)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    # Training log
    log = {
        'episode_rewards': [],
        'episode_lengths': [],
        'policy_losses': [],
        'value_losses': [],
        'mean_torques': [],
    }

    # Batch buffers
    batch_states = []
    batch_actions = []
    batch_log_probs = []
    batch_rewards = []
    batch_values = []
    batch_dones = []

    t_start = time.time()

    for ep in range(N_EPISODES):
        obs = env.reset()

        # Enable actuator override for this episode
        try:
            env.exo_r.overrideActuation(env.state, True)
            env.exo_l.overrideActuation(env.state, True)
        except:
            pass

        ep_reward = 0
        ep_torques = []

        for step in range(STEPS_PER_EPISODE):
            action, log_prob, value = policy.get_action(obs)
            next_obs, reward, done = env.step(action)

            batch_states.append(obs)
            batch_actions.append(action)
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_values.append(value)
            batch_dones.append(done)

            ep_reward += reward
            ep_torques.append(np.abs(action).mean())
            obs = next_obs

            if done:
                break

        ep_length = step + 1
        log['episode_rewards'].append(ep_reward)
        log['episode_lengths'].append(ep_length)
        log['mean_torques'].append(np.mean(ep_torques))

        # PPO update every BATCH_SIZE episodes
        if (ep + 1) % BATCH_SIZE == 0:
            returns, advantages = compute_gae(
                batch_rewards, batch_values, batch_dones)
            p_loss, v_loss = ppo_update(
                policy, optimizer,
                batch_states, batch_actions, batch_log_probs,
                returns, advantages)

            log['policy_losses'].append(p_loss)
            log['value_losses'].append(v_loss)

            # Clear batch
            batch_states = []
            batch_actions = []
            batch_log_probs = []
            batch_rewards = []
            batch_values = []
            batch_dones = []

        # Print progress
        if (ep + 1) % 10 == 0:
            elapsed = time.time() - t_start
            eps_per_sec = (ep + 1) / elapsed
            eta_hours = (N_EPISODES - ep - 1) / eps_per_sec / 3600

            recent_rewards = log['episode_rewards'][-10:]
            recent_lengths = log['episode_lengths'][-10:]
            recent_torques = log['mean_torques'][-10:]

            print(f"  Ep {ep+1:4d}/{N_EPISODES} | "
                  f"R={np.mean(recent_rewards):7.2f} | "
                  f"Len={np.mean(recent_lengths):5.1f} | "
                  f"Torque={np.mean(recent_torques):5.1f} Nm | "
                  f"ETA={eta_hours:.1f}h")

    # ---- Save results ----
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Training complete in {total_time/3600:.1f} hours")
    print("=" * 60)

    # Save policy
    torch.save(policy.state_dict(), os.path.join(OUT_DIR, 'toy_policy.pt'))
    print(f"  Policy saved: {OUT_DIR}/toy_policy.pt")

    # Save log
    with open(os.path.join(OUT_DIR, 'toy_rl_log.json'), 'w') as f:
        json.dump(log, f)
    print(f"  Log saved: {OUT_DIR}/toy_rl_log.json")

    # Print summary
    first_100 = log['episode_rewards'][:100]
    last_100 = log['episode_rewards'][-100:]
    print(f"\n  First 100 ep avg reward: {np.mean(first_100):.2f}")
    print(f"  Last  100 ep avg reward: {np.mean(last_100):.2f}")
    print(f"  Improvement: {np.mean(last_100) - np.mean(first_100):.2f}")

    first_torques = log['mean_torques'][:100]
    last_torques = log['mean_torques'][-100:]
    print(f"  First 100 ep avg torque: {np.mean(first_torques):.1f} Nm")
    print(f"  Last  100 ep avg torque: {np.mean(last_torques):.1f} Nm")

    print(f"\n  Total wall time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Episodes/sec:    {N_EPISODES/total_time:.3f}")
    print("\nDone. Show this to Yang!")