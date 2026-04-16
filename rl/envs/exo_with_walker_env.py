import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.baselines.load_deprl_reference import load_deprl_reference, wrap_deprl_env


class ExoWithWalkerSB3(gym.Env):
    """
    Stage-1 bootstrap environment:
    - frozen local DEP-RL walker supplies nominal locomotion
    - exo policy adds 2 hip torques
    - reward favors walking quality + lower effort proxy + smooth torque

    This is NOT the final co-adaptive architecture.
    It is a practical first assistance-training setup.
    """

    metadata = {"render_modes": []}
    MAX_TORQUE = 12.0
    TORQUE_SCALE = 0.25  # start small; curriculum can increase later

    def __init__(self, walker_path=None, max_steps=300):
        super().__init__()

        from myosuite.utils import gym as myogym

        self.base_env = myogym.make("myoLegWalk-v0")
        self.env = wrap_deprl_env(self.base_env)
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        self.walker = load_deprl_reference(self.env, walker_path)
        self.sim = self.base_env.unwrapped.sim
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

    @staticmethod
    def _normalize_step(result):
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
            return obs, float(reward), done, info

        if len(result) == 4:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), info

        raise ValueError(f"Unexpected step result length: {len(result)}")

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
            x is not None
            for x in [
                self.hip_r_qpos,
                self.hip_l_qpos,
                self.hip_r_qvel,
                self.hip_l_qvel,
            ]
        )
        print(f"  Joint mapping: {found}/4 hip indices found")

    def _map_root(self):
        self.root_x_qvel = 0
        self.torso_pitch_qpos = None
        self.pelvis_height_qpos = None

        # try to find useful torso/root joints
        for i in range(self.model_mj.njnt):
            name = self.model_mj.joint(i).name.lower()
            if self.torso_pitch_qpos is None and (
                "pelvis_tilt" in name or "torso" in name
            ):
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
                obs, reward, done, info = self._normalize_step(self.env.step(action))
                del reward, info

                act = self.base_env.unwrapped.sim.data.act
                if act is not None and len(act) > 0:
                    ep_efforts.append(float(np.mean(act**2)))
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

        phi_r = np.arctan2(hipd_r, hip_r)  # [-pi, pi]
        phi_l = phi_r + np.pi

        return (
            np.sin(phi_r),
            np.cos(phi_r),
            np.sin(phi_l),
            np.cos(phi_l),
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
        effort_reduction = (self.baseline_effort - current_effort) / max(
            self.baseline_effort, 1e-6
        )
        effort_bonus = 3.0 * effort_reduction

        # keep existing walking quality from MyoSuite
        walk_reward = float(base_reward)

        # compute penalties on effective (scaled) torque
        eff_action = self.TORQUE_SCALE * action
        eff_prev = self.TORQUE_SCALE * self.prev_torque
        energy_penalty = 0.003 * float(np.sum(eff_action**2))

        # encourage smooth torques
        jerk_penalty = 0.01 * float(np.sum((eff_action - eff_prev) ** 2))

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
        self.sim = self.base_env.unwrapped.sim
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

        # scale exo torques (small-authority curriculum)
        tau_r = float(self.TORQUE_SCALE * action[0])
        tau_l = float(self.TORQUE_SCALE * action[1])
        if self.hip_r_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_r_qvel] = tau_r
        if self.hip_l_qvel is not None:
            self.sim.data.qfrc_applied[self.hip_l_qvel] = tau_l

        # walker chooses muscle action from the current last simulator state
        muscle_action, _ = self.walker.predict(self.walker_obs, deterministic=True)

        self.walker_obs, base_reward, done, env_info = self._normalize_step(
            self.env.step(muscle_action)
        )
        del env_info

        self.step_count += 1

        act = self.sim.data.act
        if act is not None and len(act) > 0:
            current_effort = float(np.mean(act**2))
        else:
            current_effort = self.baseline_effort

        reward = self._compute_reward(action, base_reward, current_effort)

        # strong penalty if we terminate early (protect the walker)
        if done or self._terminated()[0]:
            # penalize earlier failures more
            early_frac = 1.0 - (self.step_count / float(self.max_steps))
            reward -= 5.0 * max(0.0, early_frac)

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
            "effort_reduction_pct": 100.0
            * (
                (self.baseline_effort - current_effort)
                / max(self.baseline_effort, 1e-6)
            ),
            "mean_abs_torque": float(np.mean(np.abs(self.TORQUE_SCALE * action))),
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
                f"|tau|={float(np.mean(np.abs(self.TORQUE_SCALE * action))):.3f} "
                f"torso={float(custom_snapshot['torso_pitch']):.3f} "
                f"pelvis_h={float(custom_snapshot['pelvis_height']):.3f}"
            )

        self.prev_torque = action.copy()
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.env.close()
