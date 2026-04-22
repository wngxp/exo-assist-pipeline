import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl.mocap_study.envs.mocap_reference import MocapReference
from rl.mocap_study.envs.reward_tracking import (
    build_tracking_indices,
    compute_tracking_reward,
)


class MocapWalkerEnv(gym.Env):
    """
    Walker-only mocap tracking environment.

    The policy directly controls the MyoSuite walker muscles and is rewarded for
    tracking the mocap reference while maintaining the base walking objective.
    """

    metadata = {"render_modes": []}
    ENV_VERSION = "mocap_stage1_walker_v1"

    def __init__(
        self,
        max_steps=300,
        reference_path="rl/mocap_study/output/reference/trial0_normalized_cycles_with_phase.csv",
        cycle_id=0,
    ):
        super().__init__()

        from myosuite.utils import gym as myogym

        self.base_env = myogym.make("myoLegWalk-v0")
        self.env = self.base_env
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            base_obs = reset_out[0]
        else:
            base_obs = reset_out

        self.sim = self.base_env.unwrapped.sim
        self.model_mj = self.sim.model

        self.max_steps = max_steps
        self.step_count = 0

        self.reference = MocapReference(reference_path, cycle_id=cycle_id)
        self.track_idx = build_tracking_indices(self.reference.pos_cols)
        self.phase = 0.0
        self.gait_period = 1.0

        self._map_joints()
        self._map_root()

        self.base_obs_dim = int(np.asarray(base_obs, dtype=np.float32).shape[0])
        self.extra_obs_dim = 15
        self.last_base_obs = np.asarray(base_obs, dtype=np.float32)

        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_obs_dim + self.extra_obs_dim,),
            dtype=np.float32,
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
        self.ankle_r_qpos = None
        self.ankle_l_qpos = None
        self.knee_r_qvel = None
        self.knee_l_qvel = None
        self.ankle_r_qvel = None
        self.ankle_l_qvel = None

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
                self.knee_r_qvel = self.model_mj.jnt_dofadr[i]
            elif "knee" in name and "_l" in name:
                self.knee_l_qpos = self.model_mj.jnt_qposadr[i]
                self.knee_l_qvel = self.model_mj.jnt_dofadr[i]
            elif "ankle" in name and "_r" in name:
                self.ankle_r_qpos = self.model_mj.jnt_qposadr[i]
                self.ankle_r_qvel = self.model_mj.jnt_dofadr[i]
            elif "ankle" in name and "_l" in name:
                self.ankle_l_qpos = self.model_mj.jnt_qposadr[i]
                self.ankle_l_qvel = self.model_mj.jnt_dofadr[i]

    def _map_root(self):
        self.root_x_qvel = 0
        self.torso_pitch_qpos = None
        self.pelvis_height_qpos = None

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

    def _estimate_phase(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        hip_r = qpos[self.hip_r_qpos] if self.hip_r_qpos is not None else 0.0
        hipd_r = qvel[self.hip_r_qvel] if self.hip_r_qvel is not None else 0.0

        phi_r = np.arctan2(hipd_r, hip_r)
        phi_l = phi_r + np.pi

        return (
            np.sin(phi_r),
            np.cos(phi_r),
            np.sin(phi_l),
            np.cos(phi_l),
        )

    def _get_current_q_dq_for_reference(self):
        q = np.zeros(len(self.reference.pos_cols), dtype=np.float32)
        dq = np.zeros(len(self.reference.vel_cols), dtype=np.float32)

        joint_map = {
            "hip_flexion_r": (self.hip_r_qpos, self.hip_r_qvel),
            "knee_angle_r": (self.knee_r_qpos, self.knee_r_qvel),
            "ankle_angle_r": (self.ankle_r_qpos, self.ankle_r_qvel),
            "hip_flexion_l": (self.hip_l_qpos, self.hip_l_qvel),
            "knee_angle_l": (self.knee_l_qpos, self.knee_l_qvel),
            "ankle_angle_l": (self.ankle_l_qpos, self.ankle_l_qvel),
        }

        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        for i, col in enumerate(self.reference.pos_cols):
            joint_name = col.replace("pos_", "")
            if joint_name in joint_map:
                qpos_idx, _ = joint_map[joint_name]
                if qpos_idx is not None:
                    q[i] = qpos[qpos_idx]

        for i, col in enumerate(self.reference.vel_cols):
            joint_name = col.replace("vel_", "")
            if joint_name in joint_map:
                _, qvel_idx = joint_map[joint_name]
                if qvel_idx is not None:
                    dq[i] = qvel[qvel_idx]

        return q, dq

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        extra = np.zeros(self.extra_obs_dim, dtype=np.float32)

        sin_phi_r, cos_phi_r, sin_phi_l, cos_phi_l = self._estimate_phase()
        extra[0] = sin_phi_r
        extra[1] = cos_phi_r
        extra[2] = sin_phi_l
        extra[3] = cos_phi_l

        if self.hip_r_qpos is not None:
            extra[4] = qpos[self.hip_r_qpos]
        if self.hip_l_qpos is not None:
            extra[5] = qpos[self.hip_l_qpos]
        if self.knee_r_qpos is not None:
            extra[6] = qpos[self.knee_r_qpos]
        if self.knee_l_qpos is not None:
            extra[7] = qpos[self.knee_l_qpos]
        if self.ankle_r_qpos is not None:
            extra[8] = qpos[self.ankle_r_qpos]
        if self.ankle_l_qpos is not None:
            extra[9] = qpos[self.ankle_l_qpos]

        if self.hip_r_qvel is not None:
            extra[10] = qvel[self.hip_r_qvel]
        if self.hip_l_qvel is not None:
            extra[11] = qvel[self.hip_l_qvel]
        if self.root_x_qvel is not None and self.root_x_qvel < len(qvel):
            extra[12] = qvel[self.root_x_qvel]
        if self.torso_pitch_qpos is not None:
            extra[13] = qpos[self.torso_pitch_qpos]
        if self.pelvis_height_qpos is not None:
            extra[14] = qpos[self.pelvis_height_qpos]

        return np.concatenate(
            [self.last_base_obs.astype(np.float32, copy=False), extra], axis=0
        )

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

        return reason is not None, reason, {
            "torso_pitch": torso_pitch,
            "pelvis_height": pelvis_h,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del options

        try:
            reset_out = self.env.reset(seed=seed)
        except TypeError:
            reset_out = self.env.reset()

        if isinstance(reset_out, tuple):
            base_obs = reset_out[0]
        else:
            base_obs = reset_out

        self.sim = self.base_env.unwrapped.sim
        self.step_count = 0
        self.phase = 0.0

        self.last_base_obs = np.asarray(base_obs, dtype=np.float32)

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(
            np.float32
        )

        base_obs, base_reward, env_done, _ = self._normalize_step(self.env.step(action))
        self.last_base_obs = np.asarray(base_obs, dtype=np.float32)

        self.step_count += 1

        dt = float(self.model_mj.opt.timestep)
        self.phase = (self.phase + dt / self.gait_period) % 1.0

        ref = self.reference.get(self.phase)
        q_cur, dq_cur = self._get_current_q_dq_for_reference()
        track_terms = compute_tracking_reward(
            q=q_cur,
            dq=dq_cur,
            q_ref=ref["pos"],
            dq_ref=ref["vel"],
            track_idx=self.track_idx,
        )

        ctrl_cost = 0.001 * float(np.sum(action**2))
        reward = 2.0 * track_terms["r_track"] + float(base_reward) - ctrl_cost

        custom_terminated, custom_reason, custom_snapshot = self._terminated()
        terminated = env_done or custom_terminated
        truncated = self.step_count >= self.max_steps

        if truncated and not terminated:
            termination_source = "time_limit"
        elif env_done and custom_terminated:
            termination_source = "env+custom"
        elif env_done:
            termination_source = "env_done"
        elif custom_terminated:
            termination_source = "custom"
        else:
            termination_source = None

        info = {
            "env_version": self.ENV_VERSION,
            "phase": float(self.phase),
            "ref_index": int(ref["index"]),
            "base_reward": float(base_reward),
            "ctrl_cost": ctrl_cost,
            "r_pos_track": float(track_terms["r_pos"]),
            "r_vel_track": float(track_terms["r_vel"]),
            "r_track": float(track_terms["r_track"]),
            "q_err_norm": float(track_terms["q_err_norm"]),
            "dq_err_norm": float(track_terms["dq_err_norm"]),
            "env_done": bool(env_done),
            "custom_terminated": bool(custom_terminated),
            "termination_source": termination_source,
            "termination_reason": custom_reason,
            "torso_pitch": float(custom_snapshot["torso_pitch"]),
            "pelvis_height": float(custom_snapshot["pelvis_height"]),
            "step_count": int(self.step_count),
        }

        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        self.env.close()
