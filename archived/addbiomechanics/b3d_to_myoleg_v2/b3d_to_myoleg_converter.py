import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Optional, Tuple


class B3DToMyoLegConverter:
    """
    将 AddBiomechanics b3d 数据（基于 Rajagopal OpenSim 模型）转换为 MyoSuite MyoLeg 模型格式。

    适配任意 DoF 数量（23 DoF 下肢版 / 37 DoF 完整版），通过名称匹配动态建立关节映射。
    """

    # MyoLeg qpos 索引 (共 35 维)
    MYOLEG_QPOS_INDICES = {
        'root_pos': slice(0, 3),
        'root_quat': slice(3, 7),
        'hip_flexion_r': 7,
        'hip_adduction_r': 8,
        'hip_rotation_r': 9,
        'knee_angle_r_translation2': 10,
        'knee_angle_r_translation1': 11,
        'knee_angle_r': 12,
        'knee_angle_r_rotation2': 13,
        'knee_angle_r_rotation3': 14,
        'ankle_angle_r': 15,
        'subtalar_angle_r': 16,
        'mtp_angle_r': 17,
        'knee_angle_r_beta_translation2': 18,
        'knee_angle_r_beta_translation1': 19,
        'knee_angle_r_beta_rotation1': 20,
        'hip_flexion_l': 21,
        'hip_adduction_l': 22,
        'hip_rotation_l': 23,
        'knee_angle_l_translation2': 24,
        'knee_angle_l_translation1': 25,
        'knee_angle_l': 26,
        'knee_angle_l_rotation2': 27,
        'knee_angle_l_rotation3': 28,
        'ankle_angle_l': 29,
        'subtalar_angle_l': 30,
        'mtp_angle_l': 31,
        'knee_angle_l_beta_translation2': 32,
        'knee_angle_l_beta_translation1': 33,
        'knee_angle_l_beta_rotation1': 34,
    }

    # MyoLeg qvel 索引 (共 34 维)
    MYOLEG_QVEL_INDICES = {
        'root_angvel': slice(0, 3),
        'root_linvel': slice(3, 6),
        'hip_flexion_r': 6,
        'hip_adduction_r': 7,
        'hip_rotation_r': 8,
        'knee_angle_r_translation2': 9,
        'knee_angle_r_translation1': 10,
        'knee_angle_r': 11,
        'knee_angle_r_rotation2': 12,
        'knee_angle_r_rotation3': 13,
        'ankle_angle_r': 14,
        'subtalar_angle_r': 15,
        'mtp_angle_r': 16,
        'knee_angle_r_beta_translation2': 17,
        'knee_angle_r_beta_translation1': 18,
        'knee_angle_r_beta_rotation1': 19,
        'hip_flexion_l': 20,
        'hip_adduction_l': 21,
        'hip_rotation_l': 22,
        'knee_angle_l_translation2': 23,
        'knee_angle_l_translation1': 24,
        'knee_angle_l': 25,
        'knee_angle_l_rotation2': 26,
        'knee_angle_l_rotation3': 27,
        'ankle_angle_l': 28,
        'subtalar_angle_l': 29,
        'mtp_angle_l': 30,
        'knee_angle_l_beta_translation2': 31,
        'knee_angle_l_beta_translation1': 32,
        'knee_angle_l_beta_rotation1': 33,
    }

    # 定义 nimble 关节名称 -> MyoLeg 关节名称的映射
    # 支持别名：如果 nimble 中有多种可能的命名，按顺序尝试第一个匹配的
    JOINT_NAME_ALIASES = {
        # pelvis 旋转
        'pelvis_tilt': ['pelvis_tilt', 'pelvis_rx'],
        'pelvis_list': ['pelvis_list', 'pelvis_ry'],
        'pelvis_rotation': ['pelvis_rotation', 'pelvis_rz'],
        # pelvis 平移
        'pelvis_tx': ['pelvis_tx', 'pelvis_x'],
        'pelvis_ty': ['pelvis_ty', 'pelvis_y'],
        'pelvis_tz': ['pelvis_tz', 'pelvis_z'],
        # 右髋
        'hip_flexion_r': ['hip_flexion_r', 'hip_flex_r', 'r_hip_flex'],
        'hip_adduction_r': ['hip_adduction_r', 'hip_add_r', 'r_hip_add'],
        'hip_rotation_r': ['hip_rotation_r', 'hip_rot_r', 'r_hip_rot'],
        # 右膝
        'knee_angle_r': ['knee_angle_r', 'knee_flex_r', 'r_knee_flex', 'r_knee_angle'],
        # 右踝
        'ankle_angle_r': ['ankle_angle_r', 'ankle_flex_r', 'r_ankle_flex'],
        'subtalar_angle_r': ['subtalar_angle_r', 'subtalar_r', 'r_subtalar'],
        'mtp_angle_r': ['mtp_angle_r', 'mtp_r', 'r_mtp'],
        # 左髋
        'hip_flexion_l': ['hip_flexion_l', 'hip_flex_l', 'l_hip_flex'],
        'hip_adduction_l': ['hip_adduction_l', 'hip_add_l', 'l_hip_add'],
        'hip_rotation_l': ['hip_rotation_l', 'hip_rot_l', 'l_hip_rot'],
        # 左膝
        'knee_angle_l': ['knee_angle_l', 'knee_flex_l', 'l_knee_flex', 'l_knee_angle'],
        # 左踝
        'ankle_angle_l': ['ankle_angle_l', 'ankle_flex_l', 'l_ankle_flex'],
        'subtalar_angle_l': ['subtalar_angle_l', 'subtalar_l', 'l_subtalar'],
        'mtp_angle_l': ['mtp_angle_l', 'mtp_l', 'l_mtp'],
    }

    def __init__(self,
                 pelvis_pos_scale: float = 1.0,
                 pelvis_pos_offset: np.ndarray = None,
                 coordinate_swap: str = 'xyz_to_xzy'):
        """
        Args:
            pelvis_pos_scale: pelvis 位置缩放因子
            pelvis_pos_offset: pelvis 位置偏移 (3,)
            coordinate_swap: 坐标轴交换策略
                'xyz_to_xzy': [tx, ty, tz] -> [tx, tz, ty]
                'xyz_to_xz_neg_y': [tx, ty, tz] -> [tx, tz, -ty]
                'identity': 不交换
        """
        self.pelvis_pos_scale = pelvis_pos_scale
        self.pelvis_pos_offset = (np.zeros(3) if pelvis_pos_offset is None
                                  else np.array(pelvis_pos_offset))
        self.coordinate_swap = coordinate_swap

    def _resolve_joint_name(self, target_name: str, available_names: List[str]) -> Optional[str]:
        """
        通过别名表在 available_names 中查找匹配的关节名
        """
        aliases = self.JOINT_NAME_ALIASES.get(target_name, [target_name])
        for alias in aliases:
            if alias in available_names:
                return alias
        return None

    def _build_joint_mapping(self, nimble_dof_names: List[str]) -> Dict[str, int]:
        """
        根据 nimble 中实际存在的 DoF 名称，建立 MyoLeg 需要的关节索引映射

        Returns:
            nimble_name -> nimble_idx 的字典，只包含 MyoLeg 需要的关节
        """
        nimble_set = set(nimble_dof_names)
        mapping = {}
        missing = []

        required_joints = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r',
            'knee_angle_r',
            'ankle_angle_r', 'subtalar_angle_r', 'mtp_angle_r',
            'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l',
            'knee_angle_l',
            'ankle_angle_l', 'subtalar_angle_l', 'mtp_angle_l',
        ]

        for joint in required_joints:
            matched = self._resolve_joint_name(joint, nimble_set)
            if matched:
                mapping[joint] = nimble_dof_names.index(matched)
            else:
                missing.append(joint)

        if missing:
            print(f"[Converter] 警告: 以下关节在 b3d 中未找到，将使用默认值 0: {missing}")

        return mapping

    def _swap_coordinates(self, vec: np.ndarray) -> np.ndarray:
        """坐标轴交换: OpenSim Y-up -> MuJoCo Z-up"""
        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
        if self.coordinate_swap == 'xyz_to_xzy':
            return np.stack([x, z, y], axis=-1)
        elif self.coordinate_swap == 'xyz_to_xz_neg_y':
            return np.stack([x, z, -y], axis=-1)
        elif self.coordinate_swap == 'identity':
            return vec
        else:
            raise ValueError(f"未知 coordinate_swap: {self.coordinate_swap}")

    def _euler_zxy_to_rotation_matrix(self, tilt: float, list_: float, rotation: float) -> np.ndarray:
        """
        nimble Rajagopal pelvis Euler 角 (ZXY order) -> 旋转矩阵

        tilt (pelvis_tilt) -> Z, list (pelvis_list) -> X, rotation (pelvis_rotation) -> Y
        R = R_y(rotation) @ R_x(list) @ R_z(tilt)
        """
        r = R.from_euler('ZXY', [tilt, list_, rotation], degrees=False)
        return r.as_matrix()

    def _rotation_opensim_to_mujoco(self, R_os: np.ndarray) -> np.ndarray:
        """
        OpenSim Y-up -> MuJoCo Z-up 旋转矩阵转换

        1. 基变换 T 交换 Y 和 Z: R_intermediate = T @ R_os @ T
        2. MyoLeg 模型固定朝向偏移 R_z(-90°): R_mj = R_z(-90°) @ R_intermediate
        """
        T = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ], dtype=np.float64)

        R_intermediate = T @ R_os @ T

        c, s = np.cos(-np.pi / 2), np.sin(-np.pi / 2)
        R_offset = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float64)

        R_mj = R_offset @ R_intermediate
        return R_mj

    def convert_single_frame(self, nimble_pos: np.ndarray,
                             nimble_vel: Optional[np.ndarray] = None,
                             nimble_dof_names: Optional[List[str]] = None,
                             nimble_dof_map: Optional[Dict[str, int]] = None
                             ) -> Dict[str, np.ndarray]:
        """
        将单帧 nimble pos/vel 转换为 myoleg qpos/qvel

        Args:
            nimble_pos: (N_dof,) 或 (N, N_dof) nimble 骨架 DoF 位置
            nimble_vel: (N_dof,) 或 (N, N_dof) nimble 骨架 DoF 速度
            nimble_dof_names: nimble DoF 名称列表，首次调用必须提供
            nimble_dof_map: nimble名称到索引的映射，可由 dof_names 构建

        Returns:
            dict with 'qpos': (35,) 或 (N, 35), 'qvel': (34,) 或 (N, 34)
        """
        single = (nimble_pos.ndim == 1)
        if single:
            nimble_pos = nimble_pos[np.newaxis, :]
            if nimble_vel is not None:
                nimble_vel = nimble_vel[np.newaxis, :]

        # 首次调用时建立映射
        if nimble_dof_names is not None:
            self._nimble_dof_names = nimble_dof_names
            self._nimble_map = self._build_joint_mapping(nimble_dof_names)
        elif not hasattr(self, '_nimble_map'):
            raise ValueError("首次调用必须提供 nimble_dof_names")

        n_frames = nimble_pos.shape[0]
        qpos = np.zeros((n_frames, 35))
        qvel = np.zeros((n_frames, 34)) if nimble_vel is not None else None

        nm = self._nimble_map

        # --- 1. 转换 Pelvis/Root ---
        tx = nimble_pos[:, nm['pelvis_tx']]
        ty = nimble_pos[:, nm['pelvis_ty']]
        tz = nimble_pos[:, nm['pelvis_tz']]
        pelvis_pos = np.stack([tx, ty, tz], axis=-1)
        pelvis_pos = self._swap_coordinates(pelvis_pos)
        pelvis_pos = pelvis_pos * self.pelvis_pos_scale + self.pelvis_pos_offset
        qpos[:, self.MYOLEG_QPOS_INDICES['root_pos']] = pelvis_pos

        # 旋转: Euler 角 -> rotation matrix -> quaternion
        tilt = nimble_pos[:, nm['pelvis_tilt']]
        list_ = nimble_pos[:, nm['pelvis_list']]
        rot = nimble_pos[:, nm['pelvis_rotation']]

        for i in range(n_frames):
            R_os = self._euler_zxy_to_rotation_matrix(tilt[i], list_[i], rot[i])
            R_mj = self._rotation_opensim_to_mujoco(R_os)
            quat = R.from_matrix(R_mj).as_quat()  # (x, y, z, w)
            # MuJoCo quaternion 格式: [w, x, y, z]
            qpos[i, self.MYOLEG_QPOS_INDICES['root_quat']] = [quat[3], quat[0], quat[1], quat[2]]

        # --- 2. 直接映射下肢关节 ---
        joint_map = [
            ('hip_flexion_r', 'hip_flexion_r'),
            ('hip_adduction_r', 'hip_adduction_r'),
            ('hip_rotation_r', 'hip_rotation_r'),
            ('knee_angle_r', 'knee_angle_r'),
            ('ankle_angle_r', 'ankle_angle_r'),
            ('subtalar_angle_r', 'subtalar_angle_r'),
            ('mtp_angle_r', 'mtp_angle_r'),
            ('hip_flexion_l', 'hip_flexion_l'),
            ('hip_adduction_l', 'hip_adduction_l'),
            ('hip_rotation_l', 'hip_rotation_l'),
            ('knee_angle_l', 'knee_angle_l'),
            ('ankle_angle_l', 'ankle_angle_l'),
            ('subtalar_angle_l', 'subtalar_angle_l'),
            ('mtp_angle_l', 'mtp_angle_l'),
        ]

        for nimble_joint, myoleg_joint in joint_map:
            if nimble_joint in nm:
                qpos[:, self.MYOLEG_QPOS_INDICES[myoleg_joint]] = \
                    nimble_pos[:, nm[nimble_joint]]

        # --- 3. MyoLeg 特有的额外 DoF (knee translation/rotation, patella) ---
        extra_dofs = [
            'knee_angle_r_translation2', 'knee_angle_r_translation1',
            'knee_angle_r_rotation2', 'knee_angle_r_rotation3',
            'knee_angle_r_beta_translation2', 'knee_angle_r_beta_translation1',
            'knee_angle_r_beta_rotation1',
            'knee_angle_l_translation2', 'knee_angle_l_translation1',
            'knee_angle_l_rotation2', 'knee_angle_l_rotation3',
            'knee_angle_l_beta_translation2', 'knee_angle_l_beta_translation1',
            'knee_angle_l_beta_rotation1',
        ]
        for dof in extra_dofs:
            qpos[:, self.MYOLEG_QPOS_INDICES[dof]] = 0.0

        # --- 4. 转换速度 ---
        if nimble_vel is not None:
            vtx = nimble_vel[:, nm['pelvis_tx']]
            vty = nimble_vel[:, nm['pelvis_ty']]
            vtz = nimble_vel[:, nm['pelvis_tz']]
            pelvis_linvel = np.stack([vtx, vty, vtz], axis=-1)
            pelvis_linvel = self._swap_coordinates(pelvis_linvel) * self.pelvis_pos_scale
            qvel[:, self.MYOLEG_QVEL_INDICES['root_linvel']] = pelvis_linvel

            # 角速度: Euler 角速度 -> 角速度矢量
            v_tilt = nimble_vel[:, nm['pelvis_tilt']]
            v_list = nimble_vel[:, nm['pelvis_list']]
            v_rot = nimble_vel[:, nm['pelvis_rotation']]

            for i in range(n_frames):
                c1, s1 = np.cos(tilt[i]), np.sin(tilt[i])
                c2, s2 = np.cos(list_[i]), np.sin(list_[i])

                omega_os = np.array([
                    -s1 * v_list[i] + c1 * c2 * v_rot[i],
                    c1 * v_list[i] + s1 * c2 * v_rot[i],
                    v_tilt[i] - s2 * v_rot[i]
                ])

                T = np.array([
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0]
                ], dtype=np.float64)
                omega_mj = T @ omega_os
                qvel[i, self.MYOLEG_QVEL_INDICES['root_angvel']] = omega_mj

            # 关节角速度直接映射
            for nimble_joint, myoleg_joint in joint_map:
                if nimble_joint in nm:
                    qvel[:, self.MYOLEG_QVEL_INDICES[myoleg_joint]] = \
                        nimble_vel[:, nm[nimble_joint]]

            # 额外 DoF 速度设为 0
            for dof in extra_dofs:
                qvel[:, self.MYOLEG_QVEL_INDICES[dof]] = 0.0

        if single:
            qpos = qpos[0]
            if qvel is not None:
                qvel = qvel[0]

        result = {'qpos': qpos}
        if qvel is not None:
            result['qvel'] = qvel
        return result

    def convert_trial(self, trial_data: Dict) -> Dict:
        """
        转换整个 trial 的数据

        Args:
            trial_data: b3d_reader.extract_trial_data() 的输出

        Returns:
            包含 'qpos', 'qvel', 'time', 'dt' 等的字典
        """
        nimble_pos = trial_data['pos']
        nimble_vel = trial_data.get('vel')
        nimble_dof_names = trial_data.get('dof_names')
        nimble_dof_map = trial_data.get('dof_name_to_idx')

        converted = self.convert_single_frame(
            nimble_pos, nimble_vel,
            nimble_dof_names=nimble_dof_names,
            nimble_dof_map=nimble_dof_map
        )

        result = {
            'qpos': converted['qpos'],
            'time': trial_data['time'],
            'dt': trial_data['dt'],
        }

        if 'qvel' in converted:
            result['qvel'] = converted['qvel']

        if 'tau' in trial_data:
            result['tau_nimble'] = trial_data['tau']
            result['dof_names_nimble'] = trial_data.get('dof_names', [])

        for key in ['com_pos', 'com_vel', 'com_acc']:
            if key in trial_data:
                result[key] = trial_data[key]

        return result

    def save_converted(self, output_path: str, converted_data: Dict):
        """保存转换后的数据为 npz 格式"""
        np.savez_compressed(output_path, **converted_data)
        print(f"已保存: {output_path}")


def build_knee_coupling_functions():
    """
    构建 OpenSim Rajagopal 模型中 knee secondary motion 与 knee_angle 的耦合函数。

    在 Rajagopal 模型中，以下运动是 knee_angle 的函数:
    - tibia anterior/posterior translation
    - tibia medial/lateral translation
    - tibia internal/external rotation
    - patella translation/rotation

    返回函数 dict，输入 knee_angle [rad]，输出各 DoF 值。
    """

    def knee_translation2(knee_angle):
        """knee_angle_r_translation2 / tibia anterior-posterior [m]"""
        return np.clip(0.0025 + 0.001 * knee_angle, 0.0, 0.0068)

    def knee_translation1(knee_angle):
        """knee_angle_r_translation1 / tibia medial-lateral [m]"""
        return np.clip(0.0005 + 0.0003 * knee_angle, 0.0, 0.0016)

    def knee_rotation2(knee_angle):
        """knee_angle_r_rotation2 [rad]"""
        return np.clip(0.005 + 0.01 * knee_angle, 0.0, 0.034)

    def knee_rotation3(knee_angle):
        """knee_angle_r_rotation3 [rad]"""
        return np.clip(0.02 + 0.08 * knee_angle, 0.0, 0.263)

    def patella_translation2(knee_angle):
        """knee_angle_r_beta_translation2 [m]"""
        return np.clip(-0.025 - 0.008 * knee_angle, -0.041, -0.011)

    def patella_translation1(knee_angle):
        """knee_angle_r_beta_translation1 [m]"""
        return np.clip(0.005 + 0.015 * knee_angle, -0.023, 0.052)

    def patella_rotation1(knee_angle):
        """knee_angle_r_beta_rotation1 [rad]"""
        return np.clip(-0.5 - 0.3 * knee_angle, -1.79, 0.01)

    return {
        'knee_angle_r_translation2': knee_translation2,
        'knee_angle_r_translation1': knee_translation1,
        'knee_angle_r_rotation2': knee_rotation2,
        'knee_angle_r_rotation3': knee_rotation3,
        'knee_angle_r_beta_translation2': patella_translation2,
        'knee_angle_r_beta_translation1': patella_translation1,
        'knee_angle_r_beta_rotation1': patella_rotation1,
        # 左腿对称
        'knee_angle_l_translation2': knee_translation2,
        'knee_angle_l_translation1': knee_translation1,
        'knee_angle_l_rotation2': knee_rotation2,
        'knee_angle_l_rotation3': knee_rotation3,
        'knee_angle_l_beta_translation2': patella_translation2,
        'knee_angle_l_beta_translation1': patella_translation1,
        'knee_angle_l_beta_rotation1': patella_rotation1,
    }


if __name__ == "__main__":
    import sys
    from b3d_reader import B3DReader

    if len(sys.argv) < 3:
        print("用法: python b3d_to_myoleg_converter.py <b3d路径> <输出npz路径>")
        sys.exit(1)

    b3d_path = sys.argv[1]
    output_path = sys.argv[2]

    reader = B3DReader(b3d_path)
    trial_data = reader.extract_trial_data(0, 0)

    converter = B3DToMyoLegConverter(
        pelvis_pos_scale=1.0,
        coordinate_swap='xyz_to_xzy'
    )
    converted = converter.convert_trial(trial_data)

    converter.save_converted(output_path, converted)

    print(f"\n转换完成!")
    print(f"  qpos shape: {converted['qpos'].shape}")
    print(f"  qvel shape: {converted.get('qvel', None) and converted['qvel'].shape}")
    print(f"  time range: [{converted['time'][0]:.3f}, {converted['time'][-1]:.3f}] s")
    print(f"  dt: {converted['dt']:.4f} s")
