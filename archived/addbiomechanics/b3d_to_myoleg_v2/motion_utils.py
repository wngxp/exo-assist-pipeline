import numpy as np
import os


def load_converted_data(npz_path: str) -> dict:
    """加载转换后的npz数据"""
    data = np.load(npz_path)
    return {k: data[k] for k in data.files}


class MyoLegReferenceMotion:
    """
    包装转换后的数据，提供与MyoLeg训练环境兼容的接口。
    
    使用方法:
        ref = MyoLegReferenceMotion('converted_data.npz')
        
        # 获取某一帧的参考状态
        qpos, qvel = ref.get_frame(100)
        
        # 插值获取任意时间点的状态
        qpos, qvel = ref.get_state_at_time(1.5)  # t=1.5s
        
        # 与环境集成
        env.reset()
        for t in range(ref.length):
            ref_qpos, ref_qvel = ref.get_frame(t)
            # 在RL奖励中使用 ref_qpos 作为目标姿态
            action = policy(obs, ref_qpos)
            obs, reward, done, info = env.step(action)
    """
    
    def __init__(self, npz_path: str):
        data = load_converted_data(npz_path)
        self.qpos = data['qpos']  # (T, 35)
        self.time = data['time']  # (T,)
        self.dt = float(data['dt'])
        self.length = self.qpos.shape[0]
        
        if 'qvel' in data:
            self.qvel = data['qvel']  # (T, 34)
        else:
            # 如果没有qvel，用有限差分估计
            self.qvel = self._estimate_qvel()
        
        # 也可以加载力矩数据（如果需要用于muscle excitation参考）
        self.tau_nimble = data.get('tau_nimble', None)
        
    def _estimate_qvel(self) -> np.ndarray:
        """通过有限差分估计qvel"""
        qvel = np.zeros((self.length, 34))
        # root线速度 = qpos位置差分
        qvel[1:-1, 3:6] = (self.qpos[2:, :3] - self.qpos[:-2, :3]) / (2 * self.dt)
        qvel[0, 3:6] = (self.qpos[1, :3] - self.qpos[0, :3]) / self.dt
        qvel[-1, 3:6] = (self.qpos[-1, :3] - self.qpos[-2, :3]) / self.dt
        
        # root角速度: 从quaternion差分计算
        for i in range(1, self.length - 1):
            qvel[i, :3] = _quat_diff_to_angular_velocity(
                self.qpos[i-1, 3:7], self.qpos[i+1, 3:7], 2 * self.dt
            )
        
        # 关节角速度
        qvel[:, 6:] = np.gradient(self.qpos[:, 7:], self.dt, axis=0, edge_order=2)
        return qvel
    
    def get_frame(self, frame_idx: int) -> tuple:
        """获取指定帧的qpos和qvel"""
        frame_idx = np.clip(frame_idx, 0, self.length - 1)
        return self.qpos[frame_idx].copy(), self.qvel[frame_idx].copy()
    
    def get_state_at_time(self, t: float) -> tuple:
        """通过线性插值获取任意时间点的状态"""
        if t <= self.time[0]:
            return self.get_frame(0)
        if t >= self.time[-1]:
            return self.get_frame(-1)
        
        idx = np.searchsorted(self.time, t)
        t0, t1 = self.time[idx-1], self.time[idx]
        alpha = (t - t0) / (t1 - t0)
        
        qpos = _slerp_qpos(self.qpos[idx-1], self.qpos[idx], alpha)
        qvel = (1 - alpha) * self.qvel[idx-1] + alpha * self.qvel[idx]
        return qpos, qvel
    
    def __len__(self):
        return self.length
    
    @property
    def duration(self) -> float:
        return self.time[-1] - self.time[0]


def _quat_diff_to_angular_velocity(q_prev: np.ndarray, q_next: np.ndarray, 
                                    dt: float) -> np.ndarray:
    """
    从两个quaternion计算角速度
    q格式: [w, x, y, z] (MuJoCo convention)
    """
    # 计算相对旋转 quaternion
    # q_rel = q_next * inverse(q_prev)
    w0, x0, y0, z0 = q_prev
    w1, x1, y1, z1 = q_next
    
    # inverse of q_prev
    w0_inv, x0_inv, y0_inv, z0_inv = w0, -x0, -y0, -z0
    
    # q_rel = q_next * q_prev_inv
    w_rel = w1*w0_inv - x1*x0_inv - y1*y0_inv - z1*z0_inv
    x_rel = w1*x0_inv + x1*w0_inv + y1*z0_inv - z1*y0_inv
    y_rel = w1*y0_inv - x1*z0_inv + y1*w0_inv + z1*x0_inv
    z_rel = w1*z0_inv + x1*y0_inv - y1*x0_inv + z1*w0_inv
    
    # 归一化
    norm = np.sqrt(w_rel**2 + x_rel**2 + y_rel**2 + z_rel**2)
    if norm > 1e-10:
        w_rel /= norm
        x_rel /= norm
        y_rel /= norm
        z_rel /= norm
    
    # 转换为角速度 (使用小角度近似)
    # omega = 2 * [x_rel, y_rel, z_rel] / dt (如果w_rel接近1)
    if w_rel < 0:
        w_rel = -w_rel
        x_rel = -x_rel
        y_rel = -y_rel
        z_rel = -z_rel
    
    # 确保取最短路径
    if w_rel > 1.0:
        w_rel = 1.0
    
    angle = 2 * np.arccos(np.clip(w_rel, -1, 1))
    if np.abs(angle) < 1e-6:
        return np.zeros(3)
    
    axis = np.array([x_rel, y_rel, z_rel])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-10:
        return np.zeros(3)
    axis /= axis_norm
    
    omega = axis * angle / dt
    return omega


def _slerp_qpos(qpos0: np.ndarray, qpos1: np.ndarray, alpha: float) -> np.ndarray:
    """
    对qpos进行插值，其中root quaternion使用SLERP，其他使用线性插值
    """
    result = qpos0.copy()
    
    # root position 线性插值
    result[:3] = (1 - alpha) * qpos0[:3] + alpha * qpos1[:3]
    
    # root quaternion SLERP
    q0 = qpos0[3:7]  # [w, x, y, z]
    q1 = qpos1[3:7]
    
    dot = np.dot(q0, q1)
    
    # 如果点积为负，取反其中一个quaternion以确保最短路径
    if dot < 0:
        q1 = -q1
        dot = -dot
    
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # 线性插值（quaternion非常接近）
        result[3:7] = q0 + alpha * (q1 - q0)
        result[3:7] /= np.linalg.norm(result[3:7])
    else:
        theta_0 = np.arccos(dot)
        theta = theta_0 * alpha
        sin_theta = np.sin(theta)
        sin_theta_0 = np.sin(theta_0)
        
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        
        result[3:7] = s0 * q0 + s1 * q1
    
    # 其他关节线性插值
    result[7:] = (1 - alpha) * qpos0[7:] + alpha * qpos1[7:]
    
    return result


# ============================================================
# MyoLeg 环境集成示例
# ============================================================

import os
import numpy as np
import mujoco
import mediapy as media
from tqdm import tqdm

def play_reference_motion_in_myoleg(npz_path: str, 
                                    render: bool = True,
                                    output_video: str = "motion_video.mp4",
                                    fps: int = 30):
    """
    播放参考运动：render=True 生成视频文件（无窗口），render=False 静默播放。
    """
    import os
    # 设置无头渲染后端（必须放在导入 mujoco 之前）
    os.environ['MUJOCO_GL'] = 'osmesa'   # 使用 CPU 渲染，稳定可靠
    
    import myosuite
    import gymnasium as gym
    import numpy as np
    import mujoco
    import imageio
    from tqdm import tqdm
    from motion_utils import MyoLegReferenceMotion

    ref = MyoLegReferenceMotion(npz_path)
    env = gym.make('myoLegWalk-v0')
    env.reset()

    if render:
        # 获取底层的 MuJoCo 模型和数据（绕过 dm_control 包装）
        # 注意：env.unwrapped.sim 可能是 dm_control 的 Physics 对象
        # 我们直接取内部的 _model 和 _data
        if hasattr(env.unwrapped.sim, '_model'):
            model = env.unwrapped.sim._model
            data = env.unwrapped.sim._data
        else:
            model = env.unwrapped.sim.model
            data = env.unwrapped.sim.data

        # 设置渲染分辨率
        height, width = 720, 1280
        # 获取相机 ID（使用第一个相机，或根据名称获取）
        camera_id = 0
        # 如果模型中有名为 "track" 的相机，可以改用：
        # camera_id = model.camera('track').id if 'track' in model.camera_names else 0
        
        renderer = mujoco.Renderer(model, height, width)
        
        frames = []
        print(f"开始渲染视频（共 {ref.length} 帧）...")
        for t in tqdm(range(ref.length)):
            # 更新状态
            data.qpos[:] = ref.qpos[t]
            data.qvel[:] = ref.qvel[t]
            mujoco.mj_forward(model, data)
            
            # 渲染当前帧
            renderer.update_scene(data, camera=camera_id)
            pixels = renderer.render()
            frames.append(pixels)
        
        renderer.close()
        env.close()
        
        # 保存视频
        print(f"正在保存视频到 {output_video}...")
        with imageio.get_writer(output_video, fps=fps, macro_block_size=1) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"视频生成完成！")
    else:
        # 静默播放
        for t in range(ref.length):
            env.unwrapped.sim.data.qpos[:] = ref.qpos[t]
            env.unwrapped.sim.data.qvel[:] = ref.qvel[t]
            env.unwrapped.sim.forward()
        env.close()
        print(f"静默播放完成，总帧数: {ref.length}")
    

def compute_mimic_reward(env_qpos: np.ndarray, ref_qpos: np.ndarray,
                        env_qvel: np.ndarray, ref_qvel: np.ndarray,
                        weights: dict = None) -> float:
    """
    计算运动模仿奖励（用于RL训练）
    
    参考: DeepMimic风格的模仿奖励 = w_pos * exp(-||qpos_diff||^2) + w_vel * exp(-||qvel_diff||^2)
    
    Args:
        env_qpos: 环境当前qpos (35,)
        ref_qpos: 参考qpos (35,)
        env_qvel: 环境当前qvel (34,)
        ref_qvel: 参考qvel (34,)
        weights: 各部分的权重和缩放因子
    
    Returns:
        奖励值 [0, 1]
    """
    if weights is None:
        weights = {
            'pos_weight': 0.5,
            'vel_weight': 0.3,
            'com_weight': 0.2,
            'pos_scale': 10.0,  # 1/pos_scale^2用于指数衰减
            'vel_scale': 1.0,
        }
    
    # qpos差异（排除root位置，因为绝对位置不重要）
    # 或者可以只比较joint angles
    joint_diff = env_qpos[7:] - ref_qpos[7:]
    pos_err = np.sum(joint_diff ** 2)
    
    # qvel差异
    vel_diff = env_qvel - ref_qvel
    vel_err = np.sum(vel_diff ** 2)
    
    # 模仿奖励
    r_pos = np.exp(-weights['pos_scale'] * pos_err)
    r_vel = np.exp(-weights['vel_scale'] * vel_err)
    
    reward = (weights['pos_weight'] * r_pos + 
              weights['vel_weight'] * r_vel)
    
    return reward


def export_for_training(npz_path: str, output_dir: str, split_ratio=0.9):
    """
    将数据导出为训练格式，分为train/val
    
    输出:
        {output_dir}/train.npz
        {output_dir}/val.npz
    """
    ref = MyoLegReferenceMotion(npz_path)
    
    # 随机划分
    n_train = int(ref.length * split_ratio)
    indices = np.random.permutation(ref.length)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_data = {
        'qpos': ref.qpos[train_idx],
        'qvel': ref.qvel[train_idx],
        'time': ref.time[train_idx],
        'dt': ref.dt,
    }
    np.savez_compressed(os.path.join(output_dir, 'train.npz'), **train_data)
    
    # 保存验证集
    val_data = {
        'qpos': ref.qpos[val_idx],
        'qvel': ref.qvel[val_idx],
        'time': ref.time[val_idx],
        'dt': ref.dt,
    }
    np.savez_compressed(os.path.join(output_dir, 'val.npz'), **val_data)
    
    print(f"训练数据已导出到 {output_dir}")
    print(f"  Train: {len(train_idx)} frames")
    print(f"  Val: {len(val_idx)} frames")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
        print(f"加载参考运动: {npz_path}")
        ref = MyoLegReferenceMotion(npz_path)
        print(f"  总帧数: {ref.length}")
        print(f"  时长: {ref.duration:.2f} s")
        print(f"  dt: {ref.dt:.4f} s")
        print(f"  qpos shape: {ref.qpos.shape}")
        print(f"  qvel shape: {ref.qvel.shape}")
        
        # 测试插值
        t_test = ref.duration * 0.5
        qpos_t, qvel_t = ref.get_state_at_time(t_test)
        print(f"\n  t={t_test:.2f}s 插值状态:")
        print(f"    qpos[0:7]: {qpos_t[:7]}")
    else:
        print("用法: python motion_utils.py <converted.npz>")
