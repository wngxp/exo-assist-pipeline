# B3D 到 MyoLeg 动捕数据转换工具

将 AddBiomechanics (b3d) 格式动捕数据集转换为 MyoSuite MyoLeg 模型可用的运动参考数据，用于强化学习运动模仿训练。

## 依赖安装

```bash
pip install nimblephysics numpy scipy myosuite gymnasium tqdm
```

## 文件说明

| 文件 | 功能 |
|---|---|
| `b3d_reader.py` | 使用 nimblephysics 读取 b3d 文件，提取 pos/vel/tau 数据 |
| `b3d_to_myoleg_converter.py` | 核心转换逻辑：坐标系转换、关节映射、pelvis 姿态转换 |
| `motion_utils.py` | 数据加载、插值、播放、奖励计算、训练集导出 |
| `convert_b3d_to_myoleg.py` | 批量转换命令行脚本 |
| `demo_pipeline.py` | 端到端演示脚本（读取→转换→播放→导出） |
| `debug_myoleg.py` | 检查 MyoLeg qpos/qvel 结构的调试工具 |

## 动态 DoF 适配（重要）

本工具支持任意 DoF 数量的 b3d 文件（23 DoF 下肢版 / 37 DoF 完整版 / 自定义模型），通过**名称匹配**自动建立关节映射，无需硬编码索引。

### 关节名称别名

如果你的 b3d 文件中关节命名与标准 Rajagopal 不同（例如 `r_knee_flex` 而不是 `knee_angle_r`），转换器会自动通过内置别名表匹配：

```python
# b3d_to_myoleg_converter.py 中的别名表（可扩展）
JOINT_NAME_ALIASES = {
    'knee_angle_r': ['knee_angle_r', 'knee_flex_r', 'r_knee_flex', 'r_knee_angle'],
    'ankle_angle_r': ['ankle_angle_r', 'ankle_flex_r', 'r_ankle_flex'],
    # ... 更多别名
}
```

如果转换时报告某些关节"未找到"，请：
1. 先用 `b3d_reader.py` 打印实际的 DoF 名称列表
2. 在 `JOINT_NAME_ALIASES` 中添加你的命名别名
3. 重新运行转换

### 验证 DoF 名称

```bash
python b3d_reader.py your_data.b3d
```

输出示例（23 DoF 下肢模型）：
```
已加载 1 个 subject，骨架共 23 DoF
DoF 列表: ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation', 'pelvis_tx', ...]
```

转换器会自动识别这些名称并建立映射。上肢/脊柱关节如果在 b3d 中存在，会被忽略（MyoLeg 不需要）。

## 快速开始

### 1. 批量转换

```bash
python convert_b3d_to_myoleg.py data/raw/ output/myoleg_data/
```

转换单个文件：
```bash
python convert_b3d_to_myoleg.py data/subject_01.b3d output/
```

### 2. 验证数据

```bash
python motion_utils.py output/subject_01_s0_t0.npz
```

### 3. 在 MyoLeg 中播放

```python
from motion_utils import MyoLegReferenceMotion, play_reference_motion_in_myoleg

# 直接播放
play_reference_motion_in_myoleg('output/subject_01_s0_t0.npz', render=True)

# 或通过Python接口使用
ref = MyoLegReferenceMotion('output/subject_01_s0_t0.npz')
qpos, qvel = ref.get_frame(100)  # 获取第100帧
```

### 4. 集成到 RL 训练

```python
import gymnasium as gym
from motion_utils import MyoLegReferenceMotion, compute_mimic_reward

# 加载参考运动
ref = MyoLegReferenceMotion('converted.npz')

# 创建环境
env = gym.make('myoLegWalk-v0')
obs, info = env.reset()

for t in range(len(ref)):
    ref_qpos, ref_qvel = ref.get_frame(t)
    
    # 你的策略
    action = policy(obs, ref_qpos)
    
    obs, _, terminated, truncated, info = env.step(action)
    
    # 计算模仿奖励
    env_qpos = env.unwrapped.sim.data.qpos.copy()
    env_qvel = env.unwrapped.sim.data.qvel.copy()
    reward = compute_mimic_reward(env_qpos, ref_qpos, env_qvel, ref_qvel)
```

## 核心转换逻辑

### 坐标系差异

| | OpenSim (nimble/b3d) | MyoLeg (MuJoCo) |
|---|---|---|
| 世界坐标系 | Y-up (X=forward, Y=up, Z=right) | Z-up (X=forward, Y=right, Z=up) |
| Pelvis | 6 DoF Euler (tilt, list, rotation, tx, ty, tz) | 7 DoF freejoint (pos[x,y,z] + quat[w,x,y,z]) |
| Knee | 1 DoF (knee_angle) | 5 DoF + 髌骨 3 DoF |
| 上肢/脊柱 | 有 | 无 |

### Pelvis 转换

```
OpenSim Y-up -> MuJoCo Z-up:
  位置: [tx, ty, tz] -> [tx, tz, ty]  (默认 xyz_to_xzy)
  旋转: R_opensim = R_y(rot) @ R_x(list) @ R_z(tilt)   (ZXY order)
        R_intermediate = T @ R_opensim @ T    (T 交换 Y/Z)
        R_mujoco = R_z(-90°) @ R_intermediate  (MyoLeg 固定偏移)
        quat = rotation_matrix_to_quaternion(R_mujoco)
```

### 关节映射

直接一对一映射的关节 (值不变)：
- `hip_flexion_r/l`, `hip_adduction_r/l`, `hip_rotation_r/l`
- `knee_angle_r/l` -> `knee_angle_r/l`
- `ankle_angle_r/l`
- `subtalar_angle_r/l`
- `mtp_angle_r/l`

MyoLeg 特有的额外 DoF (设为 0)：
- `knee_angle_*_translation2/1`
- `knee_angle_*_rotation2/3`
- `knee_angle_*_beta_translation2/1`
- `knee_angle_*_beta_rotation1`

这些额外的 knee DoF 在 OpenSim 中是隐式耦合到 `knee_angle` 的 secondary motion，在 MuJoCo 中需要显式表示。由于它们的运动范围极小（mm 级别 / 几度），设为 0 是合理的近似。若需要更精确的结果，可使用 `build_knee_coupling_functions()` 中的耦合函数。

### 可选参数

```bash
python convert_b3d_to_myoleg.py data/ output/ \
    --coord-swap xyz_to_xz_neg_y \
    --pos-scale 0.95 \
    --pos-offset 0.0 0.0 0.02
```

| 参数 | 说明 |
|---|---|
| `--coord-swap` | 坐标轴交换策略。如果播放时模型方向不对，尝试 `xyz_to_xz_neg_y` |
| `--pos-scale` | 位置缩放（用于处理不同受试者身高差异） |
| `--pos-offset` | 全局位置偏移（用于调整模型在地面上的高度） |
| `--knee-coupling` | 启用 knee secondary motion 耦合函数 |

## 数据格式

转换后的 `.npz` 文件包含：

```python
data = np.load('converted.npz')

# 必需
qpos = data['qpos']       # (T, 35)  广义坐标位置
time = data['time']       # (T,)     时间戳 [s]
dt   = data['dt']         # scalar   时间步长

# 可选
qvel = data['qvel']       # (T, 34)  广义坐标速度
com_pos = data['com_pos'] # (T, 3)   质心位置
tau_nimble = data['tau_nimble']  # (T, N_dof) 原始 nimble 力矩
```

## MyoLeg qpos/qvel 结构

**qpos (35维)**:
- [0:3] root position (x, y, z)
- [3:7] root quaternion (w, x, y, z)
- [7:10] right hip (flexion, adduction, rotation)
- [10:15] right knee complex (translation2, translation1, angle, rotation2, rotation3)
- [15:18] right ankle, subtalar, mtp
- [18:21] right patella (beta translation2, translation1, rotation1)
- [21:24] left hip
- [24:29] left knee complex
- [29:32] left ankle, subtalar, mtp
- [32:35] left patella

**qvel (34维)**:
- [0:3] root angular velocity
- [3:6] root linear velocity
- [6:33] 各关节角速度

## 注意事项

1. **nimblephysics安装**: 在Linux上可能需要安装依赖 `libpcre3-dev` 等。如遇问题，参考 [nimblephysics文档](https://nimblephysics.org/)。

2. **坐标系调试**: 由于 OpenSim 和 MuJoCo 的 pelvis 初始朝向定义不同，转换后的 root rotation 可能需要微调。如果播放时模型朝向异常（例如人躺在地上或朝错误方向），尝试修改 `coordinate_swap` 或调整 pelvis rotation 的转换矩阵。

3. **模型缩放**: b3d 数据集中的受试者身高体重不同。MyoLeg 模型是固定尺寸的，如果受试者与模型差异较大，需要缩放位置数据或使用 MuJoCo 的模型缩放功能。

4. ** Muscle Excitation**: MyoLeg 有 80 个 muscle actuators，而 Rajagopal 模型有 23 个。力矩(tau)和肌肉激励(excitation)之间的映射不是直接的。如果要使用肌肉层面的模仿，需要通过 MyoLeg 的 muscle geometry 进行 EMG-to-excitation 或 inverse dynamics 计算。

5. **地面反力**: b3d 中可能包含 GRF 和 COP 数据，可用于训练 contact-rich 的策略。这些数据可以通过 `b3d_reader` 提取，但格式与 MyoLeg 的接触力不同。

## 训练建议

对于运动模仿任务，推荐以下 RL 设置：

```python
# 参考运动跟踪奖励
r_pos = exp(-scale_pos * ||env_qpos - ref_qpos||^2)
r_vel = exp(-scale_vel * ||env_qvel - ref_qvel||^2)
r_com = exp(-scale_com * ||env_com - ref_com||^2)
r = w_pos * r_pos + w_vel * r_vel + w_com * r_com

# 活着奖励
r_alive = 1.0

# 总奖励
reward = r + r_alive
```

论文参考: "Reinforcement learning-based motion imitation for physiologically plausible musculoskeletal motor control" (2025)

## 许可证

MIT License
