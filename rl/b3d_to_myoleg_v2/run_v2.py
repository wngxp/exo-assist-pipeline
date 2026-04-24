"""
run_v2.py: 读取单个 b3d 的所有 trial，拼接成一个完整的长轨迹 npz。
输出文件自动按照原始 b3d 的目录结构镜像保存。

用法:
    conda activate b3d_convert
    python run_v2.py <b3d文件路径>

示例:
    python run_v2.py /home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/AB10_split0.b3d

输出路径规则:
    输入: .../data/addbiomechanics/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/AB10_split0.b3d
    输出: .../b3d_to_myoleg_v2/Data/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/AB10_split0_merged.npz
"""

import sys
import os
import argparse
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from b3d_reader import B3DReader
from b3d_to_myoleg_converter import B3DToMyoLegConverter

# ============ 配置 ============
# 默认输出根目录：与原始 data 结构镜像
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data"
)
# ================================


def compute_mirror_output_path(b3d_path: str, output_root: str) -> str:
    """
    根据 b3d 路径计算镜像输出路径。

    规则：找到 b3d 路径中的 'addbiomechanics' 目录，
    取之后的相对目录结构，映射到 output_root 下。
    """
    b3d_path = os.path.abspath(b3d_path)
    output_root = os.path.abspath(output_root)

    parts = b3d_path.split(os.sep)

    # 策略1: 以 addbiomechanics 为分界
    if "addbiomechanics" in parts:
        idx = parts.index("addbiomechanics")
        # +1 跳过 addbiomechanics，-1 去掉文件名，得到相对目录
        rel_parts = parts[idx + 1:-1]
        rel_dir = os.path.join(*rel_parts) if rel_parts else ""
        output_dir = os.path.join(output_root, rel_dir)
        return output_dir

    # 策略2: 以 data 为分界（回退）
    if "data" in parts:
        idx = parts.index("data")
        # +2 跳过 data/xxx（通常是 data/addbiomechanics），-1 去掉文件名
        if idx + 2 < len(parts) - 1:
            rel_parts = parts[idx + 2:-1]
            rel_dir = os.path.join(*rel_parts) if rel_parts else ""
            output_dir = os.path.join(output_root, rel_dir)
            return output_dir

    # 策略3: 完全无法识别，就用 b3d 的父文件夹名作为子目录
    b3d_dir = os.path.dirname(b3d_path)
    folder_name = os.path.basename(b3d_dir)
    output_dir = os.path.join(output_root, folder_name)
    return output_dir


def merge_trials_to_single_npz(b3d_path: str,
                                 processing_pass: int = 1,
                                 output_root: str = None):
    """
    读取 b3d 的所有 trial，转换为 MyoLeg 格式，拼接成一个连续轨迹。
    输出文件按照原始 b3d 的目录结构镜像保存。
    """
    if not os.path.isfile(b3d_path):
        raise FileNotFoundError(f"b3d 文件不存在: {b3d_path}")

    # 计算输出路径
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    output_dir = compute_mirror_output_path(b3d_path, output_root)
    os.makedirs(output_dir, exist_ok=True)

    b3d_basename = os.path.splitext(os.path.basename(b3d_path))[0]
    output_npz = os.path.join(output_dir, f"{b3d_basename}_merged.npz")

    print("=" * 60)
    print("1. 读取 b3d 文件")
    print("=" * 60)
    reader = B3DReader(b3d_path)
    info = reader.get_subject_info(0)
    n_trials = info['num_trials']

    print(f"输入:  {b3d_path}")
    print(f"输出:  {output_npz}")
    print(f"  受试者: 身高 {info['height_m']:.2f}m, 体重 {info['mass_kg']:.1f}kg")
    print(f"  共 {n_trials} 个 trial")

    print("\n" + "=" * 60)
    print("2. 逐个 trial 提取 + 转换")
    print("=" * 60)

    converter = B3DToMyoLegConverter(
        coordinate_swap='xyz_to_xzy',
        pelvis_pos_scale=1.0,
        pelvis_pos_offset=np.array([0.0, 0.0, 0.0])
    )

    # 收集所有 trial 的数据
    all_qpos = []
    all_qvel = []
    all_time = []
    all_contact = []
    all_acc_nimble = []
    all_com_pos = []
    all_com_vel = []
    all_com_acc = []
    all_tau_nimble = []

    dt = None
    trial_lengths = []
    global_time_offset = 0.0

    for trial_idx in range(n_trials):
        t_info = reader.get_trial_info(0, trial_idx)
        n_frames = t_info['length']
        trial_dt = t_info['timestep']
        trial_lengths.append(n_frames)

        print(f"  trial {trial_idx}: {n_frames} frames, dt={trial_dt:.4f}s, "
              f"duration={n_frames * trial_dt:.2f}s")

        if dt is None:
            dt = trial_dt

        # 提取 + 转换
        trial_data = reader.extract_trial_data(
            subject_idx=0, trial=trial_idx, processing_pass=processing_pass
        )
        converted = converter.convert_trial(trial_data)

        # 拼接
        all_qpos.append(converted['qpos'])
        all_qvel.append(converted['qvel'])

        # 时间轴连续拼接
        trial_time = trial_data['time'].copy()
        if global_time_offset > 0:
            trial_time += global_time_offset
        all_time.append(trial_time)
        global_time_offset = trial_time[-1] + dt

        # contact（走路训练核心数据）
        if 'contact' in converted:
            all_contact.append(converted['contact'])
        elif 'contact' in trial_data:
            all_contact.append(trial_data['contact'])

        # 加速度
        if 'acc_nimble' in converted:
            all_acc_nimble.append(converted['acc_nimble'])
        elif 'acc' in trial_data:
            all_acc_nimble.append(trial_data['acc'])

        # 质心运动学
        for key in ['com_pos', 'com_vel', 'com_acc']:
            arr_list = locals().get(f'all_{key}', [])
            if key in converted:
                arr_list.append(converted[key])
            elif key in trial_data:
                arr_list.append(trial_data[key])

        # 力矩
        if 'tau_nimble' in converted:
            all_tau_nimble.append(converted['tau_nimble'])

    print("\n" + "=" * 60)
    print("3. 拼接所有 trial")
    print("=" * 60)

    merged = {
        'qpos': np.concatenate(all_qpos, axis=0),
        'qvel': np.concatenate(all_qvel, axis=0),
        'time': np.concatenate(all_time, axis=0),
        'dt': float(dt),
        'n_trials': int(n_trials),
        'trial_boundaries': np.cumsum(trial_lengths)[:-1].tolist(),
    }

    if all_contact:
        merged['contact'] = np.concatenate(all_contact, axis=0)
        if 'contact_bodies' in converted:
            merged['contact_bodies'] = converted['contact_bodies']
        elif 'contact_bodies' in trial_data:
            merged['contact_bodies'] = trial_data['contact_bodies']
        print(f"  contact 已保存: {merged['contact'].shape}")

    if all_acc_nimble:
        merged['acc_nimble'] = np.concatenate(all_acc_nimble, axis=0)
        print(f"  acc_nimble 已保存: {merged['acc_nimble'].shape}")

    for key in ['com_pos', 'com_vel', 'com_acc', 'tau_nimble']:
        arr_list = locals().get(f'all_{key}', [])
        if arr_list:
            merged[key] = np.concatenate(arr_list, axis=0)

    # 保存
    np.savez_compressed(output_npz, **merged)

    print("\n" + "=" * 60)
    print("4. 输出验证")
    print("=" * 60)
    print(f"  文件: {output_npz}")
    print(f"  总帧数: {merged['qpos'].shape[0]}")
    print(f"  qpos: {merged['qpos'].shape}")
    print(f"  qvel: {merged['qvel'].shape}")
    print(f"  总时长: {merged['time'][-1]:.2f} s")
    print(f"  trial 边界索引: {merged['trial_boundaries']}")
    print(f"  文件大小: {os.path.getsize(output_npz) / 1024**2:.2f} MB")

    return output_npz


def main():
    parser = argparse.ArgumentParser(description='将 b3d 文件的所有 trial 拼接成单个 npz')
    parser.add_argument('b3d_path', help='输入 .b3d 文件路径')
    parser.add_argument('--output-root', default=None,
                        help=f'输出根目录 (默认: {DEFAULT_OUTPUT_ROOT})')
    parser.add_argument('--processing-pass', type=int, default=1,
                        help='处理 pass 索引 (默认: 1)')
    args = parser.parse_args()

    output_npz = merge_trials_to_single_npz(
        args.b3d_path,
        processing_pass=args.processing_pass,
        output_root=args.output_root
    )

    print("\n" + "=" * 60)
    print("完成！下一步建议")
    print("=" * 60)
    print(f"  输出文件: {output_npz}")
    print("  播放验证: python motion_utils.py " + output_npz)


if __name__ == "__main__":
    main()