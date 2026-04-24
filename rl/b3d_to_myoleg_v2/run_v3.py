#!/usr/bin/env python3
"""
run_v3.py: 批量递归转换目录下所有 b3d 文件。

功能:
    1. 递归搜索输入目录下的所有 .b3d 文件
    2. 每个 b3d 的所有 trial 拼接成一个 merged.npz
    3. 输出目录结构与原始 b3d 目录结构镜像一致

用法:
    conda activate b3d_convert
    python run_v3.py <包含b3d的根目录>

示例:
    # 转换 test 数据集
    python run_v3.py /home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/test/

    # 转换 train 数据集
    python run_v3.py /home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/

    # 转换整个 addbiomechanics（train + test）
    python run_v3.py /home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/

输出路径规则:
    输入: .../data/addbiomechanics/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/AB10_split0.b3d
    输出: .../b3d_to_myoleg_v2/Data/test/No_Arm/Camargo2021_Formatted_No_Arm/AB10_split0/AB10_split0_merged.npz
"""

import sys
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from b3d_reader import B3DReader
from b3d_to_myoleg_converter import B3DToMyoLegConverter

# ============ 配置 ============
DEFAULT_OUTPUT_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Data"
)
# ================================


def compute_mirror_output_path(b3d_path: str, output_root: str) -> str:
    """
    根据 b3d 路径计算镜像输出目录。
    以 'addbiomechanics' 为分界，保留之后的相对目录结构。
    """
    b3d_path = os.path.abspath(b3d_path)
    output_root = os.path.abspath(output_root)

    norm_path = os.path.normpath(b3d_path)
    parts = norm_path.split(os.sep)

    if "addbiomechanics" in parts:
        idx = parts.index("addbiomechanics")
        # +1 跳过 addbiomechanics，-1 去掉文件名，得到相对目录
        rel_parts = parts[idx + 1:-1]
        rel_dir = os.path.join(*rel_parts) if rel_parts else ""
        return os.path.join(output_root, rel_dir)

    # 回退：以 data 为分界
    if "data" in parts:
        idx = parts.index("data")
        if idx + 2 < len(parts) - 1:
            rel_parts = parts[idx + 2:-1]
            rel_dir = os.path.join(*rel_parts) if rel_parts else ""
            return os.path.join(output_root, rel_dir)

    # 最终回退：用父文件夹名
    b3d_dir = os.path.dirname(b3d_path)
    folder_name = os.path.basename(b3d_dir)
    return os.path.join(output_root, folder_name)


def merge_trials_to_single_npz(b3d_path: str,
                                 processing_pass: int = 1,
                                 output_root: str = None):
    """
    读取单个 b3d 的所有 trial，转换为 MyoLeg 格式，拼接成一个连续轨迹。
    输出文件按照原始 b3d 的目录结构镜像保存。
    """
    if not os.path.isfile(b3d_path):
        raise FileNotFoundError(f"b3d 文件不存在: {b3d_path}")

    output_root = output_root or DEFAULT_OUTPUT_ROOT
    output_dir = compute_mirror_output_path(b3d_path, output_root)
    os.makedirs(output_dir, exist_ok=True)

    b3d_basename = os.path.splitext(os.path.basename(b3d_path))[0]
    output_npz = os.path.join(output_dir, f"{b3d_basename}_merged.npz")

    # 如果已经存在，跳过（可以改为覆盖）
    if os.path.exists(output_npz):
        print(f"[跳过] 已存在: {output_npz}")
        return output_npz

    reader = B3DReader(b3d_path)
    info = reader.get_subject_info(0)
    n_trials = info['num_trials']

    converter = B3DToMyoLegConverter(
        coordinate_swap='xyz_to_xzy',
        pelvis_pos_scale=1.0,
        pelvis_pos_offset=np.array([0.0, 0.0, 0.0])
    )

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

        if dt is None:
            dt = trial_dt

        trial_data = reader.extract_trial_data(
            subject_idx=0, trial=trial_idx, processing_pass=processing_pass
        )
        converted = converter.convert_trial(trial_data)

        all_qpos.append(converted['qpos'])
        all_qvel.append(converted['qvel'])

        trial_time = trial_data['time'].copy()
        if global_time_offset > 0:
            trial_time += global_time_offset
        all_time.append(trial_time)
        global_time_offset = trial_time[-1] + dt

        if 'contact' in converted:
            all_contact.append(converted['contact'])
        elif 'contact' in trial_data:
            all_contact.append(trial_data['contact'])

        if 'acc_nimble' in converted:
            all_acc_nimble.append(converted['acc_nimble'])
        elif 'acc' in trial_data:
            all_acc_nimble.append(trial_data['acc'])

        for key in ['com_pos', 'com_vel', 'com_acc']:
            arr_list = locals().get(f'all_{key}', [])
            if key in converted:
                arr_list.append(converted[key])
            elif key in trial_data:
                arr_list.append(trial_data[key])

        if 'tau_nimble' in converted:
            all_tau_nimble.append(converted['tau_nimble'])

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

    if all_acc_nimble:
        merged['acc_nimble'] = np.concatenate(all_acc_nimble, axis=0)

    for key in ['com_pos', 'com_vel', 'com_acc', 'tau_nimble']:
        arr_list = locals().get(f'all_{key}', [])
        if arr_list:
            merged[key] = np.concatenate(arr_list, axis=0)

    np.savez_compressed(output_npz, **merged)
    return output_npz


def find_all_b3d_files(root_dir: str) -> list:
    """递归搜索目录下所有 .b3d 文件"""
    pattern = os.path.join(root_dir, "**", "*.b3d")
    return sorted(glob.glob(pattern, recursive=True))


def main():
    parser = argparse.ArgumentParser(description='批量递归转换 b3d 文件为 npz')
    parser.add_argument('input_dir', help='包含 .b3d 文件的根目录')
    parser.add_argument('--output-root', default=None,
                        help=f'输出根目录 (默认: {DEFAULT_OUTPUT_ROOT})')
    parser.add_argument('--processing-pass', type=int, default=1,
                        help='处理 pass 索引 (默认: 1)')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的 npz 文件')
    args = parser.parse_args()

    output_root = args.output_root or DEFAULT_OUTPUT_ROOT
    os.makedirs(output_root, exist_ok=True)

    # 搜索所有 b3d
    b3d_files = find_all_b3d_files(args.input_dir)
    if not b3d_files:
        print(f"未在 {args.input_dir} 下找到任何 .b3d 文件")
        return

    print(f"找到 {len(b3d_files)} 个 b3d 文件")
    print(f"输出根目录: {output_root}")
    print("=" * 60)

    success = 0
    failed = 0
    skipped = 0

    for b3d_path in tqdm(b3d_files, desc="转换进度"):
        # 计算输出路径，检查是否已存在
        output_dir = compute_mirror_output_path(b3d_path, output_root)
        b3d_basename = os.path.splitext(os.path.basename(b3d_path))[0]
        output_npz = os.path.join(output_dir, f"{b3d_basename}_merged.npz")

        if os.path.exists(output_npz) and not args.overwrite:
            tqdm.write(f"[跳过] {os.path.basename(b3d_path)} -> 已存在")
            skipped += 1
            continue

        try:
            merge_trials_to_single_npz(
                b3d_path,
                processing_pass=args.processing_pass,
                output_root=output_root
            )
            success += 1
        except Exception as e:
            tqdm.write(f"[错误] {b3d_path}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print("批量转换完成")
    print("=" * 60)
    print(f"  成功: {success}")
    print(f"  跳过: {skipped}")
    print(f"  失败: {failed}")
    print(f"  输出目录: {output_root}")


if __name__ == "__main__":
    main()