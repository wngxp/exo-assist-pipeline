#!/usr/bin/env python3
"""
b3d -> MyoLeg 批量转换脚本

用法:
    python convert_b3d_to_myoleg.py <输入b3d文件或文件夹> <输出目录> [选项]

示例:
    # 转换单个文件
    python convert_b3d_to_myoleg.py data/subject_01.b3d output/myoleg_data/

    # 批量转换文件夹下所有b3d
    python convert_b3d_to_myoleg.py data/raw/ output/myoleg_data/

    # 调整坐标系转换
    python convert_b3d_to_myoleg.py data/ output/ --coord-swap xyz_to_xz_neg_y --pos-scale 0.9
"""

import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm

from b3d_reader import B3DReader
from b3d_to_myoleg_converter import B3DToMyoLegConverter


def convert_single_file(b3d_path: str, output_dir: str, 
                       converter_kwargs: dict,
                       trial_indices: list = None,
                       subject_indices: list = None):
    """
    转换单个b3d文件
    
    Args:
        b3d_path: 输入.b3d文件路径
        output_dir: 输出目录
        converter_kwargs: 转换器参数
        trial_indices: 指定要转换的trial索引列表，None表示全部
        subject_indices: 指定要转换的subject索引列表（对于单个b3d文件通常为[0]）
    """
    reader = B3DReader(b3d_path)
    converter = B3DToMyoLegConverter(**converter_kwargs)
    
    base_name = os.path.splitext(os.path.basename(b3d_path))[0]
    
    subjects_to_process = subject_indices if subject_indices else range(len(reader.subjects))
    
    for s_idx in subjects_to_process:
        info = reader.get_subject_info(s_idx)
        n_trials = info['num_trials']
        
        trials_to_process = trial_indices if trial_indices else range(n_trials)
        
        for trial in trials_to_process:
            if trial >= n_trials:
                print(f"警告: {base_name} subject {s_idx} 没有trial {trial}，跳过")
                continue
            
            t_info = reader.get_trial_info(s_idx, trial)
            print(f"处理: {base_name} | subject={s_idx} | trial={trial} | frames={t_info['length']}")
            
            # 提取数据
            trial_data = reader.extract_trial_data(s_idx, trial)
            
            # 转换
            converted = converter.convert_trial(trial_data)
            
            # 保存
            out_name = f"{base_name}_s{s_idx}_t{trial}.npz"
            out_path = os.path.join(output_dir, out_name)
            converter.save_converted(out_path, converted)


def batch_convert(input_path: str, output_dir: str,
                 converter_kwargs: dict,
                 trial_indices: list = None):
    """批量转换"""
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(input_path):
        # 单个文件
        convert_single_file(input_path, output_dir, converter_kwargs, trial_indices)
    else:
        # 文件夹
        b3d_files = glob.glob(os.path.join(input_path, "**", "*.b3d"), recursive=True)
        b3d_files += glob.glob(os.path.join(input_path, "*.b3d"))
        b3d_files = sorted(list(set(b3d_files)))
        
        print(f"找到 {len(b3d_files)} 个b3d文件")
        
        for b3d_file in tqdm(b3d_files, desc="转换进度"):
            try:
                convert_single_file(b3d_file, output_dir, converter_kwargs, trial_indices)
            except Exception as e:
                print(f"\n错误: 转换 {b3d_file} 失败: {e}")
                continue
    
    print(f"\n所有数据已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='将AddBiomechanics b3d动捕数据转换为MyoLeg格式')
    parser.add_argument('input', help='输入b3d文件路径或包含b3d文件的文件夹')
    parser.add_argument('output', help='输出目录')
    parser.add_argument('--coord-swap', default='xyz_to_xzy',
                       choices=['xyz_to_xzy', 'xyz_to_xz_neg_y', 'identity'],
                       help='坐标轴交换策略 (默认: xyz_to_xzy)')
    parser.add_argument('--pos-scale', type=float, default=1.0,
                       help='位置缩放因子 (默认: 1.0)')
    parser.add_argument('--pos-offset', nargs=3, type=float, default=[0, 0, 0],
                       help='位置偏移量 (默认: 0 0 0)')
    parser.add_argument('--trials', nargs='+', type=int, default=None,
                       help='指定要转换的trial索引列表 (默认: 全部)')
    parser.add_argument('--knee-coupling', action='store_true',
                       help='使用knee secondary motion耦合函数 (实验性)')
    
    args = parser.parse_args()
    
    converter_kwargs = {
        'coordinate_swap': args.coord_swap,
        'pelvis_pos_scale': args.pos_scale,
        'pelvis_pos_offset': np.array(args.pos_offset),
    }
    
    batch_convert(args.input, args.output, converter_kwargs, args.trials)
    
    print("\n下一步:")
    print("  1. 使用 motion_utils.py 播放和验证转换后的数据")
    print("  2. 使用 myosuite 加载数据用于运动模仿训练")
    print(f"\n  示例: python motion_utils.py {args.output}/*.npz")


if __name__ == "__main__":
    main()
