#!/usr/bin/env python3
"""
简单脚本：读取 b3d 文件，转换为 MyoLeg 格式，并在 MuJoCo 中播放。
"""

import sys
import os

# 确保能导入当前目录下的模块（假设脚本与 b3d_reader.py 等在同一目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from b3d_reader import B3DReader
from b3d_to_myoleg_converter import B3DToMyoLegConverter
from motion_utils import play_reference_motion_in_myoleg

def main():
    # 您的 b3d 文件路径
    b3d_path = "/home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/No_Arm/Camargo2021_Formatted_No_Arm/AB07_split5/AB07_split5.b3d"
    
    print("1. 读取 b3d 文件...")
    reader = B3DReader(b3d_path)
    
    # 获取第一个 subject 的第一个 trial 数据
    trial_data = reader.extract_trial_data(subject_idx=0, trial=0, processing_pass=1)  # 使用 processing_pass=1 (逆运动学结果)
    print(f"   数据形状: pos={trial_data['pos'].shape}, vel={trial_data['vel'].shape}")
    
    print("2. 转换为 MyoLeg 格式...")
    converter = B3DToMyoLegConverter(
        coordinate_swap='xyz_to_xzy',   # 坐标轴交换策略，可根据需要改为 'xyz_to_xz_neg_y'
        pelvis_pos_scale=1.0,
        pelvis_pos_offset=[0.0, 0.0, 0.0]
    )
    converted = converter.convert_trial(trial_data)
    print(f"   转换后 qpos 形状: {converted['qpos'].shape}, qvel 形状: {converted['qvel'].shape}")
    
    # 保存 npz 文件到当前脚本所在的目录（即 b3d_to_myoleg 文件夹下）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_npz = os.path.join(current_dir, "converted_trial.npz")
    converter.save_converted(tmp_npz, converted)
    
    print("3. 在 MyoLeg 环境中播放...")
    # 调用 motion_utils 中的播放函数
    play_reference_motion_in_myoleg(tmp_npz, render=True)
    
    print("播放结束。")

if __name__ == "__main__":
    main()