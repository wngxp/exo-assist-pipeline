#!/usr/bin/env python3
"""
端到端演示: 从b3d动捕数据到MyoLeg参考运动的完整流程

这个脚本演示了:
1. 读取b3d文件
2. 转换为MyoLeg qpos格式
3. 在MyoLeg环境中播放
4. 导出训练数据
"""

import os
import sys
import numpy as np
import argparse

# 确保模块在路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from b3d_reader import B3DReader
from b3d_to_myoleg_converter import B3DToMyoLegConverter
from motion_utils import MyoLegReferenceMotion, compute_mimic_reward


def demo_read_convert(b3d_path: str, output_npz: str):
    """演示读取和转换"""
    print("=" * 60)
    print("步骤1: 读取b3d文件")
    print("=" * 60)
    
    reader = B3DReader(b3d_path)
    info = reader.get_subject_info(0)
    print(f"Subject信息: {info}")
    
    # 提取第一个trial
    trial_data = reader.extract_trial_data(0, 0)
    print(f"\nTrial数据维度:")
    print(f"  pos: {trial_data['pos'].shape}")
    print(f"  vel: {trial_data['vel'].shape}")
    print(f"  time: {trial_data['time'].shape}, duration={trial_data['time'][-1]:.2f}s")
    
    print("\n" + "=" * 60)
    print("步骤2: 转换为MyoLeg格式")
    print("=" * 60)
    
    converter = B3DToMyoLegConverter(
        coordinate_swap='xyz_to_xzy',
        pelvis_pos_scale=1.0,
    )
    
    converted = converter.convert_trial(trial_data)
    print(f"转换后数据:")
    print(f"  qpos: {converted['qpos'].shape}")
    print(f"  qvel: {converted['qvel'].shape}")
    print(f"  dt: {converted['dt']:.4f}s")
    
    # 保存
    converter.save_converted(output_npz, converted)
    print(f"\n已保存到: {output_npz}")
    
    return output_npz


def demo_play_in_myoleg(npz_path: str, render: bool = False, max_frames: int = None):
    """演示在MyoLeg中播放"""
    print("\n" + "=" * 60)
    print("步骤3: 在MyoLeg环境中播放")
    print("=" * 60)
    
    import myosuite
    import gymnasium as gym
    
    ref = MyoLegReferenceMotion(npz_path)
    
    # 创建环境
    render_mode = 'human' if render else None
    env = gym.make('myoLegWalk-v0', render_mode=render_mode)
    obs, info = env.reset()
    
    # 设置初始状态
    env.unwrapped.sim.data.qpos[:] = ref.qpos[0]
    env.unwrapped.sim.data.qvel[:] = ref.qvel[0]
    env.unwrapped.sim.forward()
    
    # 播放
    frames_to_play = min(len(ref), max_frames) if max_frames else len(ref)
    total_reward = 0
    
    for t in range(frames_to_play):
        # 开环: 直接将参考状态设为环境状态
        env.unwrapped.sim.data.qpos[:] = ref.qpos[t]
        env.unwrapped.sim.data.qvel[:] = ref.qvel[t]
        env.unwrapped.sim.forward()
        
        if render:
            env.render()
        
        # 计算模仿奖励 (用于验证)
        env_qpos = env.unwrapped.sim.data.qpos.copy()
        env_qvel = env.unwrapped.sim.data.qvel.copy()
        ref_qpos, ref_qvel = ref.get_frame(t)
        reward = compute_mimic_reward(env_qpos, ref_qpos, env_qvel, ref_qvel)
        total_reward += reward
        
        if t % 100 == 0:
            print(f"  Frame {t}/{frames_to_play}, mimic_reward={reward:.4f}")
    
    env.close()
    print(f"\n播放完成: {frames_to_play} frames")
    print(f"平均模仿奖励: {total_reward / frames_to_play:.4f}")


def demo_export_for_training(npz_path: str, output_dir: str):
    """演示导出训练数据"""
    print("\n" + "=" * 60)
    print("步骤4: 导出训练数据")
    print("=" * 60)
    
    from motion_utils import export_for_training
    export_for_training(npz_path, output_dir, split_ratio=0.9)


def main():
    parser = argparse.ArgumentParser(description='b3d -> MyoLeg 端到端演示')
    parser.add_argument('b3d_path', help='输入b3d文件路径')
    parser.add_argument('--output-dir', default='./demo_output', help='输出目录')
    parser.add_argument('--render', action='store_true', help='是否渲染')
    parser.add_argument('--max-frames', type=int, default=None, help='最大播放帧数')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤1&2: 读取和转换
    npz_path = os.path.join(args.output_dir, 'converted.npz')
    demo_read_convert(args.b3d_path, npz_path)
    
    # 步骤3: 播放
    demo_play_in_myoleg(npz_path, render=args.render, max_frames=args.max_frames)
    
    # 步骤4: 导出训练数据
    train_dir = os.path.join(args.output_dir, 'training_data')
    demo_export_for_training(npz_path, train_dir)
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print("\n下一步:")
    print("  1. 在RL训练中加载 npz 文件作为参考运动")
    print("  2. 使用 motion_utils.MyLegReferenceMotion 接口获取帧数据")
    print("  3. 使用 compute_mimic_reward 计算模仿奖励")


if __name__ == "__main__":
    main()
