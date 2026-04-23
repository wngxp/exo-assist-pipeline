#!/usr/bin/env python3
"""
从转换好的 npz 文件生成 MP4 视频（无头渲染）
"""

import os
import numpy as np
import mujoco
import imageio
from tqdm import tqdm

def main():
    # ====== 配置参数 ======
    script_dir = os.path.dirname(os.path.abspath(__file__))
    npz_file = os.path.join(script_dir, "converted_trial.npz")
    output_video = os.path.join(script_dir, "motion_video.mp4")   # 视频也保存到同一目录
    fps = 30
    height, width = 720, 1280
    camera_name = "track"                # 相机名称（可选 "side" 或 "track"）
    # =====================

    # 1. 获取 MyoLeg 模型 XML 路径（自动从 myosuite 安装目录查找）
    import pkg_resources
    try:
        xml_path = "/home/wxp/miniconda3/envs/b3d_convert/lib/python3.9/site-packages/myosuite/simhive/myo_sim/leg/myolegs.xml"
        print(f"使用模型: {xml_path}")
    except Exception as e:
        print("自动查找失败，请手动指定 xml_path")
        print(e)
        return

    # 2. 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # 3. 加载参考运动数据
    if not os.path.exists(npz_file):
        print(f"错误: 找不到文件 {npz_file}")
        return
    data_npz = np.load(npz_file)
    qpos_ref = data_npz['qpos']   # (T, 35)
    qvel_ref = data_npz['qvel']   # (T, 34)
    n_frames = qpos_ref.shape[0]
    print(f"总帧数: {n_frames}")

    # 4. 获取相机 ID
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id == -1:
        print(f"警告: 未找到相机 '{camera_name}'，使用相机 0")
        camera_id = 0
        # 列出所有可用相机
        print("可用的相机:")
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  {i}: {name}")

    # 5. 设置无头渲染后端
    os.environ['MUJOCO_GL'] = 'osmesa'   # 使用 CPU 渲染，稳定可靠

    # 6. 创建渲染器
    renderer = mujoco.Renderer(model, height, width)

    frames = []
    print("开始渲染...")
    for t in tqdm(range(n_frames)):
        data.qpos[:] = qpos_ref[t]
        data.qvel[:] = qvel_ref[t]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_id)
        pixels = renderer.render()
        frames.append(pixels)

    renderer.close()

    # 7. 保存视频
    print(f"正在保存视频到 {output_video} ...")
    with imageio.get_writer(output_video, fps=fps, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print("视频生成完成！")

if __name__ == "__main__":
    main()