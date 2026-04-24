"""
调试脚本：验证MyoLeg qpos结构，帮助理解坐标系

用法:
    python debug_myoleg.py
"""

import numpy as np
import myosuite
import gymnasium as gym


def inspect_myoleg_model():
    """详细检查MyoLeg模型结构"""
    env = gym.make('myoLegWalk-v0')
    model = env.unwrapped.sim.model
    
    print("=" * 60)
    print("MyoLeg 模型结构")
    print("=" * 60)
    print(f"nq (qpos维度): {model.nq}")
    print(f"nv (qvel维度): {model.nv}")
    print(f"nu (actuator维度): {model.nu}")
    print(f"nbody: {model.nbody}")
    print(f"njnt: {model.njnt}")
    print(f"nmocap: {model.nmocap}")
    
    print("\n--- 关节列表 ---")
    print(f"{'Idx':<5} {'Name':<40} {'Type':<8} {'QposAdr':<9} {'DofAdr':<8} {'Range'}")
    print("-" * 90)
    
    for i in range(model.njnt):
        jnt = model.joint(i)
        jnt_type = {0: 'free', 1: 'ball', 2: 'slide', 3: 'hinge'}[model.jnt_type[i]]
        qposadr = model.jnt_qposadr[i]
        dofadr = model.jnt_dofadr[i]
        
        # range
        if model.jnt_type[i] == 0:  # free
            rng = "N/A"
        else:
            lo, hi = model.jnt_range[i]
            rng = f"[{lo:7.3f}, {hi:7.3f}]"
        
        print(f"{i:<5} {jnt.name:<40} {jnt_type:<8} {qposadr:<9} {dofadr:<8} {rng}")
    
    print("\n--- qpos 完整映射 ---")
    for i in range(model.nq):
        # 找到对应的joint
        for j in range(model.njnt):
            qadr = model.jnt_qposadr[j]
            jtype = model.jnt_type[j]
            dim = 7 if jtype == 0 else (4 if jtype == 1 else 1)
            if qadr <= i < qadr + dim:
                offset = i - qadr
                names = {0: ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'],
                        1: ['qw', 'qx', 'qy', 'qz'],
                        2: ['slide'],
                        3: ['hinge']}[jtype]
                print(f"  qpos[{i:2d}] = {model.joint(j).name:<40} [{names[offset]}]")
                break
    
    print("\n--- qvel 完整映射 ---")
    for i in range(model.nv):
        for j in range(model.njnt):
            dadr = model.jnt_dofadr[j]
            jtype = model.jnt_type[j]
            dim = 6 if jtype == 0 else (3 if jtype == 1 else 1)
            if dadr <= i < dadr + dim:
                offset = i - dadr
                if jtype == 0:
                    names = ['ang_x', 'ang_y', 'ang_z', 'vel_x', 'vel_y', 'vel_z']
                elif jtype == 1:
                    names = ['ang_x', 'ang_y', 'ang_z']
                else:
                    names = ['vel']
                print(f"  qvel[{i:2d}] = {model.joint(j).name:<40} [{names[offset]}]")
                break
    
    print("\n--- 初始KeyFrame ---")
    for k in range(min(3, model.nkey)):
        key = model.key(k)
        qpos = model.key_qpos[k * model.nq : (k+1) * model.nq]
        print(f"\nKeyframe '{key.name}':")
        print(f"  qpos[0:7] (root): pos={qpos[:3]}, quat={qpos[3:7]}")
        print(f"  qpos[7:21] (right leg): {qpos[7:21]}")
        print(f"  qpos[21:35] (left leg): {qpos[21:35]}")
    
    env.close()


def test_random_qpos():
    """测试设置随机qpos的效果"""
    env = gym.make('myoLegWalk-v0', render_mode=None)
    
    # 获取keyframe作为参考
    model = env.unwrapped.sim.model
    if model.nkey > 0:
        key = model.keyframe(0)
        key_qpos = key.qpos.copy()
    else:
        key_qpos = model.qpos0.copy()
    
    print("\n--- 测试默认keyframe ---")
    env.unwrapped.sim.data.qpos[:] = key_qpos
    env.unwrapped.sim.forward()
    
    # 获取pelvis位置
    pelvis_id = model.body('pelvis').id
    pelvis_pos = env.unwrapped.sim.data.body_xpos[pelvis_id]
    print(f"Pelvis world position: {pelvis_pos}")
    
    # 获取左右脚位置
    try:
        r_foot_id = model.body('toes_r').id
        l_foot_id = model.body('toes_l').id
        r_foot_pos = env.unwrapped.sim.data.body_xpos[r_foot_id]
        l_foot_pos = env.unwrapped.sim.data.body_xpos[l_foot_id]
        print(f"Right foot world position: {r_foot_pos}")
        print(f"Left foot world position: {l_foot_pos}")
    except:
        pass
    
    env.close()
    return key_qpos


def test_root_rotation():
    """测试root rotation的不同表示，帮助理解坐标系"""
    env = gym.make('myoLegWalk-v0', render_mode=None)
    model = env.unwrapped.sim.model
    
    # 默认keyframe
    if model.nkey > 0:
        key = model.keyframe(0)
        key_qpos = key.qpos.copy()
    else:
        key_qpos = model.qpos0.copy()
    
    print("\n--- Root Rotation 测试 ---")
    print(f"默认keyframe root quat: {key_qpos[3:7]}")
    
    # 测试identity rotation
    key_qpos[3:7] = [1, 0, 0, 0]
    env.unwrapped.sim.data.qpos[:] = key_qpos
    env.unwrapped.sim.forward()
    pelvis_id = model.body('pelvis').id
    pelvis_xmat = env.unwrapped.sim.data.body('pelvis').xmat.reshape(3, 3)
    print(f"\nIdentity root quat -> pelvis rot matrix:\n{pelvis_xmat}")
    
    # 测试绕Z轴90度
    angle = np.pi / 2
    key_qpos[3:7] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]  # 绕Z
    env.unwrapped.sim.data.qpos[:] = key_qpos
    env.unwrapped.sim.forward()
    pelvis_xmat = env.unwrapped.sim.data.body('pelvis').xmat.reshape(3, 3)
    print(f"\nRot Z 90° -> pelvis rot matrix:\n{pelvis_xmat}")
    
    # 测试绕X轴90度
    angle = np.pi / 2
    key_qpos[3:7] = [np.cos(angle/2), np.sin(angle/2), 0, 0]  # 绕X
    env.unwrapped.sim.data.qpos[:] = key_qpos
    env.unwrapped.sim.forward()
    pelvis_xmat = env.unwrapped.sim.data.body('pelvis').xmat.reshape(3, 3)
    print(f"\nRot X 90° -> pelvis rot matrix:\n{pelvis_xmat}")
    
    env.close()


def test_body_chain():
    """测试身体链的运动学关系"""
    env = gym.make('myoLegWalk-v0', render_mode=None)
    model = env.unwrapped.sim.model
    data = env.unwrapped.sim.data
    
    # 设置默认keyframe
    if model.nkey > 0:
        key = model.keyframe(0)
        key_qpos = key.qpos.copy()
    else:
        key_qpos = model.qpos0.copy()
    data.qpos[:] = key_qpos
    env.unwrapped.sim.forward()
    
    print("\n--- 身体链位置 (keyframe) ---")
    body_names = ['pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
                  'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l']
    
    for name in body_names:
        try:
            bid = model.body(name).id
            pos = data.body_xpos[bid]
            print(f"  {name:<15} world_pos: [{pos[0]:8.4f}, {pos[1]:8.4f}, {pos[2]:8.4f}]")
        except:
            pass
    
    # 测试knee angle变化的影响
    print("\n--- Knee Angle 变化测试 ---")
    for knee_val in [0.0, 0.5, 1.0]:
        key_qpos[12] = knee_val  # knee_angle_r
        data.qpos[:] = key_qpos
        env.unwrapped.sim.forward()
        
        tibia_pos = data.body_xpos[model.body('tibia_r').id]
        print(f"  knee_angle_r={knee_val:.1f} -> tibia_r pos: [{tibia_pos[0]:.4f}, {tibia_pos[1]:.4f}, {tibia_pos[2]:.4f}]")
    
    env.close()


if __name__ == "__main__":
    print("MyoLeg 调试脚本")
    print("=" * 60)
    
    inspect_myoleg_model()
    key_qpos = test_random_qpos()
    test_root_rotation()
    test_body_chain()
    
    print("\n" + "=" * 60)
    print("调试完成。使用以上信息来校准 b3d -> myoleg 坐标转换。")
    print("=" * 60)
