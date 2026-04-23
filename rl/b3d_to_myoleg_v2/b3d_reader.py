import os
import glob
import numpy as np
from typing import List, Dict, Tuple, Optional

import nimblephysics as nimble


class B3DReader:
    """
    读取 AddBiomechanics b3d 格式动捕数据集，支持批量读取和数据提取。

    b3d 文件包含：
    - 多 trial 数据
    - 每个 trial 有多帧 Frame 数据
    - 每帧包含 pos (DoF位置/角度), vel (DoF速度), tau (DoF力矩)
    - 骨架基于 Rajagopal 模型，但可能只有下肢 (23 DoF) 或完整 (37 DoF)
    """

    def __init__(self, b3d_path: str):
        """
        Args:
            b3d_path: 单个 .b3d 文件路径，或包含多个 .b3d 文件的文件夹路径
        """
        self.subjects: List[nimble.biomechanics.SubjectOnDisk] = []
        self.paths: List[str] = []

        if os.path.isfile(b3d_path):
            if not b3d_path.endswith('.b3d'):
                raise ValueError(f"文件必须是 .b3d 格式: {b3d_path}")
            self._load_single(b3d_path)
        elif os.path.isdir(b3d_path):
            b3d_files = glob.glob(os.path.join(b3d_path, "**", "*.b3d"), recursive=True)
            for f in sorted(b3d_files):
                self._load_single(f)
        else:
            raise FileNotFoundError(f"路径不存在: {b3d_path}")

        if len(self.subjects) == 0:
            raise ValueError("未找到任何 .b3d 文件")

        # 读取第一个 subject 的骨架以获取 DoF 名称
        self.skel = self.subjects[0].readSkel(
            processingPass=0,
            ignoreGeometry=True
        )
        self.dof_names = [self.skel.getDofByIndex(i).getName()
                          for i in range(self.skel.getNumDofs())]
        self.n_dofs = self.skel.getNumDofs()

        # 动态构建名称到索引的映射
        self.dof_name_to_idx = {name: i for i, name in enumerate(self.dof_names)}

        print(f"已加载 {len(self.subjects)} 个 subject，骨架共 {self.n_dofs} DoF")
        print(f"DoF 列表: {self.dof_names}")

    def _load_single(self, path: str):
        """加载单个 b3d 文件"""
        try:
            subject = nimble.biomechanics.SubjectOnDisk(path)
            self.subjects.append(subject)
            self.paths.append(path)
        except Exception as e:
            print(f"警告: 无法加载 {path}: {e}")

    def get_subject_info(self, subject_idx: int = 0) -> Dict:
        """获取 subject 元数据"""
        subj = self.subjects[subject_idx]
        return {
            'height_m': subj.getHeightM(),
            'mass_kg': subj.getMassKg(),
            'num_trials': subj.getNumTrials(),
            'href': subj.getHref(),
            'path': self.paths[subject_idx],
        }

    def get_trial_info(self, subject_idx: int = 0, trial: int = 0) -> Dict:
        """获取 trial 信息"""
        subj = self.subjects[subject_idx]
        return {
            'length': subj.getTrialLength(trial),
            'timestep': subj.getTrialTimestep(trial),
            'num_processing_passes': subj.getTrialNumProcessingPasses(trial),
        }

    def read_trial_frames(self, subject_idx: int = 0, trial: int = 0, start_frame: int = 0, num_frames: Optional[int] = None):
        """
        读取指定trial的帧数据（包含所有processing passes）
        """
        subj = self.subjects[subject_idx]
        if num_frames is None:
            num_frames = subj.getTrialLength(trial)
        frames = subj.readFrames(
            trial=trial,
            startFrame=start_frame,
            numFramesToRead=num_frames,
            includeSensorData=True,
            includeProcessingPasses=True
        )
        return frames 

    def extract_trial_data(self, subject_idx: int = 0, trial: int = 0,
                       processing_pass: int = 0) -> Dict[str, np.ndarray]:
        subj = self.subjects[subject_idx]
        length = subj.getTrialLength(trial)
        dt = subj.getTrialTimestep(trial)
        n_dofs = self.n_dofs

        pos = np.zeros((length, n_dofs))
        vel = np.zeros((length, n_dofs))
        tau = np.zeros((length, n_dofs))
        acc = np.zeros((length, n_dofs))

        com_pos = np.zeros((length, 3))
        com_vel = np.zeros((length, 3))
        com_acc = np.zeros((length, 3))

        frames = self.read_trial_frames(subject_idx, trial, 0, length)

        for t, frame in enumerate(frames):
            pass_data = frame.processingPasses[processing_pass]
            pos[t] = pass_data.pos
            vel[t] = pass_data.vel
            tau[t] = pass_data.tau
            acc[t] = pass_data.acc

            com_pos[t] = pass_data.comPos
            com_vel[t] = pass_data.comVel
            com_acc[t] = pass_data.comAcc

        time = np.arange(length) * dt

        data = {
            'pos': pos,
            'vel': vel,
            'tau': tau,
            'acc': acc,
            'time': time,
            'dt': dt,
            'com_pos': com_pos,
            'com_vel': com_vel,
            'com_acc': com_acc,
            'dof_names': self.dof_names,
            'dof_name_to_idx': self.dof_name_to_idx,
            'n_dofs': n_dofs,
        }

        try:
            contact_bodies = subj.getGroundForceBodies()
            contact = np.zeros((length, len(contact_bodies)))
            for t, frame in enumerate(frames):
                pass_data = frame.processingPasses[processing_pass]
                contact[t] = pass_data.contact.astype(float)
            data['contact'] = contact
            data['contact_bodies'] = contact_bodies
        except:
            pass

        return data

    def extract_all_trials(self, subject_idx: int = 0,
                           processing_pass: int = 0) -> List[Dict]:
        """提取一个 subject 的所有 trial 数据"""
        subj = self.subjects[subject_idx]
        results = []
        for trial in range(subj.getNumTrials()):
            print(f"  提取 trial {trial}/{subj.getNumTrials()} ...")
            trial_data = self.extract_trial_data(subject_idx, trial, processing_pass)
            trial_info = self.get_trial_info(subject_idx, trial)
            trial_data['trial_info'] = trial_info
            results.append(trial_data)
        return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        reader = B3DReader(path)

        info = reader.get_subject_info(0)
        print(f"\nSubject 信息:")
        for k, v in info.items():
            print(f"  {k}: {v}")

        for trial in range(info['num_trials']):
            t_info = reader.get_trial_info(0, trial)
            print(f"\nTrial {trial}: {t_info['length']} frames, dt={t_info['timestep']:.4f}s")

        if info['num_trials'] > 0:
            data = reader.extract_trial_data(0, 0)
            print(f"\n数据维度:")
            print(f"  pos: {data['pos'].shape}")
            print(f"  vel: {data['vel'].shape}")
            print(f"  tau: {data['tau'].shape}")
            print(f"  com_pos: {data['com_pos'].shape}")
    else:
        print("用法: python b3d_reader.py <b3d 文件或文件夹路径>")
