import nimblephysics as nimble

path = "data/addbiomechanics/train/No_Arm/vanderZee2022_Formatted_No_Arm/p1/p1.b3d"

dataset = nimble.biomechanics.SubjectOnDisk(path)

print("Num trials:", dataset.getNumTrials())

trial = 0
print("Trial length:", dataset.getTrialLength(trial))

skel = dataset.readSkel()

trial = 0
num_frames = dataset.getTrialLength(trial)

positions = []
velocities = []

for t in range(num_frames):
    frame = dataset.readFrames(trial, t, 1)[0]
    
    positions.append(frame.pos)  # joint angles
    velocities.append(frame.vel)

import numpy as np
positions = np.array(positions)
velocities = np.array(velocities)

print(positions.shape)