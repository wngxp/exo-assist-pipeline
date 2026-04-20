import nimblephysics as nimble
import numpy as np

path = "/home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/No_Arm/Camargo2021_Formatted_No_Arm/AB07_split5/AB07_split5.b3d"

subject = nimble.biomechanics.SubjectOnDisk(path)
skel = subject.readSkel(0, ignoreGeometry=True)

trial = 0

print("trial name:", subject.getTrialName(trial))
print("trial length:", subject.getTrialLength(trial))

# Try reading one frame
frame = subject.readFrames(trial, 0, 1)
print("num frames returned:", len(frame))

f0 = frame[0]
print("frame type:", type(f0))

# Inspect available attributes
print("frame attrs:", [a for a in dir(f0) if not a.startswith("_")][:50])

# Try common fields
for attr in ["pos", "vel", "tau"]:
    if hasattr(f0, attr):
        value = getattr(f0, attr)
        print(attr, type(value), np.shape(value))