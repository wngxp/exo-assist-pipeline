import nimblephysics as nimble
import numpy as np

path = "/home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/No_Arm/Camargo2021_Formatted_No_Arm/AB07_split5/AB07_split5.b3d"

subject = nimble.biomechanics.SubjectOnDisk(path)
trial = 0

print("trial name:", subject.getTrialName(trial))
print("trial length:", subject.getTrialLength(trial))

frames = subject.readFrames(trial, 0, 1)
f0 = frames[0]

print("frame attrs:", [a for a in dir(f0) if not a.startswith("_")])
print("timestamp:", f0.t)

pp = f0.processingPasses
print("processingPasses type:", type(pp))
print("num processing passes:", len(pp))

for i, p in enumerate(pp):
    print(f"\n=== pass {i} ===")
    attrs = [a for a in dir(p) if not a.startswith("_")]
    print(attrs[:100])

    for attr in [
        "pos", "positions",
        "vel", "velocities",
        "acc", "accelerations",
        "tau", "torques",
        "groundContactWrenches"
    ]:
        if hasattr(p, attr):
            value = getattr(p, attr)
            print(attr, type(value), np.shape(value))