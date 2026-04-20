import nimblephysics as nimble
import numpy as np

path = "/home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/No_Arm/Camargo2021_Formatted_No_Arm/AB07_split5/AB07_split5.b3d"

subject = nimble.biomechanics.SubjectOnDisk(path)
print("loaded:", path)
print("num trials:", subject.getNumTrials())

skel = subject.readSkel(0, ignoreGeometry=True)
dof_names = [skel.getDofByIndex(i).getName() for i in range(skel.getNumDofs())]

print("\nDOF indices:")
for i, name in enumerate(dof_names):
    print(i, name)

# Print trial metadata
print("\nTrial summary:")
for trial in range(min(subject.getNumTrials(), 10)):
    try:
        trial_name = subject.getTrialName(trial)
    except Exception:
        trial_name = f"trial_{trial}"

    try:
        length = subject.getTrialLength(trial)
    except Exception:
        length = "unknown"

    print(f"trial {trial}: name={trial_name}, length={length}")