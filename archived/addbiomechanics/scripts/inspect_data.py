import nimblephysics as nimble

path = "data/addbiomechanics/train/No_Arm/vanderZee2022_Formatted_No_Arm/p1/p1.b3d"

dataset = nimble.biomechanics.SubjectOnDisk(path)

print("Num trials:", dataset.getNumTrials())
trial = 0
print("Trial length:", dataset.getTrialLength(trial))

num_passes = dataset.getNumProcessingPasses()
print("Num processing passes:", num_passes)

skel = dataset.readSkel(num_passes - 1, ignoreGeometry=True)
print("Num DOFs:", skel.getNumDofs())
print("Num joints:", skel.getNumJoints())

print("\nJOINTS:")
for i in range(skel.getNumJoints()):
    joint = skel.getJoint(i)
    print(i, joint.getName())

print("\nDOF NAMES:")
dofs = skel.getDofs()
for i, dof in enumerate(dofs):
    print(i, dof.getName())
    
frame = dataset.readFrames(trial, 0, 1, num_passes - 1)[0]

print("num processing passes on frame:", len(frame.processingPasses))
pp = frame.processingPasses[num_passes - 1]

print(type(pp))
print(dir(pp))

for name in dir(pp):
    if not name.startswith("_"):
        print(name)