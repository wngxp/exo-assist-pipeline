import nimblephysics as nimble

path = "data/addbiomechanics/train/No_Arm/vanderZee2022_Formatted_No_Arm/p1/p1.b3d"

dataset = nimble.biomechanics.SubjectOnDisk(path)

print("Num trials:", dataset.getNumTrials())
trial = 0
print("Trial length:", dataset.getTrialLength(trial))

# usually final processed pass = last pass
num_passes = dataset.getNumProcessingPasses()
print("Num processing passes:", num_passes)

skel = dataset.readSkel(num_passes - 1)
print("Num DOFs:", skel.getNumDofs())

for i in range(skel.getNumDofs()):
    print(i, skel.getDof(i).getName())
    
frame = dataset.readFrames(trial, 0, 1, num_passes - 1)[0]
print("pos shape:", len(frame.pos))
print("vel shape:", len(frame.vel))