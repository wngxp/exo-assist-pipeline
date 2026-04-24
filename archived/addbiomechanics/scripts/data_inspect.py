import nimblephysics as nimble

path = "/home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/No_Arm/Camargo2021_Formatted_No_Arm/AB07_split5/AB07_split5.b3d"

subject = nimble.biomechanics.SubjectOnDisk(path)

print("loaded:", path)
print("num trials:", subject.getNumTrials())
print("num dofs:", subject.readSkel(0, ignoreGeometry=True).getNumDofs())
print("dof names:")
skel = subject.readSkel(0, ignoreGeometry=True)
for i in range(skel.getNumDofs()):
    print(i, skel.getDofByIndex(i).getName())