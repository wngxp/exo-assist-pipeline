import numpy as np
import matplotlib.pyplot as plt

# Read the .sto file, skipping the header
with open('muscle_driven_state_tracking_tracked_states.sto') as f:
    for i, line in enumerate(f):
        if line.strip() == 'endheader':
            header_lines = i + 1
            break

data = np.genfromtxt('muscle_driven_state_tracking_tracked_states.sto', 
                      skip_header=header_lines, names=True, delimiter='\t',
                      deletechars='')

# Get time and hip flexion (radians -> degrees)
time = data[data.dtype.names[0]]
hip_r = np.degrees(data['/jointset/hip_r/hip_flexion_r/value'])
hip_l = np.degrees(data['/jointset/hip_l/hip_flexion_l/value'])

# Normalize to gait cycle percentage
gait_pct = (time - time[0]) / (time[-1] - time[0]) * 100

plt.figure(figsize=(10, 5))
plt.plot(gait_pct, hip_r, label='Right Hip', linewidth=2)
plt.plot(gait_pct, hip_l, label='Left Hip', linewidth=2)
plt.xlabel('Gait Cycle (%)')
plt.ylabel('Hip Flexion Angle (degrees)')
plt.title('Hip Flexion - MocoTrack 3D Walking (Muscle-Driven State Tracking)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hip_flexion_plot.png', dpi=150)
plt.show()
print("Saved to hip_flexion_plot.png")
