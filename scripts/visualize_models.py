#!/usr/bin/env python3
"""
visualize_models.py
Load WAWA exoskeleton and Rajagopal human .osim models,
plot body frame origins in 3D to determine alignment offsets.

Usage (on cortex, conda activate opensim):
    python visualize_models.py
"""

import opensim as osim
import numpy as np
import matplotlib.pyplot as plt

# --- Paths (edit as needed) ---
WAWA_PATH = "/home/wxp/repos/WAWA/urdf/FES_urdf_0306.osim"
RAJAGOPAL_PATH = "/home/wxp/repos/exo-assist-pipeline/opensim/moco-tutorial/example3DWalking/subject_walk_scaled.osim"
def get_body_positions(model_path, label_prefix=""):
    """Get default-pose body frame origins from an .osim model."""
    model = osim.Model(model_path)
    state = model.initSystem()

    names, positions = [], []
    bodies = model.getBodySet()
    for i in range(bodies.getSize()):
        body = bodies.get(i)
        # Get position of body origin in ground frame
        pos = body.getPositionInGround(state)
        names.append(f"{label_prefix}{body.getName()}")
        positions.append([pos.get(0), pos.get(1), pos.get(2)])

    return names, np.array(positions)


def main():
    # Load both models
    h_names, h_pos = get_body_positions(RAJAGOPAL_PATH, label_prefix="H:")
    w_names, w_pos = get_body_positions(WAWA_PATH, label_prefix="W:")

    # Print bounding boxes
    print("=== Rajagopal (Human) ===")
    print(f"  X range: {h_pos[:,0].min():.3f} to {h_pos[:,0].max():.3f} m")
    print(f"  Y range: {h_pos[:,1].min():.3f} to {h_pos[:,1].max():.3f} m")
    print(f"  Z range: {h_pos[:,2].min():.3f} to {h_pos[:,2].max():.3f} m")
    for n, p in zip(h_names, h_pos):
        print(f"  {n:30s}  ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    print("\n=== WAWA (Exoskeleton) ===")
    print(f"  X range: {w_pos[:,0].min():.3f} to {w_pos[:,0].max():.3f} m")
    print(f"  Y range: {w_pos[:,1].min():.3f} to {w_pos[:,1].max():.3f} m")
    print(f"  Z range: {w_pos[:,2].min():.3f} to {w_pos[:,2].max():.3f} m")
    for n, p in zip(w_names, w_pos):
        print(f"  {n:30s}  ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")

    # --- 3D scatter plot ---
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Human bodies
    ax.scatter(h_pos[:,0], h_pos[:,2], h_pos[:,1],
               c='blue', s=40, label='Rajagopal (human)')
    for n, p in zip(h_names, h_pos):
        short = n.replace("H:", "")
        if short in ('pelvis', 'femur_r', 'femur_l', 'tibia_r', 'tibia_l', 'torso'):
            ax.text(p[0], p[2], p[1], f" {short}", fontsize=7, color='blue')

    # WAWA bodies
    ax.scatter(w_pos[:,0], w_pos[:,2], w_pos[:,1],
               c='red', s=60, marker='^', label='WAWA (exo)')
    for n, p in zip(w_names, w_pos):
        short = n.replace("W:", "")
        ax.text(p[0], p[2], p[1], f" {short}", fontsize=7, color='red')

    ax.set_xlabel('X (anterior +)')
    ax.set_ylabel('Z (right +)')
    ax.set_zlabel('Y (up +)')
    ax.set_title('Body frame origins — default pose')
    ax.legend()

    # Equal aspect ratio
    all_pos = np.vstack([h_pos, w_pos])
    max_range = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2
    mid = all_pos.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[2] - max_range, mid[2] + max_range)
    ax.set_zlim(mid[1] - max_range, mid[1] + max_range)

    plt.tight_layout()
    plt.savefig("model_alignment.png", dpi=150)
    print("\nSaved: model_alignment.png")
    plt.show()


if __name__ == "__main__":
    main()