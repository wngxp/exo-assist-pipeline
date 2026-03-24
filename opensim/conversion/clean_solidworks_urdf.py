#!/usr/bin/env python3
"""
clean_solidworks_urdf.py — Clean up a SolidWorks URDF export for ROS 2.

Fixes:
  - Spaces in robot name, link names, joint names, mesh filenames
  - Zero joint limits (lower=upper=0) → ±180°
  - Zero effort/velocity → reasonable defaults
  - Zero mass/inertia → small default values
  - Renames mesh files on disk to match

Usage:
    python3 clean_solidworks_urdf.py /path/to/export_folder

The script modifies files in-place and renames mesh files.
"""

import os
import sys
import re
import shutil


def sanitize(name):
    """Replace spaces and special chars with underscores."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', name)


def clean_urdf(export_dir):
    # Find URDF file
    urdf_files = []
    for root, dirs, files in os.walk(export_dir):
        for f in files:
            if f.endswith('.urdf'):
                urdf_files.append(os.path.join(root, f))

    if not urdf_files:
        print("ERROR: No .urdf file found.")
        sys.exit(1)

    urdf_path = urdf_files[0]
    print(f"Cleaning: {urdf_path}\n")

    with open(urdf_path, 'r') as f:
        content = f.read()

    # --- Collect all names that need fixing ---
    # Find robot name
    robot_match = re.search(r'<robot\s+name="([^"]*)"', content)
    if robot_match:
        old_robot_name = robot_match.group(1)
        new_robot_name = sanitize(old_robot_name)
        if old_robot_name != new_robot_name:
            print(f"  Robot: '{old_robot_name}' → '{new_robot_name}'")

    # Find all link names
    link_names = re.findall(r'<link\s+name="([^"]*)"', content)
    # Find all joint names
    joint_names = re.findall(r'<joint\s+name="([^"]*)"', content)
    # Find all mesh filenames
    mesh_refs = re.findall(r'filename="package://[^"]*?/meshes/([^"]*)"', content)

    # Build rename map
    rename_map = {}
    for name in set(link_names + joint_names):
        clean = sanitize(name)
        if clean != name:
            rename_map[name] = clean
            print(f"  Name: '{name}' → '{clean}'")

    mesh_rename_map = {}
    for mesh in set(mesh_refs):
        clean = sanitize(mesh)
        if clean != mesh:
            mesh_rename_map[mesh] = clean
            print(f"  Mesh: '{mesh}' → '{clean}'")

    # --- Apply fixes to URDF content ---

    # Fix robot name
    old_pkg = old_robot_name
    new_pkg = new_robot_name
    content = content.replace(f'name="{old_robot_name}"', f'name="{new_robot_name}"', 1)

    # Fix package:// references (old package name → new)
    content = content.replace(f'package://{old_pkg}/', f'package://{new_pkg}/')

    # Fix link and joint name references
    # We need to be careful to replace inside attribute values only
    for old_name, new_name in rename_map.items():
        # Replace in name="...", link="...", etc.
        content = content.replace(f'"{old_name}"', f'"{new_name}"')

    # Fix mesh filenames in package:// paths
    for old_mesh, new_mesh in mesh_rename_map.items():
        content = content.replace(f'meshes/{old_mesh}', f'meshes/{new_mesh}')

    # Fix zero joint limits
    # Match: lower="0" upper="0" (with possible .0 variants)
    content = re.sub(
        r'lower="0(?:\.0)?" upper="0(?:\.0)?"',
        'lower="-3.1416" upper="3.1416"',
        content
    )

    # Fix zero effort/velocity
    content = re.sub(
        r'effort="0(?:\.0)?"',
        'effort="10"',
        content
    )
    content = re.sub(
        r'velocity="0(?:\.0)?"',
        'velocity="3.14"',
        content
    )

    # Fix zero mass (set to small value)
    content = re.sub(
        r'<mass\s+value="0(?:\.0)?" />',
        '<mass value="0.001" />',
        content
    )

    # Fix zero inertia (set to small values)
    def fix_zero_inertia(match):
        text = match.group(0)
        # Check if all values are zero
        values = re.findall(r'i[xyz]{2}="([^"]*)"', text)
        if all(float(v) == 0 for v in values):
            return text.replace('ixx="0"', 'ixx="1e-9"') \
                       .replace('iyy="0"', 'iyy="1e-9"') \
                       .replace('izz="0"', 'izz="1e-9"')
        return text

    content = re.sub(r'<inertia[^/]*/>', fix_zero_inertia, content)

    # --- Rename URDF file itself ---
    urdf_dir = os.path.dirname(urdf_path)
    old_urdf_basename = os.path.basename(urdf_path)
    new_urdf_basename = sanitize(old_urdf_basename)
    if new_urdf_basename != old_urdf_basename:
        print(f"  URDF file: '{old_urdf_basename}' → '{new_urdf_basename}'")

    new_urdf_path = os.path.join(urdf_dir, new_urdf_basename)
    with open(new_urdf_path, 'w') as f:
        f.write(content)

    if new_urdf_path != urdf_path:
        os.remove(urdf_path)

    # --- Rename mesh files on disk ---
    meshes_dir = os.path.join(export_dir, 'meshes')
    if os.path.isdir(meshes_dir):
        for old_mesh, new_mesh in mesh_rename_map.items():
            old_path = os.path.join(meshes_dir, old_mesh)
            new_path = os.path.join(meshes_dir, new_mesh)
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f"  Renamed file: {old_mesh} → {new_mesh}")

    print(f"\nDone! Cleaned URDF saved to: {new_urdf_path}")
    print(f"\nNow run:")
    print(f"  ~/repos/urdf_to_osim/solidworks_to_ros2.sh {export_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 clean_solidworks_urdf.py /path/to/export_folder")
        sys.exit(1)

    clean_urdf(sys.argv[1])
