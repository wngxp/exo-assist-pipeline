#!/usr/bin/env python3
"""
convert_urdf_to_osim.py — Convert a URDF file to OpenSim .osim format.

Handles:
  - Revolute joints → PinJoint (with axis remapping via orientation offsets)
  - Fixed joints → WeldJoint
  - Continuous joints → PinJoint (no range limits)
  - Mass, inertia, and mesh geometry

Usage:
    conda activate opensim
    python convert_urdf_to_osim.py input.urdf [output.osim]

If output is not specified, it will use the URDF filename with .osim extension.

Requirements:
    - conda env with opensim 4.5+ (Python 3.12)
    - Setup: conda create -n opensim python=3.12 && conda activate opensim
             conda install -c opensim-org opensim

Note on meshes:
    - URDF typically uses .STL meshes; OpenSim requires .vtp (ASCII format)
    - This script references .vtp files with the same base name as the .STL
    - Convert STL → VTP separately using convert_stl_to_vtp_simbody.py
"""

import xml.etree.ElementTree as ET
import opensim as osim
import sys
import os
import math
import numpy as np


def parse_vec3(text):
    """Parse a space-separated string of 3 floats into an osim.Vec3."""
    vals = [float(x) for x in text.strip().split()]
    return osim.Vec3(vals[0], vals[1], vals[2])


def parse_inertia(inertial_elem):
    """Parse URDF <inertia> element into OpenSim Inertia object."""
    inertia = inertial_elem.find("inertia")
    return osim.Inertia(
        float(inertia.get("ixx", 0)),
        float(inertia.get("iyy", 0)),
        float(inertia.get("izz", 0)),
        float(inertia.get("ixy", 0)),
        float(inertia.get("ixz", 0)),
        float(inertia.get("iyz", 0)),
    )


def axis_to_orientation(axis_xyz):
    """
    Compute the orientation offset (XYZ body-fixed Euler angles) needed
    to rotate the PinJoint's default Z-axis to the desired axis.
    Handles arbitrary axes, not just principal ones.
    """
    ax = np.array([float(v) for v in axis_xyz.strip().split()])

    # Normalize
    length = np.linalg.norm(ax)
    if length < 1e-10:
        return osim.Vec3(0, 0, 0)
    ax = ax / length

    # Target: rotate Z-axis [0,0,1] to ax
    z = np.array([0.0, 0.0, 1.0])

    # If already aligned with Z
    if np.allclose(ax, z, atol=1e-6):
        return osim.Vec3(0, 0, 0)
    # If anti-aligned with Z
    if np.allclose(ax, -z, atol=1e-6):
        return osim.Vec3(math.pi, 0, 0)

    # Rotation axis = cross(Z, target), rotation angle = acos(dot)
    v = np.cross(z, ax)
    s = np.linalg.norm(v)
    c = np.dot(z, ax)

    # Skew-symmetric matrix of v
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # Rotation matrix (Rodrigues' formula)
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

    # Extract XYZ body-fixed Euler angles from rotation matrix
    # R = Rx(a) * Ry(b) * Rz(c)
    if abs(R[0, 2]) < 1.0 - 1e-10:
        b = math.asin(R[0, 2])
        a = math.atan2(-R[1, 2], R[2, 2])
        c = math.atan2(-R[0, 1], R[0, 0])
    else:
        # Gimbal lock
        b = math.pi / 2 * np.sign(R[0, 2])
        a = math.atan2(R[1, 0], R[1, 1])
        c = 0.0

    print(f"    Axis {ax} -> Euler XYZ: ({a:.4f}, {b:.4f}, {c:.4f})")
    return osim.Vec3(a, b, c)


def get_mesh_vtp_name(visual_elem):
    """Extract mesh filename from URDF <visual> and convert to .vtp extension."""
    geom = visual_elem.find("geometry")
    if geom is None:
        return None
    mesh = geom.find("mesh")
    if mesh is None:
        return None
    filename = mesh.get("filename", "")
    # Strip package:// prefix
    if "://" in filename:
        filename = filename.split("://", 1)[1]
        # Take just the filename, not the full path
        filename = os.path.basename(filename)
    # Change extension to .vtp
    base = os.path.splitext(filename)[0]
    return base + ".vtp"


def convert_urdf_to_osim(urdf_path, osim_path=None):
    """Main conversion function."""

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.get("name", "robot")

    if osim_path is None:
        osim_path = os.path.splitext(urdf_path)[0] + ".osim"

    print(f"Converting: {urdf_path} → {osim_path}")
    print(f"Robot name: {robot_name}")

    # Create model
    model = osim.Model()
    model.setName(robot_name)
    model.setGravity(osim.Vec3(0, -9.81, 0))

    # --- Parse links (bodies) ---
    links = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        inertial = link_elem.find("inertial")

        if inertial is not None:
            mass_elem = inertial.find("mass")
            mass = float(mass_elem.get("value", 0))

            origin = inertial.find("origin")
            if origin is not None:
                com = parse_vec3(origin.get("xyz", "0 0 0"))
            else:
                com = osim.Vec3(0, 0, 0)

            inertia = parse_inertia(inertial)
        else:
            mass = 0.001  # small default mass
            com = osim.Vec3(0, 0, 0)
            inertia = osim.Inertia(1e-9, 1e-9, 1e-9, 0, 0, 0)

        body = osim.Body(name, mass, com, inertia)

        # Attach mesh geometry if available
        visual = link_elem.find("visual")
        if visual is not None:
            vtp_name = get_mesh_vtp_name(visual)
            if vtp_name:
                mesh = osim.Mesh(vtp_name)
                body.attachGeometry(mesh)
                print(f"  Body '{name}': mass={mass:.6f} kg, mesh={vtp_name}")
            else:
                print(f"  Body '{name}': mass={mass:.6f} kg, no mesh")
        else:
            print(f"  Body '{name}': mass={mass:.6f} kg, no visual")

        links[name] = body
        model.addBody(body)

    # --- Parse joints ---
    joints = root.findall("joint")

    # Find the root link (a link that is only a parent, never a child)
    child_links = set()
    for joint_elem in joints:
        child_name = joint_elem.find("child").get("link")
        child_links.add(child_name)

    parent_links = set()
    for joint_elem in joints:
        parent_name = joint_elem.find("parent").get("link")
        parent_links.add(parent_name)

    root_links = parent_links - child_links
    if not root_links:
        print("ERROR: Could not determine root link.")
        sys.exit(1)

    root_link_name = list(root_links)[0]
    print(f"\n  Root link: '{root_link_name}' (welded to ground at 0, 0.5, 0)")

    # Weld root link to ground (raised 0.5m for visibility)
    weld = osim.WeldJoint(
        "ground_to_" + root_link_name,
        model.getGround(),
        osim.Vec3(0, 0.5, 0), osim.Vec3(-math.pi/2, 0, 0),
        links[root_link_name],
        osim.Vec3(0, 0, 0), osim.Vec3(0, 0, 0),
    )
    model.addJoint(weld)

    # Process each joint
    for joint_elem in joints:
        jname = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent_name = joint_elem.find("parent").get("link")
        child_name = joint_elem.find("child").get("link")

        origin = joint_elem.find("origin")
        if origin is not None:
            location = parse_vec3(origin.get("xyz", "0 0 0"))
            rpy = origin.get("rpy", "0 0 0")
            rpy_vec = parse_vec3(rpy)
        else:
            location = osim.Vec3(0, 0, 0)
            rpy_vec = osim.Vec3(0, 0, 0)

        parent_body = links[parent_name]
        child_body = links[child_name]

        if jtype == "fixed":
            joint = osim.WeldJoint(
                jname,
                parent_body, location, rpy_vec,
                child_body, osim.Vec3(0, 0, 0), osim.Vec3(0, 0, 0),
            )
            model.addJoint(joint)
            print(f"  Joint '{jname}': WeldJoint ({parent_name} → {child_name})")

        elif jtype in ("revolute", "continuous"):
            # Get axis
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz", "0 0 1")
            else:
                axis_xyz = "0 0 1"

            orientation = axis_to_orientation(axis_xyz)

            joint = osim.PinJoint(
                jname,
                parent_body, location, orientation,
                child_body, osim.Vec3(0, 0, 0), orientation,
            )

            # Set coordinate properties
            coord = joint.updCoordinate()
            coord.setName(jname + "_angle")
            coord.setDefaultValue(0)

            if jtype == "revolute":
                limit_elem = joint_elem.find("limit")
                if limit_elem is not None:
                    lower = float(limit_elem.get("lower", "-3.1416"))
                    upper = float(limit_elem.get("upper", "3.1416"))
                    coord.set_range(0, lower)
                    coord.set_range(1, upper)
                    coord.set_clamped(True)

            model.addJoint(joint)
            axis_str = axis_xyz.strip()
            print(f"  Joint '{jname}': PinJoint ({parent_name} → {child_name}), "
                  f"axis=[{axis_str}]")

        elif jtype == "prismatic":
            # Get axis
            axis_elem = joint_elem.find("axis")
            if axis_elem is not None:
                axis_xyz = axis_elem.get("xyz", "1 0 0")
            else:
                axis_xyz = "1 0 0"

            orientation = axis_to_orientation(axis_xyz)

            joint = osim.SliderJoint(
                jname,
                parent_body, location, orientation,
                child_body, osim.Vec3(0, 0, 0), orientation,
            )

            coord = joint.updCoordinate()
            coord.setName(jname + "_displacement")
            coord.setDefaultValue(0)

            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                lower = float(limit_elem.get("lower", "-1.0"))
                upper = float(limit_elem.get("upper", "1.0"))
                coord.set_range(0, lower)
                coord.set_range(1, upper)
                coord.set_clamped(True)

            model.addJoint(joint)
            print(f"  Joint '{jname}': SliderJoint ({parent_name} → {child_name})")

        else:
            print(f"  WARNING: Joint '{jname}' has unsupported type '{jtype}', skipping.")

    # Finalize and save
    model.finalizeConnections()
    model.printToXML(osim_path)

    print(f"\nSaved: {osim_path}")
    print(f"  Bodies: {model.getBodySet().getSize()}")
    print(f"  Joints: {model.getJointSet().getSize()}")
    print(f"  Coordinates: {model.getCoordinateSet().getSize()}")
    print(f"\nRemember to convert STL meshes to ASCII VTP format!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_urdf_to_osim.py input.urdf [output.osim]")
        sys.exit(1)

    urdf_file = sys.argv[1]
    osim_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_urdf_to_osim(urdf_file, osim_file)
