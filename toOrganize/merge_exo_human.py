#!/usr/bin/env python3
"""
merge_exo_human.py
Merge WAWA exoskeleton into Rajagopal human musculoskeletal model.
Attaches via BushingForce elements at pelvis, femur_r, femur_l.

Usage (on cortex, conda activate opensim):
    python merge_exo_human.py

Output: combined_model.osim in the same directory as this script.
"""

import opensim as osim
import os
import math

# ============================================================
# CONFIGURATION — adjust these as needed
# ============================================================
WAWA_PATH = "/home/wxp/repos/WAWA/urdf/FES_urdf_0306.osim"
RAJAGOPAL_PATH = "/home/wxp/repos/exo-assist-pipeline/opensim/moco-tutorial/example3DWalking/subject_walk_scaled.osim"
OUTPUT_PATH = "/home/wxp/repos/exo-assist-pipeline/opensim/combined_model.osim"

# Translation offset to align WAWA Origin_base with Rajagopal pelvis
# (applied to the WeldJoint connecting exo base to human pelvis)
EXO_OFFSET_X = -0.07   # posterior (tune: -0.05 to -0.10)
EXO_OFFSET_Y = -0.07   # down
EXO_OFFSET_Z = 0.175  # left (centers exo on human)

# BushingForce stiffness parameters (placeholder, tune later)
TRANS_STIFFNESS = 1000.0   # N/m
ROT_STIFFNESS   = 100.0    # Nm/rad
TRANS_DAMPING   = 50.0     # N·s/m
ROT_DAMPING     = 10.0     # Nm·s/rad

# Bushing attachment pairs: (exo_body, human_body, name)
BUSHING_PAIRS = [
    ("Origin_base", "pelvis",  "bushing_pelvis"),
    ("J2_R",        "femur_r", "bushing_femur_r"),
    ("J2_L",        "femur_l", "bushing_femur_l"),
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prefix_name(name, prefix="exo_"):
    """Add prefix to avoid name collisions."""
    return f"{prefix}{name}"


def add_bushing_force(model, exo_body_name, human_body_name, force_name):
    """Add a BushingForce between an exo body and a human body."""
    bushing = osim.BushingForce()
    bushing.setName(force_name)

    bushing.set_rotational_stiffness(osim.Vec3(ROT_STIFFNESS, ROT_STIFFNESS, ROT_STIFFNESS))
    bushing.set_translational_stiffness(osim.Vec3(TRANS_STIFFNESS, TRANS_STIFFNESS, TRANS_STIFFNESS))
    bushing.set_rotational_damping(osim.Vec3(ROT_DAMPING, ROT_DAMPING, ROT_DAMPING))
    bushing.set_translational_damping(osim.Vec3(TRANS_DAMPING, TRANS_DAMPING, TRANS_DAMPING))

    # Connect sockets by reference to actual body objects
    exo_body = model.getBodySet().get(exo_body_name)
    human_body = model.getBodySet().get(human_body_name)
    bushing.connectSocket_frame1(exo_body)
    bushing.connectSocket_frame2(human_body)

    model.addForce(bushing)
    print(f"  Added BushingForce: {force_name} ({exo_body_name} <-> {human_body_name})")
    
def main():
    # --- Load models ---
    human = osim.Model(RAJAGOPAL_PATH)
    exo = osim.Model(WAWA_PATH)
    human.initSystem()
    exo.initSystem()

    print(f"Human model: {human.getBodySet().getSize()} bodies, "
          f"{human.getJointSet().getSize()} joints")
    print(f"Exo model:   {exo.getBodySet().getSize()} bodies, "
          f"{exo.getJointSet().getSize()} joints")

    # --- Clone exo bodies into human model ---
    # Strategy: 
    #   1. Replace exo ground joint with a WeldJoint to human pelvis
    #   2. Copy all other exo joints/bodies with prefixed names
    #   3. Add BushingForces for coupling

    exo_bodies = exo.getBodySet()
    exo_joints = exo.getJointSet()

    # First pass: create all exo bodies in human model (without joints)
    body_map = {}  # original_name -> new_body
    for i in range(exo_bodies.getSize()):
        eb = exo_bodies.get(i)
        new_name = prefix_name(eb.getName())

        new_body = osim.Body()
        new_body.setName(new_name)
        new_body.setMass(eb.getMass())
        new_body.setInertia(eb.getInertia())
        new_body.setMassCenter(eb.getMassCenter())

        # Copy attached geometry
        for g in range(eb.getPropertyByName("attached_geometry").size()):
            geom = eb.get_attached_geometry(g).clone()
            new_body.append_attached_geometry(geom)

        body_map[eb.getName()] = new_body
        print(f"  Cloned body: {eb.getName()} -> {new_name}")

    # Second pass: create joints connecting exo bodies
    for i in range(exo_joints.getSize()):
        ej = exo_joints.get(i)
        ej_name = ej.getName()

        parent_name = ej.getParentFrame().findBaseFrame().getName()
        child_name = ej.getChildFrame().findBaseFrame().getName()

        # Get the offset transforms
        # parent_offset = ej.getParentFrame().findTransformBetweenFrames(
        #     ej.getParentFrame(), ej.getParentFrame().findBaseFrame()
        # ).invert()
        # child_offset = ej.getChildFrame().findTransformBetweenFrames(
        #     ej.getChildFrame(), ej.getChildFrame().findBaseFrame()
        # ).invert()

        child_body = body_map[child_name]

        # If parent is ground, attach to human pelvis via WeldJoint
        if parent_name == "ground":
            pelvis = human.getBodySet().get("pelvis")
            
            # Create WeldJoint from pelvis to exo base
            offset_in_pelvis = osim.Vec3(EXO_OFFSET_X, EXO_OFFSET_Y, EXO_OFFSET_Z)
            weld = osim.WeldJoint(
                prefix_name(ej_name),
                pelvis,
                offset_in_pelvis,
                osim.Vec3(math.pi / 2, math.pi / 2, math.pi),
                child_body,
                osim.Vec3(0, 0, 0),
                osim.Vec3(0, 0, 0),
            )
            human.addBody(child_body)
            human.addJoint(weld)
            print(f"  WeldJoint: pelvis -> {child_body.getName()} "
                  f"(offset: {EXO_OFFSET_X}, {EXO_OFFSET_Y}, {EXO_OFFSET_Z})")

        else:
            # Recreate the original joint type between prefixed exo bodies
            parent_body = body_map[parent_name]
            new_joint_name = prefix_name(ej_name)

            # Get parent/child offsets from the original joint
            p_loc = ej.get_frames(0).get_translation()
            p_ori = ej.get_frames(0).get_orientation()
            c_loc = ej.get_frames(1).get_translation()
            c_ori = ej.get_frames(1).get_orientation()

            print(f"    Joint: {ej_name}, parent offset: loc={p_loc}, ori={p_ori}")
            print(f"    Joint: {ej_name}, child offset:  loc={c_loc}, ori={c_ori}")
            joint_type = ej.getConcreteClassName()

            if joint_type == "PinJoint":
                new_joint = osim.PinJoint(
                    new_joint_name,
                    parent_body, p_loc, p_ori,
                    child_body, c_loc, c_ori,
                )
                # Copy coordinate properties
                old_coord = ej.get_coordinates(0)
                new_coord = new_joint.get_coordinates(0)
                new_coord.setName(prefix_name(old_coord.getName()))
                new_coord.setRangeMin(old_coord.getRangeMin())
                new_coord.setRangeMax(old_coord.getRangeMax())
                new_coord.setDefaultValue(0.0)

            elif joint_type == "WeldJoint":
                new_joint = osim.WeldJoint(
                    new_joint_name,
                    parent_body, p_loc, p_ori,
                    child_body, c_loc, c_ori,
                )
            else:
                # Fallback: treat as PinJoint
                print(f"  WARNING: Unknown joint type '{joint_type}' "
                      f"for {ej_name}, defaulting to PinJoint")
                new_joint = osim.PinJoint(
                    new_joint_name,
                    parent_body, p_loc, p_ori,
                    child_body, c_loc, c_ori,
                )

            human.addBody(child_body)
            human.addJoint(new_joint)
            print(f"  {joint_type}: {parent_body.getName()} -> "
                  f"{child_body.getName()} (as {new_joint_name})")

    # --- Add BushingForces ---
    print("\nAdding BushingForce elements...")
    for exo_body, human_body, name in BUSHING_PAIRS:
            add_bushing_force(human, prefix_name(exo_body), human_body, name)

    # --- Finalize and save ---
    human.finalizeConnections()
    human.printToXML(OUTPUT_PATH)
    print(f"\nSaved combined model: {OUTPUT_PATH}")
    print("Open in OpenSim GUI to verify alignment.")
    print(f"Tip: re-run visualize_models.py on the combined model to check.")


if __name__ == "__main__":
    main()