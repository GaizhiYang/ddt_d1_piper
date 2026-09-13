#!/usr/bin/env python3
"""Generate the ROS/MuJoCo description used by the D1+Piper-L rollout.

The IsaacLab project contains the authoritative combined URDF.  This helper
keeps the DDT D1 MuJoCo model (including its calibrated inertias and wheel
collisions) and appends the Piper-L chain from that URDF.  It is intentionally
deterministic so it can be rerun after either upstream description changes.

Usage (from the ddt_controller repository)::

    python3 scripts/generate_d1_piper_l_assets.py

The script only copies upstream mesh files and writes generated files below
``loco_mani_description``; it does not alter the existing D1 packages.
"""

from __future__ import annotations

import argparse
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRAINING_ROOT = Path("/home/hh/loco-mani/source/LeggedManip_Lab/LeggedManip_Lab/assets/d1_piper_l")
DESCRIPTION = REPO / "loco_mani_description"

ACTIVE_JOINTS = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]

# Keep the MuJoCo motor ranges in lock-step with the IsaacLab actuator config.
# The legacy D1 XML used +/-80 N*m for the three leg joints, while the policy
# was trained with a 90 N*m effort/saturation limit.  Leaving the old range in
# place silently clips commands inside MuJoCo even when the runtime adapter
# correctly computes a 90 N*m limit.
MOTOR_EFFORT_LIMITS = {
    **{name: 90.0 for name in ACTIVE_JOINTS[:16] if not name.endswith("_foot_joint")},
    **{name: 12.0 for name in ACTIVE_JOINTS[:16] if name.endswith("_foot_joint")},
    "joint1": 20.0,
    "joint2": 20.0,
    "joint3": 15.0,
    "joint4": 7.0,
    "joint5": 5.0,
    "joint6": 5.0,
}

ARM = [
    # name, parent, xyz, rpy, mass, inertial xyz, full inertia, mesh
    ("piper_base_link", None, "0.20 0 0.09", "0 0 0", 1.02,
     "-0.00473641164 0.0000256830 0.0414515180",
     "0.00267433 0.00282612 0.00089624 -0.00000073 -0.00017389 0.00000040", "base_link"),
    ("link1", "piper_base_link", "0 0 0.123", "0 0 0", 0.71,
     "0.00032 -0.00041 -0.00348",
     "0.00050027 0.00040879 0.00045802 0.00000078 0.00000626 0.00000462", "link1"),
    ("link2", "link1", "0 0 0", "1.5707963 -0.1357866 -3.1415926", 1.16,
     "0.25852 -0.00861 0.00126",
     "0.00111915 0.06763251 0.067559 0.00182916 0.00019233 0.00000767", "link2"),
    ("link3", "link2", "0.33823 0 0", "0 0 -1.7938494", 0.5,
     "-0.01999 -0.20808 -0.00053",
     "0.01347608 0.00046377 0.01363321 0.00161034 0.00000066 0.00000162", "link3"),
    ("link4", "link3", "-0.02301 -0.32075 0", "1.5707963 0 0", 0.38,
     "0.00013 -0.00076 -0.00394",
     "0.00017844 0.00018914 0.00014613 0.00000062 0.00000067 0.00000408", "link4"),
    ("link5", "link4", "0 0 0", "-1.5707963 0 0", 0.39,
     "0.00018 -0.06104 -0.00227",
     "0.00172318 0.00017936 0.00171172 0.00000259 0.00000010 0.00000990", "link5"),
    ("link6", "link5", "0 -0.091 0", "1.5707963 0 0", 0.007,
     "-0.00003 0.00006 -0.002",
     "0.00152112 0.00138626 0.00031152 0.00000202 0.00000261 0.00000002", "link6"),
    ("flange_link", "link6", "0 0 0", "0 0 0", 0.04,
     "0.00115169 0.00078243 0.00479805",
     "0.000009935 0.000010508 0.000019653 0 0 0", "flange"),
    # The URDF declares this as an empty, massless frame.  Keep it massless in
    # MuJoCo as well; adding an arbitrary payload here changes the WBC base
    # dynamics and makes the asset no longer numerically comparable.
    ("end_effector", "flange_link", "0 0 0.10", "0 -1.5707963 0", 0.0,
     "0 0 0", "0 0 0 0 0 0", None),
    ("gripper_base", "flange_link", "0 0 0.0045", "0 0 0", 0.45,
     "-0.00018381 0.00008050 0.03214367",
     "0.00092934 0.00071447 0.00039442 0.00000034 -0.00000738 0.00000005", "gripper_base"),
    ("gripper_link1", "gripper_base", "0 0 0.138", "1.5707963 0 0", 0.025,
     "0.00065123 -0.049193 0.00972259",
     "0.00007371 0.00000781 0.00007470 -0.00000113 0.00000021 -0.00001372", "gripper_link1"),
    ("gripper_link2", "gripper_base", "0 0 0.138", "1.5707963 0 -3.1415926", 0.025,
     "0.00065123 -0.049193 0.00972259",
     "0.00007371 0.00000781 0.00007470 -0.00000113 0.00000021 -0.00001372", "gripper_link2"),
]

JOINTS = [
    ("joint1", "link1", "0 0 0", "0 0 1", "-2.6179938 2.6179938", 20),
    ("joint2", "link2", "0 0 0", "0 0 1", "0 3.1415926", 20),
    ("joint3", "link3", "0 0 0", "0 0 1", "-2.9670597 0", 15),
    ("joint4", "link4", "0 0 0", "0 0 1", "-2.2165681 2.2165681", 7),
    ("joint5", "link5", "0 0 0", "0 0 1", "-1.5620696 1.5620696", 5),
    ("joint6", "link6", "0 0 0", "0 0 1", "-2.0943951 2.0943951", 5),
]


def copy_meshes() -> None:
    dst = DESCRIPTION / "mujoco" / "assets"
    dst.mkdir(parents=True, exist_ok=True)
    d1_src = REPO / "urdfs" / "d1_description" / "meshes"
    for path in d1_src.glob("*.STL"):
        shutil.copy2(path, dst / path.name)
    arm_src = TRAINING_ROOT / "piper_l" / "meshes"
    for stem in ("base_link", "link1", "link2", "link3", "link4", "link5", "link6", "flange", "gripper_base", "gripper_link1", "gripper_link2"):
        shutil.copy2(arm_src / f"{stem}.stl", dst / f"piper_{stem}.stl")


def add_inertial(body: ET.Element, row: tuple) -> None:
    _, _, _, _, mass, pos, inertia, _ = row
    if mass <= 0.0:
        return
    ET.SubElement(body, "inertial", pos=pos, mass=str(mass), fullinertia=inertia)


def urdf_rpy_to_mujoco_quat(rpy: str) -> str:
    """Convert URDF fixed-axis XYZ RPY to MuJoCo's wxyz quaternion.

    URDF ``origin rpy`` is a fixed/extrinsic X-Y-Z rotation.  MuJoCo's
    ``body euler`` follows its configured Euler sequence and is not a safe
    textual substitute for URDF RPY (the difference is visible for Piper
    joint2 and all downstream links).  Writing a quaternion removes that
    convention ambiguity.
    """
    roll, pitch, yaw = (float(value) for value in rpy.split())
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    # Quaternion for the URDF fixed-axis XYZ matrix, returned as w x y z.
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return f"{qw:.9g} {qx:.9g} {qy:.9g} {qz:.9g}"


def build_arm() -> ET.Element:
    by_name: dict[str, ET.Element] = {}
    for row in ARM:
        name, parent, xyz, rpy, *_ = row
        body = ET.Element("body", name=name, pos=xyz)
        if rpy != "0 0 0":
            body.set("quat", urdf_rpy_to_mujoco_quat(rpy))
        add_inertial(body, row)
        mesh = row[-1]
        if mesh:
            # Keep the render mesh separate from the collider.  IsaacLab's
            # generated USD uses the URDF collision meshes with
            # ``collider_type=convex_hull``; MuJoCo likewise uses the mesh
            # geom's convex collision approximation.  Collision masks below
            # disable Piper self-collision while retaining floor contact.
            ET.SubElement(body, "geom", type="mesh", mesh=f"piper_{mesh}",
                          contype="0", conaffinity="0", group="1")
            ET.SubElement(body, "geom", name=f"piper_{mesh}_collision",
                          type="mesh", mesh=f"piper_{mesh}",
                          # Collision bit 2 is reserved for the floor.  With
                          # conaffinity=0, Piper does not collide with itself
                          # or the D1 articulation (matching
                          # enabled_self_collisions=False), but the floor's
                          # affinity mask still receives it.
                          contype="2", conaffinity="0", group="3",
                          friction="1.0 0.01 0.001", margin="0.001", condim="3")
        by_name[name] = body
        if parent:
            by_name[parent].append(body)

    for name, parent, pos, axis, limits, effort in JOINTS:
        ET.SubElement(by_name[parent], "joint", name=name, pos=pos, axis=axis, range=limits,
                      limited="true", damping="1.0", armature="0.01", frictionloss="0.01")

    # Reproduce the URDF's passive central gripper DOF and the two mimic finger
    # joints.  None of these three joints is part of the 22-D policy action,
    # but retaining their kinematics avoids silently changing the arm mass and
    # end-effector geometry.  Joint stiffness keeps the reset value at zero;
    # equality constraints implement the URDF multipliers (+/- 0.5).
    gripper_body = ET.Element("body", name="gripper_link", pos="0 0 0")
    by_name["gripper_base"].append(gripper_body)
    # MuJoCo requires a positive inertial on a body carrying a movable joint;
    # the source URDF's gripper_link is massless, so use a negligible epsilon
    # mass/inertia that has no measurable effect on the 54.98 kg robot.
    ET.SubElement(gripper_body, "inertial", pos="0 0 0", mass="1e-6",
                  diaginertia="1e-9 1e-9 1e-9")
    ET.SubElement(gripper_body, "joint", name="gripper", type="slide",
                  axis="0 0 1", range="0 0.1", limited="true",
                  stiffness="20", damping="1", armature="0.001")
    for name, axis, rng, body_name in (("gripper_joint1", "0 0 1", "0 0.05", "gripper_link1"), ("gripper_joint2", "0 0 -1", "-0.05 0", "gripper_link2")):
        body = by_name[body_name]
        ET.SubElement(body, "joint", name=name, type="slide", axis=axis, range=rng,
                      limited="true", stiffness="20", damping="1", armature="0.001")
    return by_name["piper_base_link"]


def write_mjcf() -> None:
    src = REPO / "urdfs" / "d1_description" / "mujoco" / "robot.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "assets")
        compiler.set("autolimits", "true")
    base = root.find("./worldbody/body[@name='base_link']")
    if base is None:
        raise RuntimeError("D1 MuJoCo base_link not found")
    # IsaacLab training reset uses root position z=0.45 m.  The upstream DDT
    # demo XML uses 0.8 m (a suspended visualization pose), which causes a
    # different impact transient and is not suitable for policy validation.
    base.set("pos", "0 0 0.45")
    base.append(build_arm())
    # The legacy DDT visualizer used condim=1 for every collision geom.  That
    # removes tangential contact forces and is unsuitable for the locomotion
    # policy (the IsaacLab training floor uses friction 0.5..1.2).  Keep
    # decorative/static collisions conservative, but explicitly give the four
    # wheel rollers 3D frictional contacts.
    for geom in root.iter("geom"):
        if geom.get("name", "").endswith("_foot_collision"):
            geom.set("condim", "3")
            geom.set("friction", "1.0 0.01 0.001")
    # The explicit IsaacLab actuator writes zero damping to PhysX and sets
    # the configured joint friction coefficient in the simulator.  Preserve
    # that split in MuJoCo: damping is disabled at runtime by the rollout,
    # while frictionloss carries the actuator friction (0.589 for D1 leg
    # motors, 0 for wheels, and 0.01 for Piper).
    leg_names = {n for n in ACTIVE_JOINTS[:16] if not n.endswith("_foot_joint")}
    wheel_names = {n for n in ACTIVE_JOINTS[:16] if n.endswith("_foot_joint")}
    arm_names = set(ACTIVE_JOINTS[16:])
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name in leg_names:
            joint.set("frictionloss", "0.589")
        elif name in wheel_names:
            joint.set("frictionloss", "0.0")
            # The IsaacLab wheel DCMotorCfg leaves armature unset.  The DDT
            # MJCF class default is 0.0535 (the leg-motor value), so make the
            # wheel value explicit instead of accidentally inheriting it.
            joint.set("armature", "0.0")
        elif name in arm_names:
            joint.set("frictionloss", "0.01")
    # Match the training articulation's disabled self-collisions.  Piper
    # colliders use bit 2 and no affinity, while the floor advertises bits 1
    # and 2 in scene.xml; this permits floor contact but no Piper-Piper or
    # Piper-D1 contact.
    for geom in root.iter("geom"):
        if geom.get("group") == "3" and geom.get("mesh", "").startswith("piper_"):
            geom.set("contype", "2")
            geom.set("conaffinity", "0")
    # ``actuatorfrcrange`` is not a MuJoCo joint attribute (the range belongs
    # on the motor actuator).  Older DDT XMLs contained it as an inert
    # annotation; remove it from the generated model for current MuJoCo.
    for joint in root.iter("joint"):
        joint.attrib.pop("actuatorfrcrange", None)
        joint.attrib.pop("actuatorfrclimited", None)
    sensor = root.find("sensor")
    if sensor is not None:
        # These diagnostics are useful in the original DDT bridge but are not
        # needed by this standalone rollout and some MuJoCo versions reject
        # jointactuatorfrc sensors for free-base models.
        for item in list(sensor):
            if item.tag == "jointactuatorfrc":
                sensor.remove(item)
    asset = root.find("asset")
    assert asset is not None
    # New meshes are referenced by name; the existing D1 mesh declarations are
    # retained and all Piper meshes are added here.
    for row in ARM:
        mesh = row[-1]
        if mesh:
            ET.SubElement(asset, "mesh", name=f"piper_{mesh}", file=f"piper_{mesh}.stl")
    actuator = root.find("actuator")
    assert actuator is not None
    # Update pre-existing D1 motors as well as the Piper motors appended
    # below.  MuJoCo enforces ``ctrlrange`` before integrating, so this is a
    # functional limit rather than merely descriptive metadata.
    for motor in actuator.findall("motor"):
        joint_name = motor.get("joint")
        if joint_name in MOTOR_EFFORT_LIMITS:
            effort = MOTOR_EFFORT_LIMITS[joint_name]
            motor.set("ctrlrange", f"{-effort:g} {effort:g}")
            motor.set("ctrllimited", "true")
    arm_limits = {row[0]: row[5] for row in JOINTS}
    for name, effort in arm_limits.items():
        ET.SubElement(actuator, "motor", name=f"piper_{name}", joint=name,
                      ctrlrange=f"{-effort} {effort}", ctrllimited="true")
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "joint", name="gripper_mimic_left",
                  joint1="gripper_joint1", joint2="gripper",
                  polycoef="0 0.5 0 0 0")
    ET.SubElement(equality, "joint", name="gripper_mimic_right",
                  joint1="gripper_joint2", joint2="gripper",
                  polycoef="0 -0.5 0 0 0")
    out = DESCRIPTION / "mujoco" / "robot.xml"
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=False)
    (DESCRIPTION / "mujoco" / "scene.xml").write_text(
        '<mujoco model="d1_piper_l scene">\n'
        '  <include file="robot.xml"/>\n'
        '  <statistic center="0 0 0.1" extent="0.8"/>\n'
        '  <visual>\n'
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>\n'
        '    <rgba haze="0.15 0.25 0.35 1"/>\n'
        '    <global azimuth="-130" elevation="-20"/>\n'
        '  </visual>\n'
        '  <asset>\n'
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>\n'
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>\n'
        '    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>\n'
        '  </asset>\n'
        '  <worldbody>\n'
        '    <geom name="floor" type="plane" size="0 0 0.05" material="groundplane" friction="1 0.8 0.5" contype="1" conaffinity="3"/>\n'
        '  </worldbody>\n'
        '</mujoco>\n', encoding="utf-8")


def ros2_control_xacro() -> str:
    """Return the standalone ros2_control xacro file.

    Keep the control description out of the generated plain URDF.  A xacro
    macro makes the hardware plugin/name selectable by launch files while
    retaining one authoritative list of the 22 policy joints and IMU state
    interfaces.  The defaults are the MuJoCo bridge used by the sim2sim
    package; real hardware launch files should pass their own plugin.
    """
    lines = [
        '<?xml version="1.0"?>',
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="loco_mani_ros2control">',
        '  <xacro:macro name="loco_mani_ros2_control"',
        '                 params="hardware_plugin:=\'mujoco_ros2_control/MujocoSystem\' name:=\'mujoco\'">',
        '    <ros2_control name="${name}" type="system">',
        '      <hardware><plugin>${hardware_plugin}</plugin></hardware>',
    ]
    for name in ACTIVE_JOINTS:
        lines += [
            f'      <joint name="{name}">',
            '        <command_interface name="position"/>',
            '        <command_interface name="velocity"/>',
            '        <command_interface name="effort"><param name="min">-100</param><param name="max">100</param></command_interface>',
            '        <command_interface name="kp"/><command_interface name="kd"/>',
            '        <state_interface name="position"/><state_interface name="velocity"/><state_interface name="effort"/>',
            '      </joint>',
        ]
    lines += [
        '      <sensor name="trunk_imu">',
        '        <state_interface name="orientation.x"/><state_interface name="orientation.y"/>',
        '        <state_interface name="orientation.z"/><state_interface name="orientation.w"/>',
        '        <state_interface name="angular_velocity.x"/><state_interface name="angular_velocity.y"/><state_interface name="angular_velocity.z"/>',
        '        <state_interface name="linear_acceleration.x"/><state_interface name="linear_acceleration.y"/><state_interface name="linear_acceleration.z"/>',
        '      </sensor>',
        '    </ros2_control>',
        '  </xacro:macro>',
        '  <!-- Compatibility alias matching the original D1 description. -->',
        '  <xacro:macro name="ros2control"',
        '                 params="hw_env:=\'none\' hardware_plugin:=\'\' name:=\'mujoco\'">',
        '    <xacro:property name="selected_plugin" value="${hardware_plugin}"/>',
        '    <xacro:if value="${hw_env == \'webots\'}">',
        '      <xacro:property name="selected_plugin" value="tita_webots_ros2_control::WebotsBridge"/>',
        '    </xacro:if>',
        '    <xacro:if value="${hw_env == \'gazebo\'}">',
        '      <xacro:property name="selected_plugin" value="usr_gazebo_ros2_control/GazeboBridge"/>',
        '    </xacro:if>',
        '    <xacro:if value="${hw_env == \'hw\'}">',
        '      <xacro:property name="selected_plugin" value="tita_locomotion::HardwareBridge"/>',
        '    </xacro:if>',
        '    <xacro:if value="${selected_plugin == \'\'}">',
        '      <xacro:property name="selected_plugin" value="mujoco_ros2_control/MujocoSystem"/>',
        '    </xacro:if>',
        '    <xacro:loco_mani_ros2_control hardware_plugin="${selected_plugin}" name="${name}"/>',
        '  </xacro:macro>',
        '</robot>',
    ]
    return "\n".join(lines)


def write_urdf() -> None:
    source = TRAINING_ROOT / "d1_piper_l.urdf"
    text = source.read_text(encoding="utf-8")
    # The upstream combined URDF leaves most visual material tags unnamed
    # (and uses ``name=\"\"`` for two links).  ROS ``check_urdf`` rejects
    # both forms, while MuJoCo simply ignores the visual material name.  Give
    # every generated material a stable name so the same description can be
    # consumed by ROS, xacro and MoveIt.
    text = text.replace('<material name="">', '<material name="d1_gray">')
    text = re.sub(r'<material>', '<material name="d1_gray">', text)
    text = text.replace("d1_description/meshes/", "package://loco_mani_description/meshes/d1/")
    text = text.replace("piper_l/meshes/", "package://loco_mani_description/meshes/piper/")
    # Keep the plain URDF hardware-independent.  ros2_control is injected
    # only by the xacro wrapper below, so tools such as check_urdf/RViz can
    # consume this file without requiring a hardware plugin to be installed.
    # Strip a block if a future upstream training URDF starts embedding one;
    # this prevents regeneration from silently reintroducing a stale plugin.
    text = re.sub(r"\s*<ros2_control\b[^>]*>.*?</ros2_control>\s*", "\n", text,
                  flags=re.DOTALL)
    robot_tag = '<robot name="d1_piper_l">'
    if robot_tag not in text:
        raise RuntimeError("unexpected root tag in training URDF")
    out = DESCRIPTION / "urdf" / "robot.urdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    xacro_text = text.replace(
        robot_tag,
        '<robot name="d1_piper_l" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
        '  <xacro:arg name="hardware_plugin" default="mujoco_ros2_control/MujocoSystem"/>\n'
        '  <xacro:arg name="ros2_control_name" default="mujoco"/>\n'
        '  <xacro:include filename="$(find loco_mani_description)/xacro/ros2control.xacro"/>\n'
        '  <xacro:loco_mani_ros2_control\n'
        '      hardware_plugin="$(arg hardware_plugin)"\n'
        '      name="$(arg ros2_control_name)"/>',
    )
    (DESCRIPTION / "xacro" / "robot.xacro").write_text(xacro_text, encoding="utf-8")
    (DESCRIPTION / "xacro" / "ros2control.xacro").write_text(
        ros2_control_xacro(), encoding="utf-8"
    )

    # Keep package:// paths in the ROS description while making a convenient
    # flat mesh directory for tools that inspect this package without ament.
    mesh_dst = DESCRIPTION / "meshes"
    (mesh_dst / "d1").mkdir(parents=True, exist_ok=True)
    (mesh_dst / "piper").mkdir(parents=True, exist_ok=True)
    for path in (REPO / "urdfs" / "d1_description" / "meshes").glob("*.STL"):
        shutil.copy2(path, mesh_dst / "d1" / path.name)
    for path in (TRAINING_ROOT / "piper_l" / "meshes").glob("*.stl"):
        shutil.copy2(path, mesh_dst / "piper" / path.name)
    # Keep the directory layout used by the authoritative combined URDF:
    # visual meshes are referenced as ``meshes/piper/dae/*.dae``.  ``check_urdf``
    # validates XML but does not verify that these mesh files exist, so copying
    # them to the parent directory can otherwise go unnoticed until RViz or a
    # robot-state-publisher is started.
    dae_dst = mesh_dst / "piper" / "dae"
    dae_dst.mkdir(parents=True, exist_ok=True)
    for path in (TRAINING_ROOT / "piper_l" / "meshes" / "dae").glob("*.dae"):
        shutil.copy2(path, dae_dst / path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if not TRAINING_ROOT.exists():
        raise SystemExit(f"Training asset directory not found: {TRAINING_ROOT}")
    for path in (DESCRIPTION / "mujoco", DESCRIPTION / "xacro", DESCRIPTION / "urdf"):
        path.mkdir(parents=True, exist_ok=True)
    copy_meshes()
    write_mjcf()
    write_urdf()
    print(f"Generated D1+Piper-L assets under {DESCRIPTION}")


if __name__ == "__main__":
    main()
