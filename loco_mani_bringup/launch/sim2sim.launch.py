#!/usr/bin/env python3
"""ROS 2 managed D1 + Piper-L loco-manipulation sim2sim launch."""

from __future__ import annotations

import os
import subprocess
import sys

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            OpaqueFunction, RegisterEventHandler, TimerAction)
from launch.events import Shutdown
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def _start(context, *_args, **_kwargs):
    def as_bool(name: str) -> bool:
        value = LaunchConfiguration(name).perform(context).strip().lower()
        if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
            raise RuntimeError(f"{name} must be boolean, got {value!r}")
        return value in {"1", "true", "yes", "on"}

    robot_share = get_package_share_directory("loco_mani_description")
    controller_share = get_package_share_directory("loco_mani_rl_controller")
    robot_description = xacro.process_file(
        os.path.join(robot_share, "xacro", "robot.xacro"),
        mappings={"hardware_plugin": "mujoco_ros2_control/MujocoSystem", "ros2_control_name": "mujoco"},
    ).toxml()
    robot_controllers = os.path.join(controller_share, "config", "ros2_controllers.yaml")
    if not os.path.isfile(robot_controllers):
        raise RuntimeError(f"controller configuration does not exist: {robot_controllers}")
    policy_config = LaunchConfiguration("config").perform(context).strip()
    if not os.path.isfile(policy_config):
        raise RuntimeError(f"policy configuration does not exist: {policy_config}")

    # ``rclpy`` is a CPython extension and must be loaded by the same Python
    # minor version that ROS 2 Humble was built for.  In particular, a common
    # MuJoCo/ONNX virtual environment on this machine is Python 3.9 while
    # Humble uses Python 3.10; starting a ROS node with that interpreter gives
    # the misleading ``rclpy._rclpy_pybind11`` import error.  Keep the launch
    # interface configurable, but fail over to the interpreter running this
    # launch file when the requested one is ABI-incompatible.
    requested_python = LaunchConfiguration("python_executable").perform(context).strip()
    policy_python = requested_python or sys.executable
    try:
        requested_version = subprocess.check_output(
            [policy_python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            text=True, stderr=subprocess.STDOUT, timeout=3.0,
        ).strip()
        launch_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if requested_version != launch_version:
            print(
                f"[loco_mani] python_executable={policy_python!r} is Python "
                f"{requested_version}, but ROS 2 launch uses {launch_version}; "
                f"using {sys.executable!r} for the ROS policy node.",
                flush=True,
            )
            policy_python = sys.executable
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"cannot execute python_executable={policy_python!r} to check its "
            "compatibility with ROS 2 rclpy"
        ) from exc
    mujoco_sim = Node(
        # Do not remap the executable's internal node name.  The MuJoCo
        # ros2_control plugin creates a second node named ``controller_manager``
        # in the same process and forwards the parameter-file arguments to it.
        # An explicit ``name=`` here would add ``-r __node:=...`` to those
        # arguments as well, renaming the controller manager and preventing it
        # from receiving its ``/**/controller_manager`` parameters.
        package="mujoco_sim_ros2", executable="mujoco_sim",
        output="screen", parameters=[
            {"model_package": "loco_mani_description"},
            {"model_file": "mujoco/scene.xml"},
            {"physics_plugins": ["mujoco_ros2_control::MujocoRos2ControlPlugin"]},
            {"headless": as_bool("headless")},
            {"real_time": as_bool("real_time")},
            # Count duration in MuJoCo simulation seconds.  The model is held
            # at its initial state until all ROS 2 controllers and the policy
            # process have started.
            {"duration": float(LaunchConfiguration("duration").perform(context))},
            {"start_paused": as_bool("start_paused")},
            robot_controllers,
        ])
    robot_state_publisher = Node(
        package="robot_state_publisher", executable="robot_state_publisher",
        name="robot_state_publisher", output="screen",
        parameters=[{"robot_description": robot_description}])
    manager = LaunchConfiguration("controller_manager").perform(context)

    def spawner(controller: str):
        return Node(package="controller_manager", executable="spawner", name=f"spawner_{controller}",
                    arguments=[controller, "--controller-manager", manager], output="screen")

    joint_states = spawner("joint_state_broadcaster")
    imu = spawner("imu_sensor_broadcaster")
    position = spawner("loco_mani_position_controller")
    velocity = spawner("loco_mani_velocity_controller")
    effort = spawner("loco_mani_effort_controller")
    kp = spawner("loco_mani_kp_controller")
    kd = spawner("loco_mani_kd_controller")

    policy_args = ["--config", policy_config]
    # Do not pass empty optional values as an argparse option followed by the
    # next option (``--command-file --command-file-timeout ...``); argparse
    # quite reasonably interprets the next option as a missing argument.  An
    # empty command-file means that the policy uses ROS command topics only.
    policy_path = LaunchConfiguration("policy_path").perform(context).strip()
    if policy_path and not os.path.isfile(policy_path):
        raise RuntimeError(f"policy model does not exist: {policy_path}")
    command_file = LaunchConfiguration("command_file").perform(context).strip()
    if policy_path:
        policy_args += ["--policy-path", policy_path]
    if command_file:
        policy_args += ["--command-file", command_file]
        command_file_timeout = LaunchConfiguration("command_file_timeout").perform(context).strip()
        if command_file_timeout:
            policy_args += ["--command-file-timeout", command_file_timeout]
    command_velocity = LaunchConfiguration("command_velocity").perform(context).strip()
    ee_pose = LaunchConfiguration("ee_pose").perform(context).strip()
    if command_velocity:
        policy_args += ["--command-velocity", command_velocity]
    if ee_pose:
        policy_args += ["--ee-pose", ee_pose]
    policy_args += [
                  "--joint-state-topic", LaunchConfiguration("joint_state_topic").perform(context),
                  "--imu-topic", LaunchConfiguration("imu_topic").perform(context),
                  "--twist-topic", LaunchConfiguration("twist_topic").perform(context),
                  "--pose-topic", LaunchConfiguration("pose_topic").perform(context),
                  "--position-controller", LaunchConfiguration("position_controller").perform(context),
                  "--velocity-controller", LaunchConfiguration("velocity_controller").perform(context),
                  "--effort-controller", LaunchConfiguration("effort_controller").perform(context),
                  "--kp-controller", LaunchConfiguration("kp_controller").perform(context),
                  "--kd-controller", LaunchConfiguration("kd_controller").perform(context)]
    policy_command_timeout = LaunchConfiguration("policy_command_timeout").perform(context).strip()
    if policy_command_timeout:
        policy_args += ["--command-timeout", policy_command_timeout]
    # Use a launch_ros Node action so the policy process participates in ROS 2
    # name/remapping/parameter management just like the original C++
    # locomotion controller.  ``prefix`` keeps the Python interpreter
    # selectable for machines that provide a compatible ONNX Runtime build.
    policy = Node(
        package="loco_mani_rl_controller",
        executable="d1_piper_l_ros2_node.py",
        name="d1_piper_loco_mani_policy",
        # The installed script has a Python shebang.  ``python_executable``
        # remains an explicit launch argument for environments whose ROS 2
        # interpreter is not /usr/bin/python3; the child still receives normal
        # ROS arguments and participates in launch lifecycle management.
        prefix=[policy_python],
        arguments=policy_args,
        output="screen",
    )
    release_sim = ExecuteProcess(
        cmd=["ros2", "param", "set", "/mujoco_sim_ros2_node", "start_paused", "false"],
        output="screen",
    )
    def _continue_or_shutdown(next_action, label):
        def handler(event, _context):
            if event.returncode == 0:
                return [next_action]
            return [EmitEvent(event=Shutdown(
                reason=f"{label} exited with return code {event.returncode}"))]
        return handler

    def _shutdown_on_failure(label):
        def handler(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(
                    reason=f"{label} exited with return code {event.returncode}"))]
            return []
        return handler

    def _shutdown_always(label):
        def handler(event, _context):
            return [EmitEvent(event=Shutdown(
                reason=f"{label} exited with return code {event.returncode}"))]
        return handler

    actions = [
        mujoco_sim, robot_state_publisher,
        RegisterEventHandler(OnProcessStart(target_action=mujoco_sim, on_start=[joint_states])),
        RegisterEventHandler(OnProcessExit(
            target_action=mujoco_sim,
            on_exit=lambda _event, _context: [EmitEvent(event=Shutdown(
                reason="MuJoCo simulation exited"))])),
        RegisterEventHandler(OnProcessExit(target_action=robot_state_publisher,
                                           on_exit=_shutdown_always("robot_state_publisher"))),
        RegisterEventHandler(OnProcessExit(
            target_action=joint_states,
            on_exit=_continue_or_shutdown(imu, "joint_state_broadcaster spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=imu,
            on_exit=_continue_or_shutdown(position, "imu_sensor_broadcaster spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=position,
            on_exit=_continue_or_shutdown(velocity, "position controller spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=velocity,
            on_exit=_continue_or_shutdown(effort, "velocity controller spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=effort,
            on_exit=_continue_or_shutdown(kp, "effort controller spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=kp,
            on_exit=_continue_or_shutdown(kd, "kp controller spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=kd,
            on_exit=_continue_or_shutdown(policy, "kd controller spawner"))),
        RegisterEventHandler(OnProcessExit(
            target_action=release_sim,
            on_exit=_shutdown_on_failure("simulation release"))),
        RegisterEventHandler(OnProcessExit(
            target_action=policy,
            on_exit=_shutdown_always("policy node"))),
    ]
    # ``OnActionEventBase`` in ROS 2 Humble has no handler callback when an
    # empty action list is supplied.  Register this event handler only for the
    # paused-start mode; otherwise a normal ``start_paused:=false`` launch
    # would crash immediately when the policy process emits ProcessStarted.
    if as_bool("start_paused"):
        actions.append(RegisterEventHandler(OnProcessStart(
            target_action=policy, on_start=[release_sim])))
    try:
        duration = float(LaunchConfiguration("duration").perform(context))
    except ValueError as exc:
        raise RuntimeError("duration must be numeric") from exc
    if not (duration >= 0.0):
        raise RuntimeError("duration must be non-negative (0 means unlimited)")
    return actions


def generate_launch_description():
    default_config = os.path.join(get_package_share_directory("loco_mani_rl_controller"),
                                   "config", "d1_piper_l.yaml")
    args = [
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument("policy_path", default_value=""),
        DeclareLaunchArgument("controller_manager", default_value="/controller_manager"),
        DeclareLaunchArgument("duration", default_value="0"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("real_time", default_value="true"),
        DeclareLaunchArgument(
            "start_paused", default_value="true",
            description="hold MuJoCo while ROS 2 controllers are loaded"),
        # ROS 2 Humble on Ubuntu 22.04 ships rclpy for Python 3.10.  The
        # policy process must therefore use the system interpreter (it can
        # still import the user's ONNX Runtime installation); a Python 3.9
        # environment cannot load Humble's rclpy C extension.
        DeclareLaunchArgument("python_executable", default_value=sys.executable),
        DeclareLaunchArgument("command_file", default_value=""),
        DeclareLaunchArgument("command_file_timeout", default_value="",
                              description="empty uses runtime.command_file_timeout_s from YAML"),
        DeclareLaunchArgument("policy_command_timeout", default_value="",
                              description="empty uses runtime.policy_watchdog_timeout_ms from YAML"),
        # Empty means "use commands.*.default from the policy YAML".  A
        # concrete CSV value remains an explicit launch-time override.
        DeclareLaunchArgument("command_velocity", default_value=""),
        DeclareLaunchArgument("ee_pose", default_value=""),
        DeclareLaunchArgument("joint_state_topic", default_value="joint_states"),
        DeclareLaunchArgument("imu_topic", default_value="imu_sensor_broadcaster/imu"),
        DeclareLaunchArgument("twist_topic", default_value="command/cmd_twist"),
        DeclareLaunchArgument("pose_topic", default_value="command/cmd_pose"),
        DeclareLaunchArgument("position_controller", default_value="loco_mani_position_controller"),
        DeclareLaunchArgument("velocity_controller", default_value="loco_mani_velocity_controller"),
        DeclareLaunchArgument("effort_controller", default_value="loco_mani_effort_controller"),
        DeclareLaunchArgument("kp_controller", default_value="loco_mani_kp_controller"),
        DeclareLaunchArgument("kd_controller", default_value="loco_mani_kd_controller"),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_start)])
