#!/usr/bin/env python3
"""Launch the D1 + Piper-L runtime in dry-run/shadow mode by default.

The launch file deliberately requires two independent launch arguments before
any real command can be sent: ``send:=true`` and ``hardware_gate:=true``.
Even then, the runtime refuses incomplete joint calibration and missing D1
vendor factories.
"""

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _as_bool(context, name: str) -> bool:
    value = LaunchConfiguration(name).perform(context).strip().lower()
    if value not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise RuntimeError(f"{name} must be boolean, got {value!r}")
    return value in {"1", "true", "yes", "on"}


def _start(context, *_args, **_kwargs):
    executable = os.path.join(
        get_package_prefix("loco_mani_rl_controller"), "lib",
        "loco_mani_rl_controller", "loco_mani_runtime.py",
    )
    command = [
        executable,
        "--config", LaunchConfiguration("config").perform(context),
        "--duration", LaunchConfiguration("duration").perform(context),
    ]
    policy_path = LaunchConfiguration("policy_path").perform(context).strip()
    if policy_path:
        command.extend(["--policy-path", policy_path])
    command_velocity = LaunchConfiguration("command_velocity").perform(context).strip()
    if command_velocity:
        command.extend(["--command-velocity", command_velocity])
    ee_pose = LaunchConfiguration("ee_pose").perform(context).strip()
    if ee_pose:
        command.extend(["--ee-pose", ee_pose])
    command_file = LaunchConfiguration("command_file").perform(context).strip()
    if command_file:
        command.extend([
            "--command-file", command_file,
            "--command-file-timeout", LaunchConfiguration("command_file_timeout").perform(context),
        ])
    if _as_bool(context, "send"):
        command.append("--send")
    if _as_bool(context, "shadow"):
        command.append("--shadow")
    if _as_bool(context, "hardware_gate"):
        command.append("--hardware-gate")
    if _as_bool(context, "enable_piper"):
        command.append("--enable-piper")
    if _as_bool(context, "d1_force_direct"):
        command.append("--d1-force-direct")
    if LaunchConfiguration("d1_api_factory").perform(context):
        command.extend(["--d1-api-factory", LaunchConfiguration("d1_api_factory").perform(context)])
    if LaunchConfiguration("d1_motor_out_factory").perform(context):
        command.extend(["--d1-motor-out-factory", LaunchConfiguration("d1_motor_out_factory").perform(context)])
    return [ExecuteProcess(
        cmd=[LaunchConfiguration("python_executable").perform(context), *command],
        output="screen",
    )]


def generate_launch_description():
    # Resolve the installed share directory so this launch file is portable to
    # the Jetson image.  The source-tree path remains available through the
    # package's symlink install during workstation development.
    config = os.path.join(
        get_package_share_directory("loco_mani_rl_controller"),
        "config", "d1_piper_l.yaml",
    )
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=config),
        DeclareLaunchArgument("policy_path", default_value="",
                             description="override policy path; empty uses policy.path from config"),
        DeclareLaunchArgument("duration", default_value="5.0"),
        # Empty means that the shared policy YAML supplies the initial
        # command.  A CSV value is an explicit launch-time override.
        DeclareLaunchArgument("command_velocity", default_value=""),
        DeclareLaunchArgument("ee_pose", default_value=""),
        DeclareLaunchArgument("command_file", default_value=""),
        DeclareLaunchArgument("command_file_timeout", default_value="0.5"),
        DeclareLaunchArgument("send", default_value="false",
                             description="enable vendor command sends (unsafe; default false)"),
        DeclareLaunchArgument("shadow", default_value="false",
                             description="read both real buses and run policy without enabling/sending motors"),
        DeclareLaunchArgument("hardware_gate", default_value="false",
                             description="explicit second gate for real CAN access"),
        DeclareLaunchArgument("enable_piper", default_value="false",
                             description="explicit third gate for Piper motor enable"),
        DeclareLaunchArgument(
            "d1_force_direct", default_value="false",
            description="explicitly request D1 FORCE_DIRECT RPC before sending",
        ),
        DeclareLaunchArgument(
            "d1_api_factory", default_value="d1_tita_adapter:create_d1_api",
            description="validated flat-C D1 adapter factory; only called in shadow/send modes"),
        DeclareLaunchArgument(
            "d1_motor_out_factory", default_value="d1_tita_adapter:make_motor_out",
            description="D1 command-record factory for the flat-C adapter"),
        DeclareLaunchArgument("python_executable", default_value="python3"),
        OpaqueFunction(function=_start),
    ])
