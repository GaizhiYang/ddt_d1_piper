#!/usr/bin/env python3
"""Forward existing ROS teleoperation commands to the standalone runtime.

Start this launch together with ``teleop_command`` when using the standalone
``loco_mani_runtime.py`` process. It does not access CAN and is safe to run
in isolation. Do not run it at the same time as ``keyboard.launch.py`` if
both writers use the same command file.
"""

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _start(context, *_args, **_kwargs):
    executable = os.path.join(
        get_package_prefix("loco_mani_rl_controller"),
        "lib",
        "loco_mani_rl_controller",
        "loco_mani_command_bridge.py",
    )
    return [
        ExecuteProcess(
            cmd=[
                LaunchConfiguration("python_executable").perform(context),
                executable,
                "--command-file",
                LaunchConfiguration("command_file").perform(context),
                "--publish-rate",
                LaunchConfiguration("publish_rate").perform(context),
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "command_file", default_value="/tmp/loco_mani_command.json"
            ),
            DeclareLaunchArgument("publish_rate", default_value="20.0"),
            DeclareLaunchArgument("python_executable", default_value="python3"),
            OpaqueFunction(function=_start),
        ]
    )
