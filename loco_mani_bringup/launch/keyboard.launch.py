#!/usr/bin/env python3
"""Launch the D1+Piper-L keyboard command node."""

from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _start(context, *_args, **_kwargs):
    return [Node(
        package="loco_mani_rl_controller",
        executable="loco_mani_keyboard.py",
        name="loco_mani_keyboard",
        prefix=[LaunchConfiguration("python_executable")],
        arguments=[
             "--command-file", LaunchConfiguration("command_file").perform(context),
             "--publish-rate", LaunchConfiguration("publish_rate").perform(context),
             "--velocity-step", LaunchConfiguration("velocity_step").perform(context),
             "--yaw-step", LaunchConfiguration("yaw_step").perform(context),
             "--ee-step", LaunchConfiguration("ee_step").perform(context),
             "--orientation-step", LaunchConfiguration("orientation_step").perform(context),
             "--orientation-limit", LaunchConfiguration("orientation_limit").perform(context)
             ] + (["--quiet"] if LaunchConfiguration("quiet").perform(context).lower()
                   in ("1", "true", "yes", "on") else []),
        output="screen",
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("command_file", default_value="/tmp/loco_mani_command.json"),
        DeclareLaunchArgument("publish_rate", default_value="20.0"),
        DeclareLaunchArgument("velocity_step", default_value="0.1"),
        DeclareLaunchArgument("yaw_step", default_value="0.15"),
        DeclareLaunchArgument("ee_step", default_value="0.01"),
        DeclareLaunchArgument("orientation_step", default_value="0.05"),
        DeclareLaunchArgument("orientation_limit", default_value="0.6"),
        DeclareLaunchArgument("quiet", default_value="false"),
        DeclareLaunchArgument("python_executable", default_value="/usr/bin/python3"),
        OpaqueFunction(function=_start),
    ])
