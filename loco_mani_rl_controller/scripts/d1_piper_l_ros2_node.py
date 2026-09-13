#!/usr/bin/env python3
"""ROS 2 policy node for the D1 + Piper-L MuJoCo sim2sim stack.

The MuJoCo process owns the physics and ros2_control hardware component.  This
node owns only policy execution: it consumes the state/IMU broadcasters,
evaluates the existing 22-D ONNX policy at the configured policy rate, and
publishes command arrays to five forward_command_controller instances.  The
same observation and action adapters are shared with the CAN runtime.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

try:
    from d1_piper_hardware import (
        ActionAdapter,
        HardwarePolicyRuntime,
        JointCommand,
        OnnxPolicy,
        StateSnapshot,
    )
    from loco_mani_config import load_resolved_config
    from command_file import CommandFileReader
except ImportError:  # pragma: no cover - installed/source-tree fallback
    from loco_mani_rl_controller.scripts.d1_piper_hardware import (
        ActionAdapter,
        HardwarePolicyRuntime,
        JointCommand,
        OnnxPolicy,
        StateSnapshot,
    )
    from loco_mani_rl_controller.scripts.loco_mani_config import load_resolved_config
    from loco_mani_rl_controller.scripts.command_file import CommandFileReader


JOINT_COUNT = 22
D1_COUNT = 16
DEFAULT_VELOCITY = np.zeros(3, dtype=np.float32)
DEFAULT_EE_POSE = np.asarray([0.425, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _csv(text: str, size: int, name: str) -> np.ndarray:
    values = np.asarray([float(item.strip()) for item in text.split(",")], dtype=np.float32)
    if values.size != size or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain {size} finite comma-separated values")
    return values


class D1PiperRos2Policy(Node):
    """State subscriber, policy loop and ros2_control command publisher."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("d1_piper_loco_mani_policy")
        self.config = load_resolved_config(args.config) if args.config else None
        config = self.config
        policy_path = args.policy_path or (config.path if config is not None else "")
        expected_sha256 = config.sha256 if config is not None else None
        self.policy = OnnxPolicy(
            policy_path,
            expected_sha256=expected_sha256,
            config=config,
        ) if policy_path else None
        self.runtime = HardwarePolicyRuntime(
            self.policy if self.policy is not None else (lambda _frame, _history: np.zeros(JOINT_COUNT, np.float32)),
            max_action_abs=float(config.action_clip if config is not None and args.action_clip is None
                                 else (args.action_clip if args.action_clip is not None else 100.0)),
            config=config,
        )

        self.policy_hz = float(
            args.policy_hz if args.policy_hz is not None
            else (config.frequency_hz if config is not None else 50.0))
        self.command_hz = float(
            args.command_hz if args.command_hz is not None
            else (config.command_hz if config is not None else 500.0))
        if not np.isfinite(self.policy_hz) or self.policy_hz <= 0:
            raise ValueError("policy_hz must be finite and positive")
        if not np.isfinite(self.command_hz) or self.command_hz <= 0:
            raise ValueError("command_hz must be finite and positive")
        self.command_period = 1.0 / self.command_hz
        self.policy_period = 1.0 / self.policy_hz
        # ``None`` means that the launch/CLI did not override the command.
        # Resolve that case from the shared YAML so sim2sim and hardware use
        # the same trained default instead of silently falling back to a
        # second hard-coded value in this ROS node.
        velocity_value = (
            args.command_velocity
            if args.command_velocity is not None
            else (config.command_velocity if config is not None else DEFAULT_VELOCITY)
        )
        pose_value = (
            args.ee_pose
            if args.ee_pose is not None
            else (config.ee_pose if config is not None else DEFAULT_EE_POSE)
        )
        self.command_velocity = (
            _csv(velocity_value, 3, "command_velocity")
            if isinstance(velocity_value, str)
            else np.asarray(velocity_value, dtype=np.float32).reshape(-1)
        )
        self.ee_pose = (
            _csv(pose_value, 7, "ee_pose")
            if isinstance(pose_value, str)
            else np.asarray(pose_value, dtype=np.float32).reshape(-1)
        )
        if self.command_velocity.shape != (3,) or not np.isfinite(self.command_velocity).all():
            raise ValueError("command_velocity must contain 3 finite values")
        if self.ee_pose.shape != (7,) or not np.isfinite(self.ee_pose).all():
            raise ValueError("ee_pose must contain 7 finite values")
        quat_norm = float(np.linalg.norm(self.ee_pose[3:]))
        if quat_norm < 1.0e-8:
            raise ValueError("ee_pose quaternion must be non-zero")
        self.ee_pose[3:] /= quat_norm
        self.command_timeout = float(
            args.command_timeout if args.command_timeout is not None
            else (config.command_timeout_s if config is not None else 0.2))
        if not np.isfinite(self.command_timeout) or self.command_timeout <= 0:
            raise ValueError("command_timeout must be finite and positive")

        self._state_lock = threading.Lock()
        self._snapshot = StateSnapshot()
        self._joint_seen = np.zeros(JOINT_COUNT, dtype=bool)
        self._imu_seen = False
        self._last_ros_command = 0.0
        self._ros_command_seen = False
        self._last_state = 0.0
        self._last_policy = 0.0
        self._last_report = 0.0
        self._last_command = JointCommand()
        self._running = True

        self._joint_names = tuple(self.runtime.config.joint_names)
        self._joint_indices = {name: index for index, name in enumerate(self._joint_names)}
        self._position_pub = self.create_publisher(
            Float64MultiArray, f"{args.position_controller}/commands", 10)
        self._velocity_pub = self.create_publisher(
            Float64MultiArray, f"{args.velocity_controller}/commands", 10)
        self._effort_pub = self.create_publisher(
            Float64MultiArray, f"{args.effort_controller}/commands", 10)
        self._kp_pub = self.create_publisher(
            Float64MultiArray, f"{args.kp_controller}/commands", 10)
        self._kd_pub = self.create_publisher(
            Float64MultiArray, f"{args.kd_controller}/commands", 10)

        self.create_subscription(JointState, args.joint_state_topic, self._joint_state_cb, 10)
        self.create_subscription(Imu, args.imu_topic, self._imu_cb, 10)
        self.create_subscription(Twist, args.twist_topic, self._twist_cb, 10)
        self.create_subscription(PoseStamped, args.pose_topic, self._pose_cb, 10)
        command_file_timeout = (
            args.command_file_timeout if args.command_file_timeout is not None
            else (config.command_file_timeout_s if config is not None else 0.5)
        )
        self._command_file = (
            CommandFileReader(args.command_file, command_file_timeout)
            if args.command_file else None
        )

        # A wall timer is intentional.  Policy timers must continue to service
        # the watchdog even when the MuJoCo simulation is paused and /clock is
        # not advancing.
        self._timer = self.create_timer(self.command_period, self._tick)
        self.get_logger().info(
            f"ROS 2 policy node ready: policy_hz={self.policy_hz:g}, "
            f"command_hz={self.command_hz:g}, policy={'loaded' if self.policy else 'zero-action'}"
        )

    def _joint_state_cb(self, message: JointState) -> None:
        with self._state_lock:
            for index, name in enumerate(message.name):
                policy_index = self._joint_indices.get(name)
                if policy_index is None:
                    continue
                if index < len(message.position):
                    self._snapshot.q[policy_index] = float(message.position[index])
                if index < len(message.velocity):
                    self._snapshot.dq[policy_index] = float(message.velocity[index])
                if index < len(message.effort):
                    self._snapshot.tau[policy_index] = float(message.effort[index])
                self._joint_seen[policy_index] = True
            self._last_state = time.monotonic()
            self._snapshot.monotonic_ns = time.monotonic_ns()
            self._snapshot.valid = bool(self._joint_seen.all() and self._imu_seen)
            if not self._snapshot.valid:
                self._snapshot.error = "waiting for complete joint and IMU state"

    def _imu_cb(self, message: Imu) -> None:
        with self._state_lock:
            # sensor_msgs/Imu is xyzw; policy adapters use scalar-first wxyz.
            self._snapshot.quat_wxyz[:] = (
                float(message.orientation.w), float(message.orientation.x),
                float(message.orientation.y), float(message.orientation.z)
            )
            self._snapshot.gyro[:] = (
                float(message.angular_velocity.x), float(message.angular_velocity.y),
                float(message.angular_velocity.z)
            )
            self._snapshot.accel[:] = (
                float(message.linear_acceleration.x), float(message.linear_acceleration.y),
                float(message.linear_acceleration.z)
            )
            self._imu_seen = True
            self._last_state = time.monotonic()
            self._snapshot.monotonic_ns = time.monotonic_ns()
            self._snapshot.valid = bool(self._joint_seen.all())
            if not self._snapshot.valid:
                self._snapshot.error = "waiting for complete joint state"

    def _twist_cb(self, message: Twist) -> None:
        with self._state_lock:
            self.command_velocity[:] = (
                float(message.linear.x), float(message.linear.y), float(message.angular.z)
            )
            self._last_ros_command = time.monotonic()
            self._ros_command_seen = True

    def _pose_cb(self, message: PoseStamped) -> None:
        pose = np.asarray([
            message.pose.position.x, message.pose.position.y, message.pose.position.z,
            message.pose.orientation.w, message.pose.orientation.x,
            message.pose.orientation.y, message.pose.orientation.z,
        ], dtype=np.float32)
        norm = float(np.linalg.norm(pose[3:]))
        if not np.isfinite(pose).all() or norm < 1.0e-8:
            self.get_logger().warning("ignoring invalid EE pose command")
            return
        with self._state_lock:
            self.ee_pose[:] = pose
            self.ee_pose[3:] /= norm
            self._last_ros_command = time.monotonic()
            self._ros_command_seen = True

    def _read_command_file(self) -> None:
        if self._command_file is None:
            # A topic publisher can disappear without sending an explicit
            # stop.  Do not retain a stale base velocity indefinitely in
            # sim2sim or hardware ROS deployments.  The end-effector target
            # is intentionally held, matching CommandFileReader's fail-safe
            # behavior and avoiding an abrupt arm target reset.
            with self._state_lock:
                if (self._ros_command_seen and
                        time.monotonic() - self._last_ros_command > self.command_timeout):
                    self.command_velocity.fill(0.0)
            return
        # ROS command topics take precedence while the keyboard node is alive;
        # the file remains a compatibility bridge for non-ROS command writers.
        if time.monotonic() - self._last_ros_command <= self.command_timeout:
            return
        command = self._command_file.read()
        if command is not None:
            velocity, pose = command
            with self._state_lock:
                self.command_velocity[:] = velocity
                self.ee_pose[:] = pose

    @staticmethod
    def _message(values: np.ndarray) -> Float64MultiArray:
        message = Float64MultiArray()
        message.data = [float(value) for value in np.asarray(values).reshape(-1)]
        return message

    def _publish_command(self, command: JointCommand) -> None:
        self._position_pub.publish(self._message(command.position))
        self._velocity_pub.publish(self._message(command.velocity))
        self._effort_pub.publish(self._message(command.torque))
        self._kp_pub.publish(self._message(command.kp))
        self._kd_pub.publish(self._message(command.kd))

    def _zero_command(self) -> JointCommand:
        return JointCommand(
            position=np.zeros(JOINT_COUNT, dtype=np.float32),
            velocity=np.zeros(JOINT_COUNT, dtype=np.float32),
            kp=np.zeros(JOINT_COUNT, dtype=np.float32),
            kd=np.zeros(JOINT_COUNT, dtype=np.float32),
            torque=np.zeros(JOINT_COUNT, dtype=np.float32),
        )

    def _tick(self) -> None:
        if not self._running:
            return
        self._read_command_file()
        now = time.monotonic()
        with self._state_lock:
            snapshot = self._snapshot.copy()
            velocity = self.command_velocity.copy()
            pose = self.ee_pose.copy()
            state_age = now - self._last_state if self._last_state else float("inf")

        # No state or a stale state is a fail-safe zero command.  Once state is
        # valid, the last policy command is held at the command publication
        # rate between 50-Hz inference updates.
        if not snapshot.valid or state_age > self.command_timeout:
            self._last_command = self._zero_command()
        elif now - self._last_policy >= self.policy_period:
            try:
                _frame, _history, self._last_command = self.runtime.step(
                    snapshot, velocity, pose)
                self._last_policy = now
            except Exception as exc:
                self.get_logger().error(f"policy step failed; entering zero-command mode: {exc}")
                self._last_command = self._zero_command()
                self._last_policy = now
        self._publish_command(self._last_command)

        if now - self._last_report >= 1.0:
            self.get_logger().info(
                f"state={'ready' if snapshot.valid else 'waiting'} "
                f"age={state_age:.3f}s max_action={np.max(np.abs(self.runtime.action)):.3f}"
            )
            self._last_report = now

    def close(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._publish_command(self._zero_command())
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="")
    parser.add_argument("--policy-path", default="")
    parser.add_argument("--action-clip", type=float, default=None,
                        help="raw action clip; omitted uses policy.action_clip")
    parser.add_argument("--policy-hz", type=float, default=None,
                        help="policy rate; omitted uses policy.frequency_hz")
    parser.add_argument("--command-hz", type=float, default=None,
                        help="command publication rate; omitted uses runtime.command_hz")
    parser.add_argument("--command-velocity", default=None,
                        help="vx,vy,yaw-rate; omitted uses commands.velocity.default from YAML")
    parser.add_argument("--ee-pose", default=None,
                        help="x,y,z,qw,qx,qy,qz; omitted uses commands.ee_pose.default from YAML")
    parser.add_argument("--command-file", default="")
    parser.add_argument("--command-file-timeout", type=float, default=None,
                        help="command file timeout; omitted uses runtime.command_file_timeout_s")
    parser.add_argument("--command-timeout", type=float, default=None,
                        help="policy/state watchdog; omitted uses runtime.policy_watchdog_timeout_ms")
    parser.add_argument("--joint-state-topic", default="joint_states")
    parser.add_argument("--imu-topic", default="imu_sensor_broadcaster/imu")
    parser.add_argument("--twist-topic", default="command/cmd_twist")
    parser.add_argument("--pose-topic", default="command/cmd_pose")
    parser.add_argument("--position-controller", default="loco_mani_position_controller")
    parser.add_argument("--velocity-controller", default="loco_mani_velocity_controller")
    parser.add_argument("--effort-controller", default="loco_mani_effort_controller")
    parser.add_argument("--kp-controller", default="loco_mani_kp_controller")
    parser.add_argument("--kd-controller", default="loco_mani_kd_controller")
    return parser


def main() -> int:
    # ``launch_ros.actions.Node`` appends ``--ros-args`` and remapping
    # arguments to the child command.  Parse only this executable's options;
    # leave ROS arguments for rclpy instead of making argparse reject them.
    args, ros_args = build_parser().parse_known_args()
    rclpy.init(args=ros_args)
    node = D1PiperRos2Policy(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        # launch sends SIGINT to every child.  rclpy may invalidate the
        # context before Python gets here, so cleanup must be best-effort and
        # must not turn a normal launch shutdown into a traceback/non-zero
        # policy process.
        try:
            node.close()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except (KeyboardInterrupt, RuntimeError):
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
