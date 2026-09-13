#!/usr/bin/env python3
"""Keyboard command node for the D1+Piper-L WBC policy.

The node publishes the same ROS command topics used by the legacy controller
and atomically writes a small JSON command file.  The latter is consumed by
the dependency-light MuJoCo/hardware Python runtimes, which may run in a
different Python environment from ROS 2.

Velocity keys (incremental, SI units):
  ``w/s``: forward/backward vx, ``a/d``: left/right vy, ``q/e``: yaw rate wz

End-effector keys (incremental, metres):
  ``j/l``: x, ``u/o``: y, ``i/k``: z

End-effector orientation keys (incremental, radians):
  ``t/g``: roll, ``f/h``: pitch, ``y/n``: yaw

Other keys: ``space`` or ``r`` stops the base, ``0`` resets the EE target,
``x`` exits.  Arrow keys provide an alternative for vx/vy.  Commands are
clamped to conservative limits and the current target is printed after each
key so operation does not depend on a GUI.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import tempfile
import termios
import time
import tty
from pathlib import Path

import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


DEFAULT_EE = np.asarray([0.425, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
DEFAULT_ORIENTATION_LIMIT = 0.6


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(value, upper)))


def rpy_to_quaternion(rpy: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    """Convert XYZ roll/pitch/yaw angles to a normalized ``[w,x,y,z]`` quat.

    The policy/deployment ABI uses scalar-first quaternions.  Keeping this
    conversion here (rather than relying on a ROS helper) also makes the
    command-file path independent of ROS message conventions.
    """
    roll, pitch, yaw = (float(value) for value in np.asarray(rpy).reshape(3))
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    quaternion = np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("RPY conversion produced an invalid quaternion")
    return quaternion / norm


class LocoManiKeyboard(Node):
    """Non-blocking terminal keyboard publisher."""

    def __init__(self, *, command_file: str, publish_rate: float = 20.0,
                 velocity_step: float = 0.1, yaw_step: float = 0.15,
                 ee_step: float = 0.01, orientation_step: float = 0.05,
                 orientation_limit: float = DEFAULT_ORIENTATION_LIMIT,
                 quiet: bool = False) -> None:
        super().__init__("loco_mani_keyboard")
        values = (publish_rate, velocity_step, yaw_step, ee_step,
                  orientation_step, orientation_limit)
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("keyboard rates and steps must be positive")
        self.command_file = Path(command_file).expanduser()
        self.command_file.parent.mkdir(parents=True, exist_ok=True)
        self.velocity_step = float(velocity_step)
        self.yaw_step = float(yaw_step)
        self.ee_step = float(ee_step)
        self.orientation_step = float(orientation_step)
        self.orientation_limit = float(orientation_limit)
        self.quiet = bool(quiet)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.ee_pose = DEFAULT_EE.copy()
        # Keep RPY as the user-facing state.  ee_pose[3:] is regenerated from
        # it after every change so that the published/file quaternion is
        # always normalized and remains in the documented [qw,qx,qy,qz] order.
        self.ee_rpy = np.zeros(3, dtype=np.float64)
        self.seq = 0
        self._stdin_fd: int | None = None
        self._owns_stdin_fd = False
        self._old_termios: list[int] | None = None
        self._escape = ""
        self._closed = False

        self.twist_pub = self.create_publisher(Twist, "command/cmd_twist", 10)
        self.pose_pub = self.create_publisher(PoseStamped, "command/cmd_pose", 10)
        self.timer = self.create_timer(1.0 / float(publish_rate), self._tick)
        self._setup_terminal()
        self._publish()
        self._print_help()

    def _setup_terminal(self) -> None:
        # ``ros2 launch`` normally gives child processes a pipe for stdin,
        # even when launch itself runs in an interactive shell.  Prefer that
        # descriptor when it is a TTY, otherwise open the controlling terminal
        # directly so keyboard input still works from a launch file.
        candidate: int | None = None
        try:
            if sys.stdin.isatty():
                candidate = sys.stdin.fileno()
            else:
                candidate = os.open("/dev/tty", os.O_RDWR | os.O_NONBLOCK)
                self._owns_stdin_fd = True
        except (OSError, AttributeError, ValueError):
            candidate = None
        if candidate is None or not os.isatty(candidate):
            if self._owns_stdin_fd and candidate is not None:
                os.close(candidate)
                self._owns_stdin_fd = False
            self.get_logger().warning(
                "no controlling TTY available; keyboard input is disabled. "
                "Run the keyboard node directly in an interactive terminal."
            )
            return
        self._stdin_fd = candidate
        self._old_termios = termios.tcgetattr(self._stdin_fd)
        tty.setcbreak(self._stdin_fd)

    def _restore_terminal(self) -> None:
        fd = self._stdin_fd
        if self._stdin_fd is not None and self._old_termios is not None:
            try:
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._old_termios)
            except termios.error:
                pass
        self._stdin_fd = None
        self._old_termios = None
        if self._owns_stdin_fd:
            try:
                if fd is not None:
                    os.close(fd)
            except (OSError, TypeError):
                pass
            self._owns_stdin_fd = False

    def _print_help(self) -> None:
        if self.quiet:
            return
        print(
            "[loco_mani_keyboard] w/s:vx  a/d:vy  q/e:wz | "
            "j/l:x  u/o:y  i/k:z | t/g:roll  f/h:pitch  y/n:yaw | "
            "space/r:stop  0:reset EE  x:exit"
        )
        self._print_state()

    def _print_state(self) -> None:
        if not self.quiet:
            print(
                "  cmd v=(%.2f, %.2f, %.2f), ee=(%.3f, %.3f, %.3f), "
                "rpy=(%.1f, %.1f, %.1f)deg"
                % (*self.velocity, *self.ee_pose[:3],
                   *np.rad2deg(self.ee_rpy))
            )

    def _handle_key(self, key: str) -> None:
        if key in ("x", "X", "\x03"):
            self._closed = True
            self.get_logger().info("exit requested; publishing one zero-velocity command")
            self.velocity.fill(0.0)
            self._publish()
            rclpy.shutdown()
            return
        if key in ("w", "W", "\x1b[A"):
            self.velocity[0] += self.velocity_step
        elif key in ("s", "S", "\x1b[B"):
            self.velocity[0] -= self.velocity_step
        elif key in ("a", "A", "\x1b[D"):
            self.velocity[1] += self.velocity_step
        elif key in ("d", "D", "\x1b[C"):
            self.velocity[1] -= self.velocity_step
        elif key in ("q", "Q"):
            self.velocity[2] += self.yaw_step
        elif key in ("e", "E"):
            self.velocity[2] -= self.yaw_step
        elif key in (" ", "r", "R"):
            self.velocity.fill(0.0)
        elif key in ("j", "J"):
            self.ee_pose[0] += self.ee_step
        elif key in ("l", "L"):
            self.ee_pose[0] -= self.ee_step
        elif key in ("u", "U"):
            self.ee_pose[1] += self.ee_step
        elif key in ("o", "O"):
            self.ee_pose[1] -= self.ee_step
        elif key in ("i", "I"):
            self.ee_pose[2] += self.ee_step
        elif key in ("k", "K"):
            self.ee_pose[2] -= self.ee_step
        elif key in ("t", "T"):
            self.ee_rpy[0] += self.orientation_step
        elif key in ("g", "G"):
            self.ee_rpy[0] -= self.orientation_step
        elif key in ("f", "F"):
            self.ee_rpy[1] += self.orientation_step
        elif key in ("h", "H"):
            self.ee_rpy[1] -= self.orientation_step
        elif key in ("y", "Y"):
            self.ee_rpy[2] += self.orientation_step
        elif key in ("n", "N"):
            self.ee_rpy[2] -= self.orientation_step
        elif key in ("0", "c", "C"):
            self.ee_rpy.fill(0.0)
            self.ee_pose[:] = DEFAULT_EE
        else:
            return

        self.velocity[0] = _clamp(self.velocity[0], -1.0, 1.0)
        self.velocity[1] = _clamp(self.velocity[1], -1.0, 1.0)
        self.velocity[2] = _clamp(self.velocity[2], -1.5, 1.5)
        self.ee_pose[0] = _clamp(self.ee_pose[0], 0.15, 0.75)
        self.ee_pose[1] = _clamp(self.ee_pose[1], -0.45, 0.45)
        self.ee_pose[2] = _clamp(self.ee_pose[2], 0.20, 0.90)
        self.ee_rpy[:] = np.clip(self.ee_rpy, -self.orientation_limit, self.orientation_limit)
        self.ee_pose[3:] = rpy_to_quaternion(self.ee_rpy)
        self._publish()
        self._print_state()

    def _poll_keyboard(self) -> None:
        if self._stdin_fd is None:
            return
        while True:
            ready, _, _ = select.select([self._stdin_fd], [], [], 0.0)
            if not ready:
                return
            value = os.read(self._stdin_fd, 1)
            if not value:
                return
            char = value.decode("utf-8", errors="ignore")
            if self._escape or char == "\x1b":
                self._escape += char
                if self._escape == "\x1b":
                    continue
                if len(self._escape) >= 3 or char not in "[O":
                    sequence = self._escape
                    self._escape = ""
                    self._handle_key(sequence)
                continue
            self._handle_key(char)

    def _publish(self) -> None:
        self.seq += 1
        stamp = self.get_clock().now().to_msg()
        twist = Twist()
        twist.linear.x, twist.linear.y = self.velocity[:2]
        twist.angular.z = self.velocity[2]
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "piper_base_link"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = self.ee_pose[:3]
        pose.pose.orientation.w, pose.pose.orientation.x = self.ee_pose[3], self.ee_pose[4]
        pose.pose.orientation.y, pose.pose.orientation.z = self.ee_pose[5], self.ee_pose[6]
        try:
            self.twist_pub.publish(twist)
            self.pose_pub.publish(pose)
        except Exception as exc:
            # During ROS shutdown the context can be invalidated before the
            # timer's final callback.  Continue with the command-file write
            # so the independent policy process still receives a safe stop.
            self.get_logger().debug(f"ROS command publish skipped: {exc}")
        payload = {
            "timestamp_unix": time.time(),
            "seq": self.seq,
            "command_velocity": self.velocity.astype(float).tolist(),
            "ee_pose": self.ee_pose.astype(float).tolist(),
            "source": "loco_mani_keyboard",
        }
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=str(self.command_file.parent),
                prefix=f".{self.command_file.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(payload, stream, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.command_file)
            temporary = None
        except OSError as exc:
            self.get_logger().error(f"cannot write command file {self.command_file}: {exc}")
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def _tick(self) -> None:
        if self._closed:
            return
        self._poll_keyboard()
        # Heartbeat keeps the command file fresh while the node is active.
        # Press space/r before handing control back to a human operator.
        self._publish()

    def close(self) -> None:
        if self._closed:
            self._restore_terminal()
            return
        self._closed = True
        self.velocity.fill(0.0)
        # ROS launch may already have invalidated the rclpy context during
        # shutdown.  The atomic command-file write is still attempted, but a
        # late ROS publish must not turn a normal Ctrl-C into exit code 1.
        try:
            self._publish()
        except Exception as exc:
            self.get_logger().debug(f"final keyboard publish skipped: {exc}")
        self._restore_terminal()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", default="/tmp/loco_mani_command.json")
    parser.add_argument("--publish-rate", type=float, default=20.0)
    parser.add_argument("--velocity-step", type=float, default=0.1)
    parser.add_argument("--yaw-step", type=float, default=0.15)
    parser.add_argument("--ee-step", type=float, default=0.01)
    parser.add_argument("--orientation-step", type=float, default=0.05,
                        help="roll/pitch/yaw increment in radians")
    parser.add_argument("--orientation-limit", type=float, default=DEFAULT_ORIENTATION_LIMIT,
                        help="absolute roll/pitch/yaw limit in radians")
    parser.add_argument("--quiet", action="store_true")
    # launch_ros appends ROS remapping arguments after the script options.
    # Keep argparse focused on keyboard options and pass the rest to rclpy.
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = LocoManiKeyboard(
        command_file=args.command_file, publish_rate=args.publish_rate,
        velocity_step=args.velocity_step, yaw_step=args.yaw_step,
        ee_step=args.ee_step, orientation_step=args.orientation_step,
        orientation_limit=args.orientation_limit, quiet=args.quiet,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
