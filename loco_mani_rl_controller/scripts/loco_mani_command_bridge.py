#!/usr/bin/env python3
"""Bridge ROS 2 locomotion commands to the standalone hardware runtime.

The real-time hardware process intentionally does not depend on an rclpy
executor.  This small node subscribes to the command topics used by the
existing ``teleop_command`` package and atomically writes the same JSON file
consumed by :mod:`loco_mani_runtime`.

It is deliberately command-only: it never opens either CAN interface and it
does not publish actuator commands.  Run one writer only (this bridge *or*
``loco_mani_keyboard``) for a given command file.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node


DEFAULT_POSE = np.asarray([0.425, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _finite_vector(values: list[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return array


class LocoManiCommandBridge(Node):
    """ROS command subscriber and atomic JSON writer."""

    def __init__(self, *, command_file: str, publish_rate: float = 20.0) -> None:
        super().__init__("loco_mani_command_bridge")
        if not np.isfinite(publish_rate) or publish_rate <= 0.0:
            raise ValueError("publish_rate must be finite and positive")
        self.command_file = Path(command_file).expanduser()
        self.command_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._ee_pose = DEFAULT_POSE.copy()
        self._seq = 0
        self._closed = False
        self.create_subscription(Twist, "command/cmd_twist", self._twist_cb, 10)
        self.create_subscription(PoseStamped, "command/cmd_pose", self._pose_cb, 10)
        self._timer = self.create_timer(1.0 / float(publish_rate), self._tick)
        self._write_command()
        self.get_logger().info(f"writing ROS commands to {self.command_file}")

    def _twist_cb(self, message: Twist) -> None:
        velocity = np.asarray(
            [message.linear.x, message.linear.y, message.angular.z], dtype=np.float64
        )
        if not np.isfinite(velocity).all():
            self.get_logger().warning("ignoring non-finite Twist command")
            return
        with self._lock:
            self._velocity[:] = velocity
        self._write_command()

    def _pose_cb(self, message: PoseStamped) -> None:
        pose = np.asarray(
            [
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
                message.pose.orientation.w,
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(pose).all():
            self.get_logger().warning("ignoring non-finite EE pose command")
            return
        norm = float(np.linalg.norm(pose[3:]))
        if norm < 1.0e-12:
            self.get_logger().warning("ignoring EE pose with zero quaternion")
            return
        pose[3:] /= norm
        with self._lock:
            self._ee_pose[:] = pose
        self._write_command()

    def _write_command(self) -> None:
        with self._lock:
            velocity = self._velocity.copy()
            pose = self._ee_pose.copy()
            self._seq += 1
            sequence = self._seq
        payload = {
            "timestamp_unix": time.time(),
            "seq": sequence,
            "command_velocity": velocity.tolist(),
            "ee_pose": pose.tolist(),
            "source": "loco_mani_command_bridge",
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.command_file.parent),
                prefix=f".{self.command_file.name}.",
                suffix=".tmp",
                delete=False,
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
        if not self._closed:
            self._write_command()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # A bridge shutdown must stop translational/rotational velocity while
        # retaining the last EE target.  The runtime's command timeout then
        # handles the missing heartbeat conservatively.
        with self._lock:
            self._velocity.fill(0.0)
        try:
            self._write_command()
        except Exception as exc:  # pragma: no cover - shutdown race
            self.get_logger().debug(f"final command write skipped: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-file", default="/tmp/loco_mani_command.json")
    parser.add_argument("--publish-rate", type=float, default=20.0)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = LocoManiCommandBridge(
        command_file=args.command_file, publish_rate=args.publish_rate
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        node.close()
        try:
            node.destroy_node()
        except (KeyboardInterrupt, RuntimeError):
            pass
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
