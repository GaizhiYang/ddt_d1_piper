#!/usr/bin/env python3
"""Small dependency-free live command-file adapter.

The ROS keyboard node runs with the system Python (where ``rclpy`` is
available), while the MuJoCo policy process may run in a separate environment.
This module keeps their process boundary to an atomic JSON file and can be
imported by both the simulation and hardware runtimes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


class CommandFileReader:
    """Read keyboard commands and fail safe on malformed or stale updates."""

    def __init__(self, path: str | Path, timeout_s: float = 0.5) -> None:
        self.path = Path(path)
        self.timeout_s = float(timeout_s)
        if not np.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("command file timeout must be finite and positive")
        # Track ctime as well as mtime.  Some filesystems/co-mounted folders
        # expose coarse mtime resolution; an atomic replacement can therefore
        # have the same mtime while ctime still changes.  Missing an update
        # would defeat the stale-command safety behavior.
        self._file_stamp: tuple[int, int, int] | None = None
        self._last_timestamp = 0.0
        self._command: tuple[np.ndarray, np.ndarray] | None = None

    def _safe_command(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._command is None:
            return None
        # Holding an end-effector target is useful, but a dead keyboard must
        # never leave a locomotion velocity command running indefinitely.
        return np.zeros(3, dtype=np.float32), self._command[1].copy()

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        now = time.time()
        if self._command is not None and now - self._last_timestamp > self.timeout_s:
            self._command = self._safe_command()
            self._last_timestamp = now

        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return self._command
        stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        # Do not rely solely on inode timestamps here.  Some overlay/network
        # filesystems have coarse timestamp resolution, and an in-place writer
        # can replace a stale command with a same-size file before either
        # mtime or ctime visibly changes.  Commands are a few hundred bytes
        # and are read at policy rate (not the 500-Hz command rate), so parsing
        # the unchanged file again is a worthwhile safety tradeoff.
        self._file_stamp = stamp
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            velocity = np.asarray(payload["command_velocity"], dtype=np.float32).reshape(-1)
            pose = np.asarray(payload["ee_pose"], dtype=np.float32).reshape(-1)
            if velocity.size != 3 or pose.size != 7:
                raise ValueError("command file must contain velocity[3] and ee_pose[7]")
            if not np.isfinite(velocity).all() or not np.isfinite(pose).all():
                raise ValueError("command file contains NaN or infinity")
            timestamp = float(payload.get("timestamp_unix", now))
            if not np.isfinite(timestamp) or abs(now - timestamp) > self.timeout_s:
                raise ValueError("command file update is stale")
            quat_norm = float(np.linalg.norm(pose[3:]))
            if quat_norm < 1.0e-8:
                raise ValueError("command file quaternion is zero")
            pose[3:] /= quat_norm
            self._command = velocity, pose
            self._last_timestamp = timestamp
        except ValueError as exc:
            # An explicitly stale writer update is a safety event: stop base
            # motion immediately while retaining the EE target.  Other value
            # errors (for example a transient partial write) are ignored.
            if "stale" in str(exc):
                self._command = self._safe_command()
                self._last_timestamp = now
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            # A writer may be between temporary-file creation and os.replace;
            # retain the last valid command and wait for the next mtime change.
            pass
        return self._command


def load_live_command(path: str | Path, timeout_s: float = 0.5):
    """Convenience helper used by tests and small integrations."""
    return CommandFileReader(path, timeout_s).read()
