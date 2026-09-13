#!/usr/bin/env python3
"""ctypes wrapper for the flat D1 vendor adapter.

This module intentionally contains no implicit device discovery.  The shared
library path must be supplied through ``LOCO_MANI_D1_ADAPTER_LIB`` (or the
factory argument), and constructing ``CtypesD1Api`` opens the bus.  The
runtime only calls the factory after its explicit hardware gate and complete
calibration checks.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MOTOR_COUNT = 16


class _Motor(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float),
        ("velocity", ctypes.c_float),
        ("torque", ctypes.c_float),
        ("status", ctypes.c_uint16),
    ]


class _Imu(ctypes.Structure):
    _fields_ = [
        ("timestamp", ctypes.c_uint32),
        ("accel", ctypes.c_float * 3),
        ("gyro", ctypes.c_float * 3),
        ("quaternion_xyzw", ctypes.c_float * 4),
    ]


@dataclass(frozen=True)
class MotorRecord:
    """Vendor command record passed to ``send_motors_can``.

    This is deliberately separate from feedback records.  The DDT input and
    output structures happen to have similarly named fields, but their ABI
    and semantics are different; keeping two types prevents a malformed
    feedback packet from being mistaken for a command.
    """
    timestamp: int
    position: float
    kp: float
    velocity: float
    kd: float
    torque: float


@dataclass(frozen=True)
class FeedbackMotorRecord:
    """Normalized read-only D1 feedback record exposed to ``D1Backend``."""

    position: float
    velocity: float
    torque: float


@dataclass(frozen=True)
class ImuRecord:
    timestamp: int
    accl: tuple[float, float, float]
    gyro: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


class CtypesD1Api:
    """Object protocol consumed by :class:`d1_piper_hardware.D1Backend`."""

    def __init__(self, library: str | os.PathLike[str], *, interface: str = "can0") -> None:
        if interface != "can0":
            raise ValueError("D1 adapter is intentionally fixed to can0")
        path = Path(library).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        self._lib = ctypes.CDLL(str(path))
        self._bind()
        self._handle = self._create(MOTOR_COUNT, interface.encode("ascii"))
        if not self._handle:
            raise RuntimeError("loco_mani_d1_create failed; no CAN handle was returned")
        self._motors = (_Motor * MOTOR_COUNT)()
        self._imu = _Imu()
        self._fresh = False

    def _bind(self) -> None:
        pointer = ctypes.c_void_p
        self._create = self._lib.loco_mani_d1_create
        self._create.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        self._create.restype = pointer
        self._destroy = self._lib.loco_mani_d1_destroy
        self._destroy.argtypes = [pointer]
        self._destroy.restype = None
        self._read = self._lib.loco_mani_d1_read
        self._read.argtypes = [pointer, ctypes.POINTER(_Motor), ctypes.POINTER(_Imu)]
        self._read.restype = ctypes.c_int
        self._send = self._lib.loco_mani_d1_send
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self._send.argtypes = [pointer, float_pointer, float_pointer, float_pointer,
                               float_pointer, float_pointer]
        self._send.restype = ctypes.c_int
        self._set_force_direct = getattr(self._lib, "loco_mani_d1_set_force_direct", None)
        if self._set_force_direct is not None:
            self._set_force_direct.argtypes = [pointer]
            self._set_force_direct.restype = ctypes.c_int
        self._motors_timeout = self._lib.loco_mani_d1_is_motors_timeout
        self._motors_timeout.argtypes = [pointer]
        self._motors_timeout.restype = ctypes.c_int
        self._imu_timeout = self._lib.loco_mani_d1_is_imu_timeout
        self._imu_timeout.argtypes = [pointer]
        self._imu_timeout.restype = ctypes.c_int
        self._last_error = self._lib.loco_mani_d1_last_error
        self._last_error.argtypes = [pointer]
        self._last_error.restype = ctypes.c_char_p
        self._feedback_timestamp = getattr(self._lib, "loco_mani_d1_feedback_timestamp", None)
        if self._feedback_timestamp is not None:
            self._feedback_timestamp.argtypes = [pointer]
            self._feedback_timestamp.restype = ctypes.c_uint32

    def _error(self) -> str:
        value = self._last_error(self._handle)
        return value.decode("utf-8", errors="replace") if value else "D1 adapter error"

    def _refresh(self) -> None:
        result = int(self._read(self._handle, self._motors, ctypes.byref(self._imu)))
        if result != 0:
            raise RuntimeError(f"D1 adapter read failed ({result}): {self._error()}")
        self._fresh = True

    def get_motors_in(self) -> list[FeedbackMotorRecord]:
        self._refresh()
        return [FeedbackMotorRecord(
            position=float(item.position),
            velocity=float(item.velocity),
            torque=float(item.torque),
        ) for item in self._motors]

    def get_motors_status(self) -> list[int]:
        if not self._fresh:
            self._refresh()
        return [int(item.status) for item in self._motors]

    def get_imu_data(self) -> ImuRecord:
        if not self._fresh:
            self._refresh()
        # The DDT API documents x/y/z/w, which D1Backend converts to w/x/y/z.
        return ImuRecord(
            timestamp=int(self._imu.timestamp),
            accl=tuple(float(value) for value in self._imu.accel),
            gyro=tuple(float(value) for value in self._imu.gyro),
            quaternion=tuple(float(value) for value in self._imu.quaternion_xyzw),
        )

    def is_motors_timeout(self) -> bool:
        return bool(self._motors_timeout(self._handle))

    def is_imu_timeout(self) -> bool:
        return bool(self._imu_timeout(self._handle))

    def feedback_timestamp(self) -> int | None:
        """Return the most recent vendor feedback timestamp when exported."""
        if getattr(self, "_feedback_timestamp", None) is None:
            return None
        return int(self._feedback_timestamp(self._handle))

    def send_motors_can(self, records: list[Any]) -> bool:
        if len(records) != MOTOR_COUNT:
            raise ValueError(f"D1 command requires {MOTOR_COUNT} records")
        import numpy as np

        position = np.asarray([float(getattr(item, "position")) for item in records], dtype=np.float32)
        velocity = np.asarray([float(getattr(item, "velocity")) for item in records], dtype=np.float32)
        kp = np.asarray([float(getattr(item, "kp")) for item in records], dtype=np.float32)
        kd = np.asarray([float(getattr(item, "kd")) for item in records], dtype=np.float32)
        torque = np.asarray([float(getattr(item, "torque")) for item in records], dtype=np.float32)
        result = int(self._send(self._handle,
                                position.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                kp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                kd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                torque.ctypes.data_as(ctypes.POINTER(ctypes.c_float))))
        if result == 0:
            raise RuntimeError(f"D1 adapter send failed: {self._error()}")
        return True

    def set_force_direct(self) -> bool:
        """Request D1 MCU direct mode through the validated C adapter.

        This method is intentionally separate from construction and ordinary
        reads.  Older adapter libraries may not expose it, in which case the
        caller receives a clear refusal instead of an undocumented RPC guess.
        """
        if self._set_force_direct is None:
            raise RuntimeError("D1 adapter does not expose FORCE_DIRECT RPC")
        result = int(self._set_force_direct(self._handle))
        if result == 0:
            raise RuntimeError(f"D1 FORCE_DIRECT request failed: {self._error()}")
        return True

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle:
            self._destroy(handle)

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        try:
            self.close()
        except Exception:
            pass


def create_d1_api() -> CtypesD1Api:
    """Factory for ``loco_mani_runtime --d1-api-factory``."""
    library = os.environ.get("LOCO_MANI_D1_ADAPTER_LIB", "").strip()
    if not library:
        raise RuntimeError("LOCO_MANI_D1_ADAPTER_LIB is not set")
    return CtypesD1Api(library, interface="can0")


def make_motor_out(**kwargs: Any) -> MotorRecord:
    """Factory passed to ``D1Backend`` for the flat C adapter."""
    return MotorRecord(
        timestamp=int(kwargs.get("timestamp", 0)),
        position=float(kwargs["position"]),
        kp=float(kwargs["kp"]),
        velocity=float(kwargs["velocity"]),
        kd=float(kwargs["kd"]),
        torque=float(kwargs["torque"]),
    )
