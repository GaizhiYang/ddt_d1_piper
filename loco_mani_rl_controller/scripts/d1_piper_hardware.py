#!/usr/bin/env python3
"""Hardware-independent double-CAN adapters for D1 + Piper-L.

This module is deliberately conservative.  It contains the state/command
contract and the safety gate used by a future real-time node, but importing it
or constructing a backend never opens ``can0``/``can1``.  A D1 vendor API is
passed in by an adapter object because the shipped ``libtita_robot.so`` is a
C++ ABI, while Piper is accessed through the Python ``pyAgxArm`` SDK.

The intended process architecture is one non-blocking state thread per bus,
one 500-Hz command loop, and a 50-Hz policy loop.  The classes here are also
useful in unit tests and shadow mode where ``dry_run=True``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

try:
    from loco_mani_config import (
        ACTION_SCALE as CONFIG_ACTION_SCALE,
        DEFAULT_Q as CONFIG_DEFAULT_Q,
        KD as CONFIG_KD,
        KP as CONFIG_KP,
        WHEEL_INDICES as CONFIG_WHEEL_INDICES,
        PolicyConfig,
        load_resolved_config,
        sanitize_commands,
    )
except ImportError:  # source-tree and installed-script fallback
    _config_path = Path(__file__).with_name("loco_mani_config.py")
    _config_spec = importlib.util.spec_from_file_location("loco_mani_config", _config_path)
    if _config_spec is None or _config_spec.loader is None:
        raise ImportError(f"cannot load {_config_path}")
    _config_module = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = _config_module
    _config_spec.loader.exec_module(_config_module)
    CONFIG_ACTION_SCALE = _config_module.ACTION_SCALE
    CONFIG_DEFAULT_Q = _config_module.DEFAULT_Q
    CONFIG_KD = _config_module.KD
    CONFIG_KP = _config_module.KP
    CONFIG_WHEEL_INDICES = _config_module.WHEEL_INDICES
    PolicyConfig = _config_module.PolicyConfig
    load_resolved_config = _config_module.load_resolved_config
    sanitize_commands = _config_module.sanitize_commands


JOINT_COUNT = 22
D1_COUNT = 16
PIPER_COUNT = 6
WHEEL_INDICES = CONFIG_WHEEL_INDICES.copy()
DEFAULT_Q = CONFIG_DEFAULT_Q.copy()
ACTION_SCALE = CONFIG_ACTION_SCALE.copy()
KP = CONFIG_KP.copy()
KD = CONFIG_KD.copy()


@dataclasses.dataclass
class StateSnapshot:
    """A timestamped, policy-order state snapshot.

    Quaternion is always converted to ``[w, x, y, z]`` here.  D1's native
    ``CanfdApi`` reports ``[x, y, z, w]`` and Piper does not provide the base
    IMU; keeping conversion at this boundary prevents observation code from
    depending on a bus-specific convention.
    """

    q: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    dq: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    tau: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    quat_wxyz: np.ndarray = dataclasses.field(
        default_factory=lambda: np.asarray([1.0, 0.0, 0.0, 0.0], np.float32)
    )
    gyro: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3, np.float32))
    accel: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3, np.float32))
    monotonic_ns: int = 0
    valid: bool = False
    error: str = ""

    def copy(self) -> "StateSnapshot":
        return StateSnapshot(
            q=self.q.copy(), dq=self.dq.copy(), tau=self.tau.copy(),
            quat_wxyz=self.quat_wxyz.copy(), gyro=self.gyro.copy(),
            accel=self.accel.copy(), monotonic_ns=int(self.monotonic_ns),
            valid=bool(self.valid), error=str(self.error),
        )


@dataclasses.dataclass
class JointCommand:
    """One policy-order command after action scaling and safety limiting."""

    position: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    velocity: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    kp: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    kd: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))
    torque: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(JOINT_COUNT, np.float32))

    def validate(self) -> None:
        fields = (self.position, self.velocity, self.kp, self.kd, self.torque)
        if any(np.asarray(value).shape != (JOINT_COUNT,) for value in fields):
            raise ValueError("all command fields must have shape (22,)")
        if not all(np.isfinite(value).all() for value in fields):
            raise ValueError("command contains NaN or infinity")


@dataclasses.dataclass(frozen=True)
class JointCalibration:
    """Explicit policy-to-bus calibration for all 22 active joints.

    ``bus_index`` is the index expected by the vendor API (D1 is 0..15,
    Piper is 0..5), while ``can_id`` is retained as an audit field.  The
    encoder convention is ``q_bus = direction * q_policy + offset``.  Thus
    feedback is converted with ``q_policy = direction * (q_bus - offset)``;
    velocity and torque use the same sign and no offset.

    Identity calibration is only intended for dry-run tests.  A non-dry
    hardware connection must use :meth:`from_config(..., require_complete=True)`
    so that no physical mapping is guessed from XML or list order.
    """

    bus_index: np.ndarray
    can_id: np.ndarray
    direction: np.ndarray
    position_offset_rad: np.ndarray

    @property
    def complete(self) -> bool:
        """Whether this mapping is safe to use for a real bus connection."""
        if not (
            self.bus_index.shape == (JOINT_COUNT,)
            and self.can_id.shape == (JOINT_COUNT,)
            and self.direction.shape == (JOINT_COUNT,)
            and self.position_offset_rad.shape == (JOINT_COUNT,)
        ):
            return False
        d1_index, piper_index = self.bus_index[:D1_COUNT], self.bus_index[D1_COUNT:]
        d1_ids, piper_ids = self.can_id[:D1_COUNT], self.can_id[D1_COUNT:]
        return bool(
            np.all((0 <= d1_index) & (d1_index < D1_COUNT))
            and np.all((0 <= piper_index) & (piper_index < PIPER_COUNT))
            and len(set(d1_index.tolist())) == D1_COUNT
            and len(set(piper_index.tolist())) == PIPER_COUNT
            and np.all(d1_ids >= 0)
            and np.all(piper_ids >= 0)
            and len(set(d1_ids.tolist())) == D1_COUNT
            and len(set(piper_ids.tolist())) == PIPER_COUNT
            and np.all(np.isin(self.direction, (-1.0, 1.0)))
            and np.isfinite(self.position_offset_rad).all()
        )

    @classmethod
    def identity(cls) -> "JointCalibration":
        return cls(
            bus_index=np.r_[np.arange(D1_COUNT), np.arange(PIPER_COUNT)].astype(np.int64),
            can_id=np.full(JOINT_COUNT, -1, dtype=np.int64),
            direction=np.ones(JOINT_COUNT, dtype=np.float32),
            position_offset_rad=np.zeros(JOINT_COUNT, dtype=np.float32),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any], *, require_complete: bool = False) -> "JointCalibration":
        validate_joint_calibration(config, require_complete=require_complete)
        joints = config["joints"]
        entries = config["joint_calibration"]
        for name in joints:
            bus = entries[name].get("bus")
            index = entries[name].get("bus_index")
            if index is not None:
                limit = D1_COUNT if bus == "d1" else PIPER_COUNT
                if not isinstance(index, int) or not 0 <= index < limit:
                    raise ValueError(f"{name}: bus_index must be in [0, {limit - 1}]")
        if not require_complete and any(
            entries[name].get("bus_index") is None
            or entries[name].get("can_id") is None
            or entries[name].get("direction") is None
            or entries[name].get("position_offset_rad") is None
            for name in joints
        ):
            return cls.identity()
        bus_index = np.asarray([int(entries[name]["bus_index"]) for name in joints], np.int64)
        can_id = np.asarray([int(entries[name]["can_id"]) for name in joints], np.int64)
        direction = np.asarray([float(entries[name]["direction"]) for name in joints], np.float32)
        offset = np.asarray([float(entries[name]["position_offset_rad"]) for name in joints], np.float32)
        if np.any(bus_index[:D1_COUNT] < 0) or np.any(bus_index[:D1_COUNT] >= D1_COUNT):
            raise ValueError("D1 bus_index values must be in [0, 15]")
        if np.any(bus_index[D1_COUNT:] < 0) or np.any(bus_index[D1_COUNT:] >= PIPER_COUNT):
            raise ValueError("Piper bus_index values must be in [0, 5]")
        if len(set(bus_index[:D1_COUNT].tolist())) != D1_COUNT or len(set(bus_index[D1_COUNT:].tolist())) != PIPER_COUNT:
            raise ValueError("bus_index values must be unique within each bus")
        if not np.isfinite(offset).all() or not np.isfinite(direction).all():
            raise ValueError("calibration contains non-finite direction/offset")
        return cls(bus_index=bus_index, can_id=can_id, direction=direction,
                   position_offset_rad=offset)

    def _indices(self, bus: str) -> tuple[np.ndarray, np.ndarray]:
        if bus == "d1":
            policy = np.arange(D1_COUNT, dtype=np.int64)
        elif bus == "piper":
            policy = np.arange(D1_COUNT, JOINT_COUNT, dtype=np.int64)
        else:
            raise ValueError(f"unknown calibration bus {bus!r}")
        bus_indices = self.bus_index[policy]
        order = np.argsort(bus_indices)
        return policy[order], bus_indices[order]

    def feedback(self, q_bus: Sequence[float], dq_bus: Sequence[float],
                 tau_bus: Sequence[float], *, bus: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_bus = np.asarray(q_bus, np.float32).reshape(-1)
        dq_bus = np.asarray(dq_bus, np.float32).reshape(-1)
        tau_bus = np.asarray(tau_bus, np.float32).reshape(-1)
        expected = D1_COUNT if bus == "d1" else PIPER_COUNT
        if q_bus.size != expected or dq_bus.size != expected or tau_bus.size != expected:
            raise ValueError(f"{bus} feedback must contain {expected} values")
        policy, bus_indices = self._indices(bus)
        q = np.zeros(expected, np.float32)
        dq = np.zeros(expected, np.float32)
        tau = np.zeros(expected, np.float32)
        local = policy - (0 if bus == "d1" else D1_COUNT)
        q[local] = self.direction[policy] * (q_bus[bus_indices] - self.position_offset_rad[policy])
        dq[local] = self.direction[policy] * dq_bus[bus_indices]
        tau[local] = self.direction[policy] * tau_bus[bus_indices]
        return q, dq, tau

    def command(self, position: Sequence[float], velocity: Sequence[float],
                torque: Sequence[float], *, bus: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = np.asarray(position, np.float32).reshape(-1)
        velocity = np.asarray(velocity, np.float32).reshape(-1)
        torque = np.asarray(torque, np.float32).reshape(-1)
        expected = D1_COUNT if bus == "d1" else PIPER_COUNT
        if position.size != expected or velocity.size != expected or torque.size != expected:
            raise ValueError(f"{bus} command must contain {expected} values")
        policy, bus_indices = self._indices(bus)
        p = np.zeros(expected, np.float32)
        v = np.zeros(expected, np.float32)
        t = np.zeros(expected, np.float32)
        local = policy - (0 if bus == "d1" else D1_COUNT)
        p[bus_indices] = self.direction[policy] * position[local] + self.position_offset_rad[policy]
        v[bus_indices] = self.direction[policy] * velocity[local]
        t[bus_indices] = self.direction[policy] * torque[local]
        return p, v, t


def merge_snapshots(d1: StateSnapshot, piper: StateSnapshot) -> StateSnapshot:
    """Merge independently timestamped bus snapshots in policy joint order."""
    if not d1.valid:
        return d1.copy()
    if not piper.valid:
        result = d1.copy()
        result.valid = False
        result.error = "invalid Piper snapshot: " + piper.error
        return result
    result = d1.copy()
    result.q[16:] = piper.q[16:]
    result.dq[16:] = piper.dq[16:]
    result.tau[16:] = piper.tau[16:]
    # The merged timestamp is the older of the two samples.  This makes the
    # watchdog conservative when one bus lags behind the other.
    result.monotonic_ns = min(int(d1.monotonic_ns), int(piper.monotonic_ns))
    result.error = ""
    return result


class ObservationAdapter:
    """Build the exact 82-D frame and term-major 3-frame policy history."""

    FRAME_DIM = 82
    HISTORY_LEN = 3
    _TERM_SLICES = (
        slice(0, 3), slice(3, 6), slice(6, 28), slice(28, 50),
        slice(50, 72), slice(72, 75), slice(75, 82),
    )

    @staticmethod
    def _projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_wxyz, np.float32).reshape(-1)
        if quat.size != 4 or not np.isfinite(quat).all():
            raise ValueError("IMU quaternion must contain four finite values")
        norm = float(np.linalg.norm(quat))
        if norm <= 1.0e-8:
            raise ValueError("IMU quaternion norm must be non-zero")
        qw, qx, qy, qz = quat / norm
        rotation = np.asarray([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ], np.float32)
        return rotation.T @ np.asarray([0.0, 0.0, -1.0], np.float32)

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self.default_q = np.asarray(self.config.default_q, np.float32).copy()
        self.wheel_indices = np.asarray(self.config.wheel_indices, np.int64).copy()
        self.observation_scales = np.asarray(self.config.observation_scales, np.float32).copy()
        self.history_length = int(self.config.history_length)
        self.observation_noise_ranges = tuple(self.config.observation_noise_ranges)
        if (self.default_q.shape != (JOINT_COUNT,) or self.wheel_indices.size != 4
                or self.observation_scales.shape != (7,) or self.history_length <= 0):
            raise ValueError("invalid observation configuration dimensions")
        self.last_action = np.zeros(JOINT_COUNT, np.float32)
        self._history: list[np.ndarray | None] = [None] * len(self._TERM_SLICES)

    @property
    def history(self) -> np.ndarray | None:
        if any(item is None for item in self._history):
            return None
        return np.concatenate([item.reshape(-1) for item in self._history if item is not None]).astype(np.float32)

    def reset(self) -> None:
        self.last_action.fill(0.0)
        self._history = [None] * len(self._TERM_SLICES)

    def frame(self, snapshot: StateSnapshot, command_velocity: Sequence[float],
              ee_pose: Sequence[float]) -> np.ndarray:
        if not snapshot.valid:
            raise RuntimeError("cannot build observation from invalid state: " + snapshot.error)
        velocity = np.asarray(command_velocity, np.float32)
        pose = np.asarray(ee_pose, np.float32)
        if velocity.shape != (3,) or pose.shape != (7,):
            raise ValueError("command_velocity must be (3,) and ee_pose must be (7,)")
        q = np.asarray(snapshot.q, np.float32).copy()
        dq = np.asarray(snapshot.dq, np.float32)
        q_rel = q - self.default_q
        q_rel[self.wheel_indices] = 0.0
        terms = [
            np.asarray(snapshot.gyro, np.float32),
            self._projected_gravity(snapshot.quat_wxyz), q_rel, dq,
            self.last_action, velocity, pose,
        ]
        frame = np.concatenate([
            np.asarray(term, dtype=np.float32) * self.observation_scales[index]
            for index, term in enumerate(terms)
        ]).astype(np.float32)
        if frame.shape != (self.FRAME_DIM,) or not np.isfinite(frame).all():
            raise RuntimeError("hardware observation is not finite 82-D")
        return np.clip(frame, -self.config.observation_clip, self.config.observation_clip)

    def append(self, frame: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, np.float32)
        if frame.shape != (self.FRAME_DIM,) or not np.isfinite(frame).all():
            raise ValueError("frame must be a finite 82-D vector")
        blocks = []
        for index, term_slice in enumerate(self._TERM_SLICES):
            term = frame[term_slice]
            old = self._history[index]
            if old is None:
                old = np.tile(term, (self.history_length, 1))
            else:
                old = np.concatenate([old[1:], term[None, :]], axis=0)
            self._history[index] = old
            blocks.append(old.reshape(-1))
        return np.concatenate(blocks).astype(np.float32)


class ActionAdapter:
    """Map raw policy actions to explicit D1 and Piper command fields."""

    @staticmethod
    def to_command(action: Sequence[float], snapshot: StateSnapshot,
                   config: PolicyConfig | None = None) -> JointCommand:
        raw = np.asarray(action, np.float32)
        if raw.shape != (JOINT_COUNT,) or not np.isfinite(raw).all():
            raise ValueError("action must be a finite 22-D vector")
        command = JointCommand()
        cfg = config or PolicyConfig()
        default_q = np.asarray(cfg.default_q, np.float32)
        action_scale = np.asarray(cfg.action_scale, np.float32)
        kp = np.asarray(cfg.kp, np.float32)
        kd = np.asarray(cfg.kd, np.float32)
        wheel_indices = np.asarray(cfg.wheel_indices, np.int64)
        command.position[:] = default_q + raw * action_scale
        command.velocity[:] = 0.0
        command.kp[:] = kp
        command.kd[:] = kd
        # Wheel terms are velocity actions, not position actions.  Holding the
        # measured wheel angle while sending kp=0 avoids an accidental brake.
        command.position[wheel_indices] = snapshot.q[wheel_indices]
        command.velocity[wheel_indices] = raw[wheel_indices] * action_scale[wheel_indices]
        # The policy action is a target, not a feed-forward torque.  The D1
        # low-level PD and Piper MIT controllers close the loop on p/v/kp/kd.
        command.torque.fill(0.0)
        command.validate()
        return command


class HardwarePolicyRuntime:
    """Transport-independent policy step used by shadow and future CAN nodes."""

    def __init__(self, policy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 *, max_action_abs: float = 10.0,
                 config: PolicyConfig | None = None) -> None:
        if not np.isfinite(max_action_abs) or max_action_abs <= 0:
            raise ValueError("max_action_abs must be a finite positive value")
        self.policy_fn = policy_fn
        self.config = config or PolicyConfig()
        self.max_action_abs = float(max_action_abs)
        self.observations = ObservationAdapter(self.config)
        self.action = np.zeros(JOINT_COUNT, np.float32)

    def reset(self) -> None:
        self.observations.reset()
        self.action.fill(0.0)

    def step(self, snapshot: StateSnapshot, command_velocity: Sequence[float],
             ee_pose: Sequence[float]) -> tuple[np.ndarray, np.ndarray, JointCommand]:
        command_velocity, ee_pose = sanitize_commands(
            command_velocity, ee_pose, self.config)
        frame = self.observations.frame(snapshot, command_velocity, ee_pose)
        history = self.observations.append(frame)
        action = np.asarray(self.policy_fn(frame, history), np.float32).reshape(-1)
        if action.size != JOINT_COUNT or not np.isfinite(action).all():
            raise RuntimeError("policy_fn must return a finite 22-D action")
        # The training export permits a wide raw action range, while hardware
        # deployment uses a conservative configurable gate (10 by default).
        action = np.clip(action, -self.max_action_abs, self.max_action_abs)
        self.action = action.copy()
        self.observations.last_action = self.action.copy()
        return frame, history, ActionAdapter.to_command(self.action, snapshot, self.config)


class OnnxPolicy:
    """Side-effect-free ONNX callable implementing the hardware policy ABI.

    The constructor only reads the model file and creates an inference
    session; it never opens either CAN interface.  Both the current flattened
    ``obs[246]`` export and legacy two-input ``frame/history`` exports are
    accepted.  ``expected_sha256`` can be supplied from the deployment YAML
    to prevent accidentally running a different checkpoint on the robot.
    """

    def __init__(self, path: str | Path, *, expected_sha256: str | None = None,
                 providers: Sequence[str] | None = None,
                 config: PolicyConfig | None = None) -> None:
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest.lower() != str(expected_sha256).lower():
            raise ValueError(f"policy SHA256 mismatch: expected {expected_sha256}, got {digest}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("onnxruntime is required for hardware policy inference") from exc
        self.path = model_path
        self.sha256 = digest
        self.history_length = int((config or PolicyConfig()).history_length)
        self.session = ort.InferenceSession(
            str(model_path), providers=list(providers) if providers else ["CPUExecutionProvider"]
        )
        self.inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(self.inputs) not in (1, 2) or not outputs:
            raise ValueError(f"expected one or two policy inputs and at least one output, got {len(self.inputs)}")
        if any(getattr(item, "type", "") not in ("", "tensor(float)") for item in self.inputs):
            raise ValueError("policy inputs must be float32 tensors")
        if getattr(outputs[0], "type", "") not in ("", "tensor(float)"):
            raise ValueError("policy output must be a float32 tensor")
        output_shape = tuple(int(x) for x in outputs[0].shape if isinstance(x, (int, np.integer)))
        if output_shape and output_shape[-1] != JOINT_COUNT:
            raise ValueError(f"policy output width must be {JOINT_COUNT}, got {output_shape}")
        self.input_names = [item.name for item in self.inputs]
        self.output_name = outputs[0].name
        self.input_widths = [
            next((int(x) for x in reversed(item.shape) if isinstance(x, (int, np.integer))), None)
            for item in self.inputs
        ]

    def __call__(self, frame: np.ndarray, history: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame, np.float32).reshape(-1)
        history = np.asarray(history, np.float32).reshape(-1)
        expected_history = ObservationAdapter.FRAME_DIM * self.history_length
        if frame.size != ObservationAdapter.FRAME_DIM or history.size != expected_history:
            raise ValueError("policy callable requires frame[82] and history[246]")
        if len(self.inputs) == 1:
            width = self.input_widths[0]
            if width not in (None, history.size, frame.size):
                raise ValueError(f"unsupported policy input width {width}")
            value = frame if width == frame.size else history
            result = self.session.run([self.output_name], {self.input_names[0]: value[None, :]})[0]
        else:
            values = (frame, history)
            feed = {}
            for index, name in enumerate(self.input_names):
                width = self.input_widths[index]
                value = values[index] if width not in (frame.size, history.size) else (frame if width == frame.size else history)
                feed[name] = value[None, :]
            result = self.session.run([self.output_name], feed)[0]
        action = np.asarray(result, np.float32).reshape(-1)
        if action.size != JOINT_COUNT or not np.isfinite(action).all():
            raise RuntimeError("ONNX policy returned an invalid 22-D action")
        return action


def validate_joint_calibration(config: dict[str, Any], *, require_complete: bool = False) -> None:
    """Validate explicit 22-joint bus/sign/offset metadata before hardware use."""
    calibration = config.get("joint_calibration", {})
    joints = config.get("joints", [])
    if len(joints) != JOINT_COUNT or set(joints) != set(calibration):
        raise ValueError("joint_calibration must contain exactly the 22 policy joints")
    d1_indices: list[int] = []
    piper_indices: list[int] = []
    d1_ids: list[int] = []
    piper_ids: list[int] = []
    for name in joints:
        item = calibration[name]
        bus = item.get("bus")
        if bus not in {"d1", "piper"}:
            raise ValueError(f"{name}: bus must be d1 or piper")
        expected_bus = "d1" if joints.index(name) < D1_COUNT else "piper"
        if bus != expected_bus:
            raise ValueError(f"{name}: bus must be {expected_bus} for the fixed policy order")
        if require_complete:
            index = item.get("bus_index")
            if not isinstance(index, int):
                raise ValueError(f"{name}: bus_index must be filled before hardware use")
            limit = D1_COUNT if bus == "d1" else PIPER_COUNT
            if not 0 <= index < limit:
                raise ValueError(f"{name}: bus_index must be in [0, {limit - 1}]")
            can_id = item.get("can_id")
            if not isinstance(can_id, int) or can_id < 0:
                raise ValueError(f"{name}: can_id must be filled before hardware use")
            if item.get("direction") not in {-1, 1}:
                raise ValueError(f"{name}: direction must be +1 or -1 before hardware use")
            offset = item.get("position_offset_rad")
            if not isinstance(offset, (int, float)) or not math.isfinite(float(offset)):
                raise ValueError(f"{name}: position_offset_rad must be filled before hardware use")
            if bus == "d1":
                d1_indices.append(index)
                d1_ids.append(can_id)
            else:
                piper_indices.append(index)
                piper_ids.append(can_id)
    if require_complete:
        if len(d1_indices) != D1_COUNT or len(set(d1_indices)) != D1_COUNT:
            raise ValueError("D1 bus_index values must be a complete unique permutation of [0, 15]")
        if len(piper_indices) != PIPER_COUNT or len(set(piper_indices)) != PIPER_COUNT:
            raise ValueError("Piper bus_index values must be a complete unique permutation of [0, 5]")
        if len(set(d1_ids)) != D1_COUNT:
            raise ValueError("D1 CAN IDs must be unique")
        if len(set(piper_ids)) != PIPER_COUNT:
            raise ValueError("Piper CAN IDs must be unique")


class D1Api(Protocol):
    """Minimal object protocol expected from a D1 vendor adapter.

    The concrete adapter may wrap ``can_device::CanfdApi`` through pybind11,
    ctypes or a ROS2 component.  Methods return the structures documented in
    ``hardware/examples/README.md`` (16 motor records and one IMU record).
    """

    def get_motors_in(self) -> Any: ...
    def get_motors_status(self) -> Any: ...
    def get_imu_data(self) -> Any: ...
    # Some released DDT headers expose only ``send_motors_can`` while older
    # ``hardware_bridge`` revisions expose ``send_leg_motors_can``.  The
    # concrete binding may implement either one (or both); D1Backend.send()
    # selects the available method at runtime.
    def send_motors_can(self, motors: Any) -> bool: ...

    def send_leg_motors_can(self, motors: Any, leg_index: int) -> bool: ...

    def set_force_direct(self) -> bool: ...

    def feedback_timestamp(self) -> int | None: ...


def _field(obj: Any, name: str, default: Any = None) -> Any:
    # The released C++ ``CanfdApi`` returns pointers for vectors and IMU data
    # (``const T *``), while pybind adapters may expose either the pointed-to
    # object or a wrapper with ``contents``.  Normalize both forms at this
    # single boundary so the policy code never depends on binding details.
    if hasattr(obj, "contents"):
        obj = obj.contents
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_records(value: Any) -> list[Any]:
    # The C++ API may be exposed as a pointer/reference by a binding.
    if value is None:
        return []
    if hasattr(value, "contents"):
        value = value.contents
    try:
        return list(value)
    except TypeError:
        return []


class D1Backend:
    """D1 ``can0`` backend with an explicit vendor-API injection point."""

    def __init__(self, api_factory: Callable[[], D1Api] | None = None,
                 motor_out_factory: Callable[..., Any] | None = None,
                 *, interface: str = "can0", motor_count: int = D1_COUNT,
                 dry_run: bool = True,
                 calibration: JointCalibration | None = None,
                 config: PolicyConfig | None = None,
                 feedback_timeout_s: float = 0.020) -> None:
        if interface != "can0":
            raise ValueError("D1 backend is intentionally fixed to can0")
        if motor_count != D1_COUNT:
            raise ValueError(f"D1 backend requires exactly {D1_COUNT} motors")
        self.api_factory = api_factory
        # ``api_motor_out_t`` is a C++ type and is not constructible in a
        # portable way from Python.  Require the deployment adapter to inject
        # a factory rather than silently guessing a ctypes/pybind ABI.
        self.motor_out_factory = motor_out_factory
        self.interface = interface
        self.motor_count = motor_count
        self.dry_run = bool(dry_run)
        self.config = config or PolicyConfig()
        if not np.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be finite and positive")
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.calibration = calibration or JointCalibration.identity()
        self.api: D1Api | None = None
        # The runtime has an independent feedback reader and command loop.
        # Keep vendor calls serialized even when a deployment injects a
        # pybind adapter whose thread-safety is weaker than the flat-C
        # adapter shipped in this repository.
        self._io_lock = threading.RLock()
        self.connected = False
        self.read_only = False
        self.last_snapshot = StateSnapshot()
        self._last_vendor_timestamp: int | None = None
        self._last_vendor_timestamp_ns = 0

    def connect(self, *, hardware_gate: bool = False, read_only: bool = False) -> None:
        with self._io_lock:
            if self.dry_run:
                self.connected = True
                self._last_vendor_timestamp = None
                self._last_vendor_timestamp_ns = time.monotonic_ns()
                # A dry-run snapshot should represent the trained standing
                # pose, rather than an all-zero encoder packet.  This keeps
                # shadow-mode observations meaningful while still never
                # opening can0.
                self.last_snapshot.q[:D1_COUNT] = np.asarray(self.config.default_q[:D1_COUNT], np.float32)
                self.last_snapshot.dq.fill(0.0)
                self.last_snapshot.tau.fill(0.0)
                self.last_snapshot.quat_wxyz[:] = (1.0, 0.0, 0.0, 0.0)
                self.last_snapshot.gyro.fill(0.0)
                self.last_snapshot.accel[:] = (0.0, 0.0, 9.81)
                self.last_snapshot.valid = True
                self.last_snapshot.monotonic_ns = time.monotonic_ns()
                self.read_only = bool(read_only)
                return
            if not hardware_gate:
                raise PermissionError("D1 CAN connect requires hardware_gate=True")
            if not self.calibration.complete and not read_only:
                raise PermissionError("D1 CAN connect requires complete joint calibration")
            if self.api_factory is None:
                raise RuntimeError("no D1 API adapter supplied; libtita_robot is C++ ABI")
            self.api = self.api_factory()
            self.connected = True
            self._last_vendor_timestamp = None
            self._last_vendor_timestamp_ns = 0
            self.read_only = bool(read_only)

    def disconnect(self) -> None:
        with self._io_lock:
            self.connected = False
            api = self.api
            self.api = None
            self._last_vendor_timestamp = None
            self._last_vendor_timestamp_ns = 0
            self.read_only = False
            # Clear ownership before invoking vendor teardown.  If a vendor
            # close method raises, a subsequent cleanup call still observes a
            # disconnected backend instead of attempting the same destructor
            # twice.
            close = getattr(api, "close", None)
            if callable(close):
                close()

    def emergency_stop(self, *, hardware_gate: bool = False) -> None:
        """Invoke a deployment-provided D1 emergency stop if available.

        ``CanfdApi`` releases have no stable Python-safe E-stop ABI.  A
        validated pybind adapter may expose ``emergency_stop()``; when it
        does, use it.  Otherwise this backend deliberately refuses to invent
        an RPC packet from undocumented struct layout/firmware semantics.
        The runtime still issues a zero command as a best-effort fallback.
        """
        with self._io_lock:
            if self.dry_run:
                return
            if self.read_only:
                raise PermissionError("D1 emergency stop unavailable in read-only inspection mode")
            if not hardware_gate or self.api is None:
                raise PermissionError("D1 emergency stop requires connected backend and hardware_gate=True")
            stop = getattr(self.api, "emergency_stop", None)
            if callable(stop):
                stop()

    def set_force_direct(self, *, hardware_gate: bool = False) -> bool:
        """Explicitly request D1 MCU FORCE_DIRECT mode.

        No caller should infer this transition from ``connect``.  The vendor
        adapter exposes a fixed C ABI for the RPC, and the extra gate makes the
        state-changing operation visible in deployment code and logs.
        """
        with self._io_lock:
            if self.dry_run:
                return True
            if self.read_only:
                raise PermissionError("D1 FORCE_DIRECT refused in read-only inspection mode")
            if not hardware_gate or self.api is None or not self.connected:
                raise PermissionError(
                    "D1 FORCE_DIRECT requires connected backend and hardware_gate=True")
            request = getattr(self.api, "set_force_direct", None)
            if not callable(request):
                raise RuntimeError("D1 API adapter does not expose FORCE_DIRECT RPC")
            return bool(request())

    def read(self) -> StateSnapshot:
        with self._io_lock:
            if not self.connected:
                raise RuntimeError("D1 backend is not connected")
            if self.dry_run:
                self.last_snapshot.monotonic_ns = time.monotonic_ns()
                return self.last_snapshot.copy()
            assert self.api is not None
            try:
                records = _as_records(self.api.get_motors_in())
                if len(records) != D1_COUNT:
                    raise RuntimeError(f"D1 returned {len(records)} motors, expected {D1_COUNT}")
                statuses = _as_records(self.api.get_motors_status())
                if statuses and len(statuses) != D1_COUNT:
                    raise RuntimeError(f"D1 returned {len(statuses)} motor statuses, expected {D1_COUNT}")
                if any(int(status) != 0 for status in statuses):
                    raise RuntimeError("D1 motor status reports a fault")
                is_motors_timeout = getattr(self.api, "is_motors_timeout", None)
                if callable(is_motors_timeout) and bool(is_motors_timeout()):
                    raise RuntimeError("D1 motor feedback timeout")
                q_bus = np.asarray([float(_field(item, "position", 0.0)) for item in records], np.float32)
                dq_bus = np.asarray([float(_field(item, "velocity", 0.0)) for item in records], np.float32)
                tau_bus = np.asarray([float(_field(item, "torque", 0.0)) for item in records], np.float32)
                q, dq, tau = self.calibration.feedback(q_bus, dq_bus, tau_bus, bus="d1")
                imu = self.api.get_imu_data()
                is_imu_timeout = getattr(self.api, "is_imu_timeout", None)
                if callable(is_imu_timeout) and bool(is_imu_timeout()):
                    raise RuntimeError("D1 IMU feedback timeout")
                quat_xyzw = np.asarray(_field(imu, "quaternion", [0, 0, 0, 1]), np.float32)
                if quat_xyzw.shape != (4,):
                    raise RuntimeError("D1 IMU quaternion must have four values")
                snapshot = StateSnapshot(
                    q=np.r_[q, np.zeros(PIPER_COUNT, np.float32)],
                    dq=np.r_[dq, np.zeros(PIPER_COUNT, np.float32)],
                    tau=np.r_[tau, np.zeros(PIPER_COUNT, np.float32)],
                    quat_wxyz=np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], np.float32),
                    gyro=np.asarray(_field(imu, "gyro", [0, 0, 0]), np.float32),
                    accel=np.asarray(_field(imu, "accl", [0, 0, 0]), np.float32),
                    monotonic_ns=time.monotonic_ns(), valid=True,
                )
                if not all(np.isfinite(value).all() for value in (
                    snapshot.q, snapshot.dq, snapshot.tau, snapshot.quat_wxyz,
                    snapshot.gyro, snapshot.accel,
                )):
                    raise RuntimeError("D1 feedback contains NaN or infinity")
                # The flat adapter exposes the vendor IMU timestamp.  If it
                # is available, do not treat repeated cached frames as fresh
                # merely because this Python method was called again.
                timestamp_fn = getattr(self.api, "feedback_timestamp", None)
                vendor_timestamp = (timestamp_fn() if callable(timestamp_fn)
                                    else _field(imu, "timestamp", None))
                if vendor_timestamp is not None and int(vendor_timestamp) != 0:
                    vendor_timestamp = int(vendor_timestamp) & 0xFFFFFFFF
                    now_ns = time.monotonic_ns()
                    if vendor_timestamp != self._last_vendor_timestamp:
                        self._last_vendor_timestamp = vendor_timestamp
                        self._last_vendor_timestamp_ns = now_ns
                    if (self._last_vendor_timestamp_ns <= 0 or
                            now_ns - self._last_vendor_timestamp_ns >
                            int(self.feedback_timeout_s * 1e9)):
                        raise RuntimeError("D1 feedback timestamp is stale")
                    snapshot.monotonic_ns = self._last_vendor_timestamp_ns
                self.last_snapshot = snapshot
            except Exception as exc:
                self.last_snapshot = StateSnapshot(monotonic_ns=time.monotonic_ns(), error=str(exc))
            return self.last_snapshot.copy()

    def send(self, command: JointCommand, *, hardware_gate: bool = False) -> bool:
        with self._io_lock:
            command.validate()
            if self.dry_run:
                return True
            if self.read_only:
                raise PermissionError("D1 send refused: backend is in read-only inspection mode")
            if not hardware_gate:
                raise PermissionError("D1 CAN send requires hardware_gate=True")
            if not self.connected or self.api is None:
                raise RuntimeError("D1 backend is not connected")
            if self.motor_out_factory is None:
                raise RuntimeError(
                    "no motor_out_factory supplied; construct api_motor_out_t in a "
                    "validated pybind/ctypes adapter"
                )

            # The shipped DDT API has appeared in two compatible source-level
            # variants: one sends one packet group per leg, the other accepts
            # all 16 records in a single ``send_motors_can`` call.  Build
            # records in policy/bus order once and dispatch to whichever
            # symbol the validated binding provides.  ``motor_out_factory``
            # must return the vendor api_motor_out_t and is called with named
            # fields documented in hardware/examples/README.md.
            d1_position, d1_velocity, d1_torque = self.calibration.command(
                command.position[:D1_COUNT], command.velocity[:D1_COUNT],
                command.torque[:D1_COUNT], bus="d1")
            # ``command()`` returns position/velocity/torque in bus-index
            # order.  Gains are policy-order arrays, so use the inverse
            # mapping here too; otherwise a non-identity field calibration
            # would silently attach another joint's kp/kd to each motor.
            d1_policy_order, _ = self.calibration._indices("d1")
            records = [self.motor_out_factory(
                # The released ``api_motor_out_t`` uses a uint32 microsecond
                # timestamp (some older docs show uint64).  Masking here
                # keeps the value ABI-safe for both bindings; the vendor
                # packet itself is periodic, so wraparound is harmless.
                # The DDT packet timestamp is wall-clock microseconds (the
                # vendor ``get_current_time()`` helper uses ``gettimeofday``),
                # not a monotonic-clock epoch.  Keep the Python fallback
                # compatible with that wire-level convention.
                timestamp=int((time.time_ns() // 1000) & 0xFFFFFFFF),
                position=float(d1_position[i]),
                velocity=float(d1_velocity[i]),
                kp=float(command.kp[d1_policy_order[i]]),
                kd=float(command.kd[d1_policy_order[i]]),
                torque=float(d1_torque[i]),
            ) for i in range(D1_COUNT)]

            send_legs = getattr(self.api, "send_leg_motors_can", None)
            if callable(send_legs):
                for leg_index in range(4):
                    if not bool(send_legs(records[leg_index * 4:(leg_index + 1) * 4], leg_index)):
                        return False
                return True

            send_all = getattr(self.api, "send_motors_can", None)
            if callable(send_all):
                return bool(send_all(records))
            raise RuntimeError(
                "D1 API binding exposes neither send_leg_motors_can nor send_motors_can"
            )


class PiperBackend:
    """Piper-L ``can1`` backend using the pyAgxArm factory and MIT API."""

    def __init__(self, *, interface: str = "can1", robot: str = "piper_l",
                 bitrate: int = 1_000_000, firmware_profile: str = "auto",
                 dry_run: bool = True,
                 calibration: JointCalibration | None = None,
                 config: PolicyConfig | None = None,
                 feedback_timeout_s: float = 0.020) -> None:
        if interface != "can1":
            raise ValueError("Piper backend is intentionally fixed to can1")
        if robot != "piper_l":
            raise ValueError("this backend only supports piper_l")
        self.interface, self.robot, self.bitrate = interface, robot, int(bitrate)
        self.firmware_profile = firmware_profile
        self.dry_run = bool(dry_run)
        self.config = config or PolicyConfig()
        if not np.isfinite(feedback_timeout_s) or feedback_timeout_s <= 0.0:
            raise ValueError("feedback_timeout_s must be finite and positive")
        self.feedback_timeout_s = float(feedback_timeout_s)
        self.calibration = calibration or JointCalibration.identity()
        self.arm: Any = None
        # pyAgxArm owns one python-can bus object.  The runtime uses a reader
        # thread and a command loop, therefore all driver calls for this arm
        # must be serialized.  A reentrant lock also makes future lifecycle
        # helpers safe without imposing a lock ordering on callers.
        self._io_lock = threading.RLock()
        self.connected = False
        self.enabled = False
        self._motion_mode_set = False
        self._auto_motion_mode_disabled = False
        self.read_only = False
        self.last_snapshot = StateSnapshot()
        # pyAgxArm returns cached MessageAbstract objects.  Their ``hz`` field
        # describes the stream historically, not whether this particular
        # call observed a new CAN frame.  Keep a signature of source
        # timestamps and a local age so a disconnected/stalled can1 cannot be
        # mistaken for fresh feedback merely because the SDK cache is polled
        # frequently.
        self._feedback_signature: tuple[Any, ...] | None = None
        self._feedback_signature_ns = 0

    def _make_arm(self, profile: str) -> Any:
        sdk = importlib.import_module("pyAgxArm")
        cfg = sdk.create_agx_arm_config(
            robot=self.robot, comm="can", firmeware_version=profile,
            interface="socketcan", channel=self.interface, bitrate=self.bitrate,
            auto_connect=False,
        )
        return sdk.AgxArmFactory.create_arm(cfg)

    def connect(self, *, hardware_gate: bool = False, read_only: bool = False) -> None:
        if self.dry_run:
            self.connected = True
            self.last_snapshot.q[16:] = np.asarray(self.config.default_q[16:], np.float32)
            self.last_snapshot.dq.fill(0.0)
            self.last_snapshot.tau.fill(0.0)
            self.last_snapshot.quat_wxyz[:] = (1.0, 0.0, 0.0, 0.0)
            self.last_snapshot.gyro.fill(0.0)
            self.last_snapshot.accel[:] = (0.0, 0.0, 9.81)
            self.last_snapshot.valid = True
            self.last_snapshot.monotonic_ns = time.monotonic_ns()
            self.read_only = bool(read_only)
            self._feedback_signature = None
            self._feedback_signature_ns = time.monotonic_ns()
            return
        if not hardware_gate:
            raise PermissionError("Piper CAN connect requires hardware_gate=True")
        if not self.calibration.complete and not read_only:
            raise PermissionError("Piper CAN connect requires complete joint calibration")
        with self._io_lock:
            sdk = importlib.import_module("pyAgxArm")
            profile = self.firmware_profile
            firmware: dict[str, Any] | None = None
            candidate: Any = None
            try:
                if profile == "auto":
                    # Firmware probing uses a temporary default driver.  This
                    # is the first operation that opens can1 and therefore
                    # remains behind the explicit hardware gate.
                    probe = self._make_arm("default")
                    try:
                        probe.connect()
                        firmware = probe.get_firmware(timeout=1.0)
                    finally:
                        probe.disconnect()
                    if not firmware or not firmware.get("software_version"):
                        raise RuntimeError("unable to read Piper firmware on can1")
                # The returned firmware profile is kept in the backend and is
                # reported by the inspection tool; no enable or motion frame
                # is emitted during this probe.
                if profile == "auto":
                    profile = sdk.resolve_firmware_profile(
                        self.robot, firmware["software_version"]
                    )
                if profile not in {"default", "v183", "v188", "v189"}:
                    raise ValueError(f"unsupported Piper firmware profile: {profile}")
                self.firmware_profile = profile
                candidate = self._make_arm(profile)
                candidate.connect()
                if not bool(candidate.is_connected()):
                    raise RuntimeError("Piper driver did not remain connected")
                self.arm = candidate
                self.connected = True
                self.enabled = False
                self._motion_mode_set = False
                # ``move_mit`` calls the SDK's ``_maybe_set_motion_mode`` on
                # every joint command when automatic mode switching is
                # enabled (the SDK default).  We set MIT explicitly once in
                # ``send`` and disable that per-call behavior so a six-axis
                # command cycle does not emit six redundant mode frames.
                disable_auto_mode = getattr(candidate, "set_auto_set_motion_mode_enabled", None)
                if callable(disable_auto_mode):
                    disable_auto_mode(False)
                    self._auto_motion_mode_disabled = True
                else:
                    self._auto_motion_mode_disabled = False
                self.read_only = bool(read_only)
                self._feedback_signature = None
                self._feedback_signature_ns = 0
            except Exception:
                # A failed connect may have already created a CAN bus and
                # reader thread.  Always tear that candidate down before
                # propagating the error, otherwise a failed firmware/profile
                # selection can leak can1 into the next launch attempt.
                if candidate is not None:
                    try:
                        candidate.disconnect()
                    except Exception:
                        pass
                self.arm = None
                self.connected = False
                self.enabled = False
                self._motion_mode_set = False
                self._auto_motion_mode_disabled = False
                self.read_only = False
                self._feedback_signature = None
                self._feedback_signature_ns = 0
                raise

    def disconnect(self) -> None:
        with self._io_lock:
            arm = self.arm
            self.arm = None
            self.connected = False
            self.enabled = False
            self._motion_mode_set = False
            self._auto_motion_mode_disabled = False
            self.read_only = False
            self._feedback_signature = None
            self._feedback_signature_ns = 0
            if arm is not None:
                disconnect = getattr(arm, "disconnect", None)
                if callable(disconnect):
                    disconnect()

    @staticmethod
    def _message_value(message: Any, name: str, default: float = 0.0) -> float:
        return float(_field(_field(message, "msg", message), name, default))

    def _call_locked(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call one SDK method while serializing access to its CAN bus."""
        with self._io_lock:
            if self.arm is None:
                raise RuntimeError("Piper driver is disconnected")
            return getattr(self.arm, method)(*args, **kwargs)

    def read(self) -> StateSnapshot:
        """Read a coherent Piper snapshot without racing MIT transmissions."""
        with self._io_lock:
            return self._read_locked()

    def _read_locked(self) -> StateSnapshot:
        if not self.connected:
            raise RuntimeError("Piper backend is not connected")
        if self.dry_run:
            self.last_snapshot.monotonic_ns = time.monotonic_ns()
            return self.last_snapshot.copy()
        assert self.arm is not None
        q = np.zeros(PIPER_COUNT, np.float32)
        dq = np.zeros(PIPER_COUNT, np.float32)
        tau = np.zeros(PIPER_COUNT, np.float32)
        try:
            # Position is taken from the grouped joint-angle feedback.  The
            # high-speed motor frame is authoritative for velocity/torque but
            # may expose a motor-axis position representation on some
            # firmware; using both without an explicit conversion would shift
            # the policy zero.  Fall back to high-speed position only for old
            # adapters that do not implement get_joint_angles().
            joint_angles = None
            get_joint_angles = getattr(self.arm, "get_joint_angles", None)
            if callable(get_joint_angles):
                joint_angles = self.arm.get_joint_angles()
            if joint_angles is not None:
                hz = _field(joint_angles, "hz", None)
                if hz is not None and float(hz) <= 0.0:
                    raise RuntimeError("Piper joint-angle feedback rate is zero")
                timestamp = _field(joint_angles, "timestamp", None)
                if timestamp is None or float(timestamp) <= 0.0:
                    raise RuntimeError("Piper joint-angle feedback timestamp is unavailable")
                # pyAgxArm returns MessageAbstract[ArmMsgFeedbackJointStates];
                # ``msg`` is a structured object with joint_1..joint_6 fields,
                # not an iterable list.  Accept both the structured SDK
                # object and list/array values exposed by lightweight bindings.
                values = _field(joint_angles, "msg", joint_angles)
                if all(hasattr(values, f"joint_{index}") for index in range(1, PIPER_COUNT + 1)):
                    values = [getattr(values, f"joint_{index}")
                              for index in range(1, PIPER_COUNT + 1)]
                elif isinstance(values, dict) and all(f"joint_{index}" in values
                                                       for index in range(1, PIPER_COUNT + 1)):
                    values = [values[f"joint_{index}"]
                              for index in range(1, PIPER_COUNT + 1)]
                values = np.asarray(values, dtype=np.float32).reshape(-1)
                if values.size != PIPER_COUNT or not np.isfinite(values).all():
                    raise RuntimeError("invalid Piper joint-angle feedback")
                q[:] = values
            motor_timestamps: list[float] = []
            for index in range(1, PIPER_COUNT + 1):
                state = self.arm.get_motor_states(index)
                if state is None:
                    raise RuntimeError(f"missing Piper motor feedback joint{index}")
                if joint_angles is None:
                    q[index - 1] = self._message_value(state, "position")
                dq[index - 1] = self._message_value(state, "velocity")
                tau[index - 1] = self._message_value(state, "torque")
                # Every high-speed feedback frame has a freshness rate in the
                # SDK.  A missing/zero rate means the reader has not received
                # a complete sample yet; accepting it would turn stale
                # zeros into a valid policy state.
                hz = _field(state, "hz", None)
                if hz is None or float(hz) <= 0.0:
                    raise RuntimeError(f"Piper joint{index} feedback rate is zero")
                timestamp = _field(state, "timestamp", None)
                if timestamp is not None:
                    try:
                        timestamp = float(timestamp)
                    except (TypeError, ValueError):
                        raise RuntimeError(f"Piper joint{index} feedback timestamp is invalid")
                    if not np.isfinite(timestamp) or timestamp <= 0.0:
                        raise RuntimeError(f"Piper joint{index} feedback timestamp is invalid")
                    motor_timestamps.append(timestamp)
            is_ok = getattr(self.arm, "is_ok", None)
            if callable(is_ok) and not bool(self.arm.is_ok()):
                raise RuntimeError("Piper driver reports communication error")
            get_status = getattr(self.arm, "get_arm_status", None)
            if callable(get_status):
                arm_status = self.arm.get_arm_status()
                if arm_status is None:
                    raise RuntimeError("Piper arm status feedback is unavailable")
                status_hz = _field(arm_status, "hz", None)
                if status_hz is not None and float(status_hz) <= 0.0:
                    raise RuntimeError("Piper arm status feedback rate is zero")
                status_msg = _field(arm_status, "msg", arm_status)
                arm_code = _field(status_msg, "arm_status", None)
                if arm_code is not None:
                    # The SDK uses IntEnumBase, so compare by integer value
                    # rather than importing a particular firmware enum class.
                    try:
                        if int(arm_code) != 0:  # ArmStatus.NORMAL
                            raise RuntimeError(f"Piper arm status fault: {arm_code}")
                    except (TypeError, ValueError):
                        raise RuntimeError(f"Piper arm status is not numeric: {arm_code!r}")
                err_status = _field(status_msg, "err_status", None)
                if err_status is not None:
                    # The SDK exposes both communication and joint-angle
                    # limit flags on ``err_status``.  Treat either class of
                    # asserted flag as invalid feedback.  The angle-limit
                    # flags are especially important here: the arm can still
                    # report finite positions while its controller has
                    # already latched a limit fault.
                    for index in range(1, PIPER_COUNT + 1):
                        checks = (
                            (f"communication_status_joint_{index}", "communication"),
                            (f"joint_{index}_angle_limit", "angle-limit"),
                        )
                        for field_name, label in checks:
                            flag = _field(err_status, field_name, False)
                            if bool(flag):
                                raise RuntimeError(
                                    f"Piper joint{index} {label} fault")
                # ``err_code`` is the original 16-bit bitfield.  Checking it
                # as well as the decoded attributes protects against a newer
                # SDK exposing an additional/reserved fault bit that this
                # adapter does not yet know by name.
                err_code = _field(status_msg, "err_code", None)
                if err_code is not None:
                    try:
                        if int(err_code) != 0:
                            raise RuntimeError(f"Piper arm error code: {int(err_code)}")
                    except (TypeError, ValueError):
                        raise RuntimeError(f"Piper arm error code is not numeric: {err_code!r}")
                ctrl_mode = _field(status_msg, "ctrl_mode", None)
                if (ctrl_mode is not None and self.enabled and self._motion_mode_set
                        and int(ctrl_mode) != 1):
                    raise RuntimeError(f"Piper control mode is not CAN control: {ctrl_mode}")
                mode_feedback = _field(status_msg, "mode_feedback", None)
                if mode_feedback is not None:
                    # Firmware <v188 encodes MIT as 0x04; v188/v189 use 0x06.
                    expected_mit_mode = 0x06 if self.firmware_profile in {"v188", "v189"} else 0x04
                    if (int(mode_feedback) != expected_mit_mode
                            and self.enabled and self._motion_mode_set):
                        raise RuntimeError(
                            f"Piper mode feedback {mode_feedback} does not match "
                            f"{self.firmware_profile} MIT code {expected_mit_mode}")
            # A real pyAgxArm sample carries timestamps on the grouped joint
            # message and on each high-speed motor message.  If those fields
            # are available, reject a repeated cache after the configured
            # timeout.  Lightweight test/dummy bindings may omit timestamps;
            # in that case the existing positive-FPS checks remain the only
            # available freshness signal.
            joint_timestamp = None
            if joint_angles is not None:
                value = _field(joint_angles, "timestamp", None)
                if value is not None:
                    try:
                        joint_timestamp = float(value)
                    except (TypeError, ValueError):
                        raise RuntimeError("Piper joint-angle feedback timestamp is invalid")
                    if not np.isfinite(joint_timestamp) or joint_timestamp <= 0.0:
                        raise RuntimeError("Piper joint-angle feedback timestamp is invalid")
            if joint_timestamp is not None or len(motor_timestamps) == PIPER_COUNT:
                signature = (joint_timestamp, *motor_timestamps)
                now_ns = time.monotonic_ns()
                if signature != self._feedback_signature:
                    self._feedback_signature = signature
                    self._feedback_signature_ns = now_ns
                elif (self._feedback_signature_ns <= 0 or
                      now_ns - self._feedback_signature_ns >
                      int(self.feedback_timeout_s * 1e9)):
                    raise RuntimeError("Piper feedback timestamp is stale")
            q, dq, tau = self.calibration.feedback(q, dq, tau, bus="piper")
            self.last_snapshot.q[16:] = q
            self.last_snapshot.dq[16:] = dq
            self.last_snapshot.tau[16:] = tau
            self.last_snapshot.monotonic_ns = time.monotonic_ns()
            self.last_snapshot.valid = bool(all(np.isfinite(value).all() for value in (q, dq, tau)))
            self.last_snapshot.error = "" if self.last_snapshot.valid else "non-finite Piper feedback"
        except Exception as exc:
            self.last_snapshot.valid = False
            self.last_snapshot.error = str(exc)
            self.last_snapshot.monotonic_ns = time.monotonic_ns()
        return self.last_snapshot.copy()

    def firmware_info(self) -> dict[str, Any] | None:
        """Return the SDK firmware record without enabling or commanding joints."""
        with self._io_lock:
            if not self.connected or self.arm is None:
                raise RuntimeError("Piper driver is disconnected")
            get_firmware = getattr(self.arm, "get_firmware", None)
            if not callable(get_firmware):
                raise RuntimeError("Piper SDK does not expose get_firmware")
            result = get_firmware(timeout=1.0)
            return dict(result) if result is not None else None

    def enable(self, *, hardware_gate: bool = False) -> None:
        if self.dry_run:
            return
        if self.read_only:
            raise PermissionError("Piper enable refused: backend is in read-only inspection mode")
        if not hardware_gate or self.arm is None:
            raise PermissionError("Piper enable requires a connected backend and hardware_gate=True")
        with self._io_lock:
            if not bool(self._call_locked("enable")):
                raise RuntimeError("Piper enable was not acknowledged")
            self.enabled = True

    def disable(self, *, hardware_gate: bool = False) -> None:
        if self.dry_run:
            self.enabled = False
            return
        if self.read_only:
            raise PermissionError("Piper disable refused: backend is in read-only inspection mode")
        if not hardware_gate or self.arm is None:
            raise PermissionError("Piper disable requires hardware_gate=True")
        with self._io_lock:
            self._call_locked("disable")
            self.enabled = False

    def emergency_stop(self, *, hardware_gate: bool = False) -> None:
        if self.dry_run:
            return
        if self.read_only:
            raise PermissionError("Piper emergency stop unavailable in read-only inspection mode")
        if not hardware_gate or self.arm is None:
            raise PermissionError("Piper emergency stop requires hardware_gate=True")
        with self._io_lock:
            self._call_locked("electronic_emergency_stop")
            self.enabled = False

    def send(self, command: JointCommand, *, hardware_gate: bool = False) -> bool:
        command.validate()
        if self.dry_run:
            return True
        if self.read_only:
            raise PermissionError("Piper MIT send refused: backend is in read-only inspection mode")
        if not hardware_gate or self.arm is None or not self.connected:
            raise PermissionError("Piper MIT send requires connected backend and hardware_gate=True")
        if not self.enabled:
            raise RuntimeError("Piper MIT send refused because motors are not explicitly enabled")
        piper_position, piper_velocity, piper_torque = self.calibration.command(
            command.position[16:], command.velocity[16:], command.torque[16:], bus="piper")
        piper_policy_order, _ = self.calibration._indices("piper")

        # The SDK clamps out-of-range MIT fields internally and only prints a
        # warning.  Treating that as success would make the command actually
        # sent to the arm differ from the command checked by
        # ``SafetySupervisor``.  Reject it here instead.  The physical YAML
        # limits are applied in policy order; SDK limits are the final hard
        # envelope and are independent of firmware profile except for the
        # feed-forward encoding range.
        piper_limits = tuple(self.config.position_limits[16:])
        velocity_limits = np.asarray(self.config.velocity_limits[16:], dtype=np.float32)
        kp_values = np.asarray(command.kp, dtype=np.float32)
        kd_values = np.asarray(command.kd, dtype=np.float32)
        if len(piper_limits) != PIPER_COUNT or velocity_limits.shape != (PIPER_COUNT,):
            raise ValueError("Piper policy limits must contain six entries")
        for local_index in range(PIPER_COUNT):
            policy_index = int(piper_policy_order[local_index])
            policy_local = policy_index - D1_COUNT
            # Safety/configuration limits are expressed in the policy/URDF
            # coordinate system.  Apply the calibration only after checking
            # those limits; otherwise a non-zero encoder offset would be
            # incorrectly compared against the policy limits below.
            policy_position = float(command.position[policy_index])
            policy_velocity = float(command.velocity[policy_index])
            position = float(piper_position[local_index])
            velocity = float(piper_velocity[local_index])
            kp = float(kp_values[policy_index])
            kd = float(kd_values[policy_index])
            torque = float(piper_torque[local_index])
            limits = piper_limits[policy_local]
            position_lo, position_hi = (-12.5, 12.5)
            if limits is not None:
                policy_position_lo, policy_position_hi = float(limits[0]), float(limits[1])
            else:
                policy_position_lo, policy_position_hi = (-np.inf, np.inf)
            velocity_hi = min(45.0, float(velocity_limits[policy_local]))
            # The pyAgxArm SDK has three different MIT feed-forward wire
            # envelopes for Piper firmware.  Do not use the DEFAULT
            # ``8*b`` table for v183: the v183 driver documents a uniform
            # +/-8 N*m input range for all six joints.  v188/v189 use the
            # 12-bit +/-16 N*m encoding.  Reject values before calling the
            # SDK, which otherwise only prints a warning and silently clamps
            # the command (making the safety audit differ from the frame
            # actually transmitted).
            if self.firmware_profile in {"v188", "v189"}:
                torque_hi = 16.0
            elif self.firmware_profile == "v183":
                torque_hi = 8.0
            elif self.firmware_profile == "default":
                # Piper-L's DEFAULT driver uses +/- (8 * b_i), where the
                # SDK's piper_l torque-b table is (4, 2.5, 4, 1, 1, 1).
                torque_hi = 8.0 * float((4.0, 2.5, 4.0, 1.0, 1.0, 1.0)[local_index])
            else:
                raise ValueError(
                    f"unsupported Piper firmware profile for MIT limits: {self.firmware_profile!r}")
            if not (policy_position_lo <= policy_position <= policy_position_hi):
                raise ValueError(
                    f"Piper joint{local_index + 1} policy p_des={policy_position} outside "
                    f"[{policy_position_lo}, {policy_position_hi}]")
            # The SDK receives calibrated bus coordinates and has its own
            # broad protocol envelope.  Check that envelope separately from
            # the policy/URDF limits above.
            if not (position_lo <= position <= position_hi):
                raise ValueError(
                    f"Piper joint{local_index + 1} bus p_des={position} outside "
                    f"[{position_lo}, {position_hi}]")
            if abs(policy_velocity) > velocity_hi + 1.0e-6:
                raise ValueError(
                    f"Piper joint{local_index + 1} v_des={policy_velocity} exceeds {velocity_hi}")
            if not (0.0 <= kp <= 500.0):
                raise ValueError(f"Piper joint{local_index + 1} kp={kp} outside [0, 500]")
            if not (-5.0 <= kd <= 5.0):
                raise ValueError(f"Piper joint{local_index + 1} kd={kd} outside [-5, 5]")
            if abs(torque) > min(torque_hi, float(self.config.torque_limit[policy_index])) + 1.0e-6:
                raise ValueError(
                    f"Piper joint{local_index + 1} t_ff={torque} exceeds configured limit")
        with self._io_lock:
            if not self._motion_mode_set:
                self._call_locked("set_motion_mode", "mit")
                self._motion_mode_set = True
            for index in range(PIPER_COUNT):
                policy_index = int(piper_policy_order[index])
                self._call_locked(
                    "move_mit",
                    joint_index=index + 1,
                    p_des=float(piper_position[index]),
                    v_des=float(piper_velocity[index]),
                    kp=float(command.kp[policy_index]),
                    kd=float(command.kd[policy_index]),
                    t_ff=float(piper_torque[index]),
                )
        return True


class SafetySupervisor:
    """Latched safety checks shared by both bus senders."""

    def __init__(self, *, state_timeout_ms: float = 20.0,
                 command_timeout_ms: float = 40.0, max_tilt_deg: float = 35.0,
                 min_base_height_m: float = 0.25, max_torque_rate_nm_s: float = 300.0,
                 require_base_height: bool = False,
                 torque_limits: Sequence[float] | None = None,
                 position_limits: Sequence[Sequence[float] | None] | None = None,
                 velocity_limits: Sequence[float] | None = None) -> None:
        self.state_timeout_ns = int(float(state_timeout_ms) * 1e6)
        self.command_timeout_ns = int(float(command_timeout_ms) * 1e6)
        self.max_tilt_deg = float(max_tilt_deg)
        self.min_base_height_m = float(min_base_height_m)
        self.max_torque_rate = float(max_torque_rate_nm_s)
        # The current D1 CAN feedback has no validated world-referenced
        # height measurement.  Make height protection explicit: when enabled
        # without a supplied height source, fail closed instead of silently
        # skipping the check.
        self.require_base_height = bool(require_base_height)
        self.torque_limits = np.asarray(
            torque_limits if torque_limits is not None else [90, 90, 90, 12] * 4 + [20, 20, 15, 7, 5, 5],
            np.float32,
        )
        if self.torque_limits.shape != (JOINT_COUNT,) or not np.isfinite(self.torque_limits).all():
            raise ValueError("torque_limits must contain 22 finite values")
        self.position_lower = np.full(JOINT_COUNT, -np.inf, dtype=np.float32)
        self.position_upper = np.full(JOINT_COUNT, np.inf, dtype=np.float32)
        if position_limits is not None:
            if len(position_limits) != JOINT_COUNT:
                raise ValueError("position_limits must contain 22 entries")
            for index, limits in enumerate(position_limits):
                if limits is None:
                    continue
                if len(limits) != 2:
                    raise ValueError(f"position_limits[{index}] must be [lower, upper] or null")
                lower, upper = (float(limits[0]), float(limits[1]))
                if not np.isfinite([lower, upper]).all() or lower > upper:
                    raise ValueError(f"invalid position_limits[{index}]={limits}")
                self.position_lower[index] = lower
                self.position_upper[index] = upper
        self.velocity_limits = np.full(JOINT_COUNT, np.inf, dtype=np.float32)
        if velocity_limits is not None:
            self.velocity_limits = np.asarray(velocity_limits, dtype=np.float32)
            # ``+inf`` is an explicit, useful value for an actuator whose
            # vendor feedback has no configured speed envelope (for example
            # a wheel hold position or a newly added policy joint).  It is
            # safe in the comparisons below, whereas NaN and negative values
            # would silently disable the limit check.  Torque and position
            # limits remain finite because they define hard physical bounds.
            if self.velocity_limits.shape != (JOINT_COUNT,):
                raise ValueError("velocity_limits must contain 22 values")
            if np.any(np.isnan(self.velocity_limits)) or np.any(self.velocity_limits < 0):
                raise ValueError("velocity_limits must be non-negative; +inf is allowed")
        self.estop = False
        self.fault = ""
        self._last_tau = np.zeros(JOINT_COUNT, np.float32)
        self._last_command_ns = 0

    def trigger(self, reason: str) -> None:
        self.fault = str(reason)

    def clear(self, *, hardware_gate: bool = False) -> None:
        if not hardware_gate:
            raise PermissionError("clearing a safety latch requires hardware_gate=True")
        self.fault = ""
        self.estop = False

    def check_state(self, snapshot: StateSnapshot, *, base_height_m: float | None = None,
                    enforce_joint_limits: bool = True) -> None:
        now = time.monotonic_ns()
        arrays_valid = (
            np.asarray(snapshot.q).shape == (JOINT_COUNT,)
            and np.asarray(snapshot.dq).shape == (JOINT_COUNT,)
            and np.asarray(snapshot.tau).shape == (JOINT_COUNT,)
            and np.asarray(snapshot.gyro).shape == (3,)
            and np.asarray(snapshot.accel).shape == (3,)
            and np.asarray(snapshot.quat_wxyz).shape == (4,)
            and all(np.isfinite(np.asarray(value)).all()
                    for value in (snapshot.q, snapshot.dq, snapshot.tau,
                                  snapshot.gyro, snapshot.accel, snapshot.quat_wxyz))
        )
        if not arrays_valid:
            self.trigger("state snapshot shape or finite-value check failed")
        elif not snapshot.valid:
            self.trigger("invalid state snapshot: " + snapshot.error)
        elif snapshot.monotonic_ns <= 0 or now - snapshot.monotonic_ns > self.state_timeout_ns:
            self.trigger("state snapshot timeout")
        if self.require_base_height and base_height_m is None:
            self.trigger("base height source unavailable")
        elif base_height_m is not None:
            if not math.isfinite(float(base_height_m)):
                self.trigger("base height is not finite")
            elif base_height_m < self.min_base_height_m:
                self.trigger("base height below safety threshold")
        # Feedback limits are checked as well as command targets.  A broken
        # encoder or sign mapping must not be allowed to keep sending torque
        # merely because the requested target is inside its range.
        if arrays_valid and enforce_joint_limits:
            if np.any(np.asarray(snapshot.q) < self.position_lower) or np.any(
                np.asarray(snapshot.q) > self.position_upper
            ):
                self.trigger("joint position feedback outside configured limits")
            if np.any(np.abs(np.asarray(snapshot.dq)) > self.velocity_limits):
                self.trigger("joint velocity feedback outside configured limits")
        quat = np.asarray(snapshot.quat_wxyz, np.float64)
        norm = np.linalg.norm(quat)
        if arrays_valid and norm <= 1e-8:
            self.trigger("invalid IMU quaternion")
        elif arrays_valid:
            quat /= norm
            qw, qx, qy, qz = quat
            gravity_z = -(1.0 - 2.0 * (qx * qx + qy * qy))
            tilt = math.degrees(math.acos(float(np.clip(-gravity_z, -1.0, 1.0))))
            if tilt > self.max_tilt_deg:
                self.trigger(f"tilt {tilt:.1f} deg exceeds threshold")

    def limit_command(self, command: JointCommand, *, state: StateSnapshot | None = None,
                      now_ns: int | None = None, dt: float = 0.002) -> JointCommand:
        command.validate()
        if self.fault or self.estop:
            return JointCommand()
        now = time.monotonic_ns() if now_ns is None else int(now_ns)
        if self._last_command_ns and now - self._last_command_ns > self.command_timeout_ns:
            self.trigger("command watchdog timeout")
            return JointCommand()
        limited = JointCommand(
            position=command.position.copy(), velocity=command.velocity.copy(),
            kp=command.kp.copy(), kd=command.kd.copy(), torque=command.torque.copy(),
        )
        # Position/velocity limits are applied to targets before they reach
        # either vendor backend.  Wheels may intentionally use +/-inf for
        # position because their position field is only a measured hold value.
        limited.position = np.clip(limited.position, self.position_lower, self.position_upper)
        limited.velocity = np.clip(limited.velocity, -self.velocity_limits, self.velocity_limits)
        limited.torque = np.clip(limited.torque, -self.torque_limits, self.torque_limits)
        if state is not None:
            q = np.asarray(state.q, np.float32).reshape(-1)
            dq = np.asarray(state.dq, np.float32).reshape(-1)
            if q.shape != (JOINT_COUNT,) or dq.shape != (JOINT_COUNT,):
                self.trigger("state shape invalid while limiting effective PD torque")
                return JointCommand()
            if not np.isfinite(q).all() or not np.isfinite(dq).all():
                self.trigger("state non-finite while limiting effective PD torque")
                return JointCommand()

            # The vendor-side actuator computes the same impedance expression
            # for both buses.  Project the requested target back into a bound
            # on the *effective* torque, rather than limiting only t_ff.  The
            # gains are scaled only when the requested impedance would violate
            # the safety/rate envelope; this is preferable to transmitting a
            # target that looks bounded while its internal PD term is not.
            kp = np.asarray(limited.kp, np.float32)
            kd = np.asarray(limited.kd, np.float32)
            max_delta = self.max_torque_rate * max(float(dt), 1e-6)
            rate_lower = np.maximum(-self.torque_limits, self._last_tau - max_delta)
            rate_upper = np.minimum(self.torque_limits, self._last_tau + max_delta)
            # The feed-forward part itself must be inside the same rate window
            # before the PD contribution is considered.
            limited.torque = np.clip(limited.torque, rate_lower, rate_upper)
            pd = kp * (limited.position - q) + kd * (limited.velocity - dq)
            nominal = limited.torque + pd
            target = np.clip(nominal, rate_lower, rate_upper)
            delta = target - limited.torque
            scale = np.ones(JOINT_COUNT, dtype=np.float32)
            nonzero = np.abs(pd) > 1.0e-6
            ratio = np.zeros(JOINT_COUNT, dtype=np.float32)
            ratio[nonzero] = delta[nonzero] / pd[nonzero]
            # If the projection lies on the same ray as the PD effort, scale
            # gains to reach it.  If it lies on the opposite ray, hold only
            # the feed-forward part; this keeps the actuator inside the latch.
            scale[nonzero] = np.where(
                (ratio[nonzero] >= 0.0) & (ratio[nonzero] <= 1.0),
                ratio[nonzero], 0.0)
            limited.kp = kp * scale
            limited.kd = kd * scale
            effective = limited.torque + scale * pd
        else:
            # Callers without feedback (for example configuration-only tests)
            # still receive the historical feed-forward limit behavior.
            max_delta = self.max_torque_rate * max(float(dt), 1e-6)
            effective = limited.torque
        self._last_tau = np.clip(effective, -self.torque_limits, self.torque_limits)
        self._last_command_ns = now
        return limited

    def emergency_stop(self) -> None:
        self.estop = True
        self.fault = "emergency stop"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to inspect the hardware config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config/d1_piper_l.yaml")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="safety default; this option is kept for explicitness")
    parser.add_argument("--check-policy", action="store_true",
                        help="load and verify the configured ONNX policy without opening CAN")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    resolved = load_resolved_config(args.config)
    policy = config.get("policy", {})
    buses = config.get("buses", {})
    safety = config.get("safety", {})
    result = {
        "dry_run": True,
        "can0": buses.get("d1", {}).get("interface"),
        "can1": buses.get("piper", {}).get("interface"),
        "policy_abi": [resolved.frame_dim, resolved.history_length, resolved.action_dim],
        "enable_on_start": safety.get("enable_on_start"),
        "hardware_opened": False,
    }
    if args.check_policy:
        policy_path = resolved.path
        if not policy_path:
            raise SystemExit("policy.path is empty")
        checked = OnnxPolicy(policy_path, expected_sha256=resolved.sha256, config=resolved)
        result["policy_sha256"] = checked.sha256
        result["policy_inputs"] = checked.input_names
        result["policy_output"] = checked.output_name
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
