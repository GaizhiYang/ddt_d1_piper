#!/usr/bin/env python3
"""Run a D1+Piper-L policy in MuJoCo.

This is the reference sim2sim controller for the first integration stage.  It
uses named MuJoCo joints (never array positions) and reproduces the IsaacLab
WBC interface: 22 actions, an 82-D frame and three-frame history.  Both the
two-input ONNX export used by the legacy DDT controller and the one-input
flattened export produced by recent RSL-RL versions are accepted.

The executable deliberately has no CAN dependency.  The same observation and
action adapter is used by the future ROS2 hardware backend, which keeps CAN
transport and policy semantics independently testable.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

try:
    from loco_mani_config import (
        ACTUATOR_NAMES as CONFIG_ACTUATOR_NAMES,
        ACTION_SCALE as CONFIG_ACTION_SCALE,
        DC_SATURATION_EFFORT as CONFIG_DC_SATURATION_EFFORT,
        DC_VELOCITY_LIMIT as CONFIG_DC_VELOCITY_LIMIT,
        DEFAULT_Q as CONFIG_DEFAULT_Q,
        JOINT_NAMES as CONFIG_JOINT_NAMES,
        KD as CONFIG_KD,
        KP as CONFIG_KP,
        OBS_NOISE_RANGES as CONFIG_OBS_NOISE_RANGES,
        OBSERVATION_SCALES as CONFIG_OBSERVATION_SCALES,
        TORQUE_LIMIT as CONFIG_TORQUE_LIMIT,
        WHEEL_INDICES as CONFIG_WHEEL_INDICES,
        PolicyConfig,
        load_resolved_config,
        sanitize_commands,
    )
except ImportError:  # source-tree importlib and installed-script fallback
    _config_path = Path(__file__).with_name("loco_mani_config.py")
    _config_spec = importlib.util.spec_from_file_location("loco_mani_config", _config_path)
    if _config_spec is None or _config_spec.loader is None:
        raise ImportError(f"cannot load {_config_path}")
    _config_module = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = _config_module
    _config_spec.loader.exec_module(_config_module)
    CONFIG_ACTUATOR_NAMES = _config_module.ACTUATOR_NAMES
    CONFIG_ACTION_SCALE = _config_module.ACTION_SCALE
    CONFIG_DC_SATURATION_EFFORT = _config_module.DC_SATURATION_EFFORT
    CONFIG_DC_VELOCITY_LIMIT = _config_module.DC_VELOCITY_LIMIT
    CONFIG_DEFAULT_Q = _config_module.DEFAULT_Q
    CONFIG_JOINT_NAMES = _config_module.JOINT_NAMES
    CONFIG_KD = _config_module.KD
    CONFIG_KP = _config_module.KP
    CONFIG_OBS_NOISE_RANGES = _config_module.OBS_NOISE_RANGES
    CONFIG_OBSERVATION_SCALES = _config_module.OBSERVATION_SCALES
    CONFIG_TORQUE_LIMIT = _config_module.TORQUE_LIMIT
    CONFIG_WHEEL_INDICES = _config_module.WHEEL_INDICES
    PolicyConfig = _config_module.PolicyConfig
    load_resolved_config = _config_module.load_resolved_config
    sanitize_commands = _config_module.sanitize_commands

try:
    from command_file import CommandFileReader
except ImportError:  # source-tree contract tests/importlib and installed path
    _command_file_path = Path(__file__).with_name("command_file.py")
    _spec = importlib.util.spec_from_file_location("loco_mani_command_file", _command_file_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load {_command_file_path}")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    CommandFileReader = _module.CommandFileReader

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised on deployment host
    raise SystemExit("MuJoCo Python bindings are required (pip install mujoco)") from exc


JOINT_NAMES = list(CONFIG_JOINT_NAMES)
ACTUATOR_NAMES = list(CONFIG_ACTUATOR_NAMES)
DEFAULT_Q = CONFIG_DEFAULT_Q.copy()
# IsaacLab action-term order is FL, FR, RL, RR leg terms followed by the arm.
# The MuJoCo XML is generated in exactly this order, so no hidden remapping is
# required between policy outputs and actuators.
ACTION_SCALE = CONFIG_ACTION_SCALE.copy()
# The IsaacLab policy's ``last_action`` term is the action-manager output
# after all action terms are concatenated.  The D1+Piper configuration defines
# terms in the order FL pos/vel, FR pos/vel, RL pos/vel, RR pos/vel, arm pos;
# this is exactly the 22-element policy order above.
KP = CONFIG_KP.copy()
KD = CONFIG_KD.copy()
TORQUE_LIMIT = CONFIG_TORQUE_LIMIT.copy()
# IsaacLab DCMotor uses a linear torque-speed envelope before the configured
# continuous-effort limit.  Piper uses DelayedPDActuator (a fixed effort
# limit), so the curve only applies to the 16 D1 motors.
DC_SATURATION_EFFORT = CONFIG_DC_SATURATION_EFFORT.copy()
DC_VELOCITY_LIMIT = CONFIG_DC_VELOCITY_LIMIT.copy()
WHEEL_INDICES = CONFIG_WHEEL_INDICES.copy()


def _parse_csv(text: str, size: int, name: str) -> np.ndarray:
    values = np.asarray([float(v.strip()) for v in text.split(",")], dtype=np.float32)
    if values.size != size:
        raise ValueError(f"{name} requires {size} comma-separated values, got {values.size}")
    return values


def _load_action_replay(path: str | Path, action_dim: int = 22) -> np.ndarray:
    """Load a fixed policy-action trace for deterministic sim2sim replay.

    Supported formats are ``.npy``/``.npz``, JSON and CSV.  The resulting
    array is always ``[num_policy_steps, action_dim]`` and is interpreted at
    the policy update rate (50 Hz by default), not at the MuJoCo physics rate.
    JSON may be either a list of rows or ``{"actions": rows}``; NPZ prefers
    the ``actions`` key.  A CSV with 23 columns is accepted as
    ``time, action[22]`` and drops the first column, which is convenient for
    the diagnostics CSV generated by this script.
    """
    replay_path = Path(path)
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    suffix = replay_path.suffix.lower()
    raw: Any
    if suffix == ".npy":
        raw = np.load(replay_path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(replay_path, allow_pickle=False)
        try:
            if "actions" in archive.files:
                raw = archive["actions"]
            elif len(archive.files) == 1:
                raw = archive[archive.files[0]]
            else:
                raise ValueError(
                    f"{replay_path} must contain an 'actions' array (keys={archive.files})"
                )
        finally:
            archive.close()
    elif suffix == ".json":
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("actions", payload.get("action"))
        if payload is None:
            raise ValueError(f"{replay_path} JSON must contain an 'actions' field")
        raw = payload
    elif suffix in {".csv", ".txt"}:
        rows: list[list[float]] = []
        header: list[str] | None = None
        with replay_path.open("r", newline="", encoding="utf-8") as stream:
            for line_number, row in enumerate(csv.reader(stream), start=1):
                if not row or all(not cell.strip() for cell in row):
                    continue
                try:
                    rows.append([float(cell.strip()) for cell in row])
                except ValueError:
                    if not rows and line_number == 1:
                        header = [cell.strip() for cell in row]
                        continue
                    raise ValueError(f"non-numeric replay row at {replay_path}:{line_number}")
        if header is not None and rows:
            # The diagnostics CSV emitted by this controller contains state,
            # action, torque, q and dq columns.  When its header is present,
            # select exactly the 22 ``action_<joint>`` columns rather than
            # accepting an ambiguous wide matrix.
            action_columns = [i for i, name in enumerate(header)
                              if name.startswith("action_")]
            if len(action_columns) == action_dim:
                rows = [[row[i] for i in action_columns] for row in rows]
        raw = rows
    else:
        raise ValueError(f"unsupported action replay extension: {replay_path.suffix}")

    actions = np.asarray(raw, dtype=np.float32)
    if actions.ndim == 1:
        if actions.size != action_dim:
            raise ValueError(f"action replay must contain {action_dim} values per row, got {actions.size}")
        actions = actions.reshape(1, action_dim)
    if actions.ndim != 2:
        raise ValueError(f"action replay must be a 2-D array, got shape={actions.shape}")
    if actions.shape[1] == action_dim + 1:
        # The diagnostics CSV starts with a monotonically increasing time
        # column.  Do not silently drop an arbitrary extra feature: require
        # that the first column is finite and non-decreasing.
        timestamps = actions[:, 0]
        if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError("23-column replay is not a valid time,action trace")
        actions = actions[:, 1:]
    if actions.shape[1] != action_dim:
        raise ValueError(
            f"action replay width must be {action_dim} (or {action_dim + 1} with time), got {actions.shape}"
        )
    if actions.shape[0] == 0:
        raise ValueError("action replay is empty")
    if not np.isfinite(actions).all():
        raise ValueError("action replay contains NaN or infinity")
    return actions


def _load_initial_state(path: str | Path) -> dict[str, np.ndarray]:
    """Load an IsaacLab reset-state JSON for deterministic action replay.

    The exporter writes policy-order ``q``/``dq`` and the floating-base pose.
    Loading that state is optional, but is strongly recommended for a trace
    comparison: IsaacLab reset events may randomize joints, root pose or
    dynamics even when the policy seed is fixed.
    """
    state_path = Path(path)
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid initial-state JSON: {state_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("initial-state JSON must contain an object")

    def vector(name: str, width: int, *, required: bool = True) -> np.ndarray | None:
        value = payload.get(name)
        if value is None and not required:
            return None
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != width or not np.isfinite(array).all():
            raise ValueError(f"initial-state field {name!r} must contain {width} finite values")
        return array

    result: dict[str, np.ndarray] = {
        "root_pos": vector("root_pos", 3),
        "root_quat": vector("root_quat", 4),
        "q": vector("q", 22),
        "dq": vector("dq", 22),
    }
    root_lin_vel = vector("root_lin_vel", 3, required=False)
    root_ang_vel = vector("root_ang_vel", 3, required=False)
    if root_lin_vel is not None:
        result["root_lin_vel"] = root_lin_vel
    if root_ang_vel is not None:
        result["root_ang_vel"] = root_ang_vel
    quat = result["root_quat"]
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        raise ValueError("initial-state root_quat must be non-zero")
    result["root_quat"] = quat / norm
    return result


def _load_command_replay(path: str | Path) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Load policy-rate command samples exported by IsaacLab."""
    replay_path = Path(path)
    if not replay_path.is_file():
        raise FileNotFoundError(replay_path)
    rate_hz: float | None = None
    suffix = replay_path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(replay_path, allow_pickle=False)
        try:
            velocity_key = "command_velocity" if "command_velocity" in archive.files else "velocity"
            pose_key = "command_ee_pose" if "command_ee_pose" in archive.files else "ee_pose"
            if velocity_key not in archive.files or pose_key not in archive.files:
                raise ValueError(
                    f"{replay_path} must contain command_velocity and command_ee_pose "
                    f"(keys={archive.files})"
                )
            velocity, pose = archive[velocity_key], archive[pose_key]
            if "rate_hz" in archive.files:
                rate = np.asarray(archive["rate_hz"]).reshape(-1)
                if rate.size:
                    rate_hz = float(rate[0])
        finally:
            archive.close()
    elif suffix == ".json":
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "commands" not in payload:
            velocity = payload.get("command_velocity", payload.get("velocity"))
            pose = payload.get("command_ee_pose", payload.get("ee_pose"))
            if payload.get("rate_hz") is not None:
                rate_hz = float(payload["rate_hz"])
        else:
            rows = payload.get("commands") if isinstance(payload, dict) else payload
            array = np.asarray(rows, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 10:
                raise ValueError("JSON commands must contain rows of 10 values: velocity[3] + ee_pose[7]")
            velocity, pose = array[:, :3], array[:, 3:]
    else:
        raise ValueError(f"unsupported command replay extension: {replay_path.suffix}; use .npz or .json")
    velocity = np.asarray(velocity, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"command velocity replay must have shape [N,3], got {velocity.shape}")
    if pose.ndim != 2 or pose.shape[1] != 7 or pose.shape[0] != velocity.shape[0]:
        raise ValueError(f"command EE replay must have shape [N,7] matching velocity, got {pose.shape}")
    if velocity.shape[0] == 0 or not np.isfinite(velocity).all() or not np.isfinite(pose).all():
        raise ValueError("command replay must be non-empty and finite")
    if rate_hz is not None and (not np.isfinite(rate_hz) or rate_hz <= 0):
        raise ValueError("command replay rate_hz must be finite and positive")
    return velocity, pose, rate_hz


class Policy:
    def __init__(self, path: str, action_clip: float = 100.0) -> None:
        self.path = Path(path) if path else None
        self.kind = "zero"
        self.session: Any = None
        self.model: Any = None
        self.input_names: list[str] = []
        self.input_shape: tuple[int, ...] = ()
        self.action_clip = float(action_clip)
        if not np.isfinite(self.action_clip) or self.action_clip <= 0:
            raise ValueError("action_clip must be a finite positive value")
        if self.path is None:
            print("[loco_mani] no policy_path: running zero-action safety mode")
            return
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if self.path.suffix.lower() == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required for .onnx policies") from exc
            self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
            self.input_names = [item.name for item in self.session.get_inputs()]
            self.input_shape = tuple(int(x) for x in self.session.get_inputs()[0].shape
                                     if isinstance(x, (int, np.integer)))
            self.kind = "onnx"
            print(f"[loco_mani] ONNX inputs={self.input_names}")
        elif self.path.suffix.lower() in {".pt", ".jit", ".pth"}:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("PyTorch is required for TorchScript policies") from exc
            self.model = torch.jit.load(str(self.path), map_location="cpu").eval()
            self.kind = "torch"
        else:
            raise ValueError(f"unsupported policy extension: {self.path.suffix}")

    def __call__(self, frame: np.ndarray, history: np.ndarray) -> np.ndarray:
        if self.kind == "zero":
            return np.zeros(22, dtype=np.float32)
        if self.kind == "onnx":
            if len(self.input_names) == 1:
                expected = self.input_shape[-1] if self.input_shape else history.size
                if expected == frame.size:
                    model_input = frame
                elif expected == history.size:
                    model_input = history
                else:
                    raise RuntimeError(f"policy input expects {expected} values, frame={frame.size}, history={history.size}")
                output = self.session.run(None, {self.input_names[0]: model_input[None, :].astype(np.float32)})
            else:
                feed: dict[str, np.ndarray] = {}
                for i, name in enumerate(self.input_names):
                    feed[name] = (frame if i == 0 else history)[None, :].astype(np.float32)
                output = self.session.run(None, feed)
            action = output[0]
        else:
            import torch as torch_module
            try:
                action = self.model(torch_module.from_numpy(history[None, :]))
            except Exception:
                action = self.model(torch_module.from_numpy(frame[None, :]), torch_module.from_numpy(history[None, :]))
            if isinstance(action, (tuple, list)):
                action = action[0]
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 22:
            raise RuntimeError(f"policy output must have 22 values, got {action.size}")
        # The training action terms are clipped to [-100, 100].  Keep this
        # configurable for a conservative diagnostic run, but do not impose
        # an undocumented +/-20 clamp on the exported policy.
        return np.clip(action, -self.action_clip, self.action_clip)


class D1PiperRollout:
    # 3 + 3 + 22 + 22 + 22 + 3 + 7.  The seven pose command values are
    # [x, y, z, qw, qx, qy, qz].
    FRAME_DIM = 82
    HISTORY_LEN = 3
    # The values come from the training ObservationCfg.  Keeping them in one
    # table makes the exported ABI auditable and avoids accidentally applying
    # noise to ``last_action`` or command terms.
    OBS_NOISE_RANGES = CONFIG_OBS_NOISE_RANGES

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 policy: Policy, command_velocity: np.ndarray, ee_pose: np.ndarray,
                 initial_height: float = 0.45, piper_delay_steps: int = 0,
                 passive_damping_scale: float = 0.0,
                 passive_frictionloss_scale: float = 1.0,
                 dc_motor_curve: bool = True,
                 observation_noise: bool = False,
                 noise_seed: int | None = None,
                 initial_state: dict[str, np.ndarray] | None = None,
                 config: PolicyConfig | None = None) -> None:
        self.model, self.data, self.policy = model, data, policy
        self.config = config or PolicyConfig()
        self.default_q = np.asarray(self.config.default_q, np.float32).copy()
        self.action_scale = np.asarray(self.config.action_scale, np.float32).copy()
        self.kp = np.asarray(self.config.kp, np.float32).copy()
        self.kd = np.asarray(self.config.kd, np.float32).copy()
        self.torque_limit = np.asarray(self.config.torque_limit, np.float32).copy()
        self.wheel_indices = np.asarray(self.config.wheel_indices, np.int64).copy()
        self.dc_saturation_effort = np.asarray(self.config.dc_saturation_effort, np.float32).copy()
        self.dc_velocity_limit = np.asarray(self.config.dc_velocity_limit, np.float32).copy()
        self.observation_scales = np.asarray(self.config.observation_scales, np.float32).copy()
        self.observation_noise_ranges = tuple(self.config.observation_noise_ranges)
        self.history_length = int(self.config.history_length)
        if (self.default_q.shape != (22,) or self.action_scale.shape != (22,)
                or self.kp.shape != (22,) or self.kd.shape != (22,)
                or self.torque_limit.shape != (22,)
                or self.dc_saturation_effort.shape != (22,)
                or self.dc_velocity_limit.shape != (22,)
                or self.observation_scales.shape != (7,)
                or self.history_length <= 0):
            raise ValueError("resolved policy configuration has invalid vector dimensions")
        self.command_velocity = command_velocity.astype(np.float32)
        self.ee_pose = ee_pose.astype(np.float32)
        self.qadr = [self._joint_qadr(name) for name in JOINT_NAMES]
        self.vadr = [self._joint_vadr(name) for name in JOINT_NAMES]
        self.aids = [self._actuator_id(name) for name in ACTUATOR_NAMES]
        self.action = np.zeros(22, dtype=np.float32)
        self.target_action = np.zeros(22, dtype=np.float32)
        self.initial_height = float(initial_height)
        self.piper_delay_steps = max(0, int(piper_delay_steps))
        if self.piper_delay_steps > 4:
            raise ValueError("piper_delay_steps must be in [0, 4] for the trained DelayedPDActuator")
        self.dc_motor_curve = bool(dc_motor_curve)
        self.observation_noise = bool(observation_noise)
        self._noise_rng = np.random.default_rng(noise_seed)
        # IsaacLab's DelayedPDActuator delays the arm command by simulation
        # steps.  Keep a small explicit queue so this behavior can be tested
        # without hiding the delay in the MuJoCo XML.
        # DelayBuffer(max_delay) in IsaacLab stores max_delay+1 samples and
        # fills every slot with the first sample.  A deque with the same
        # semantics avoids an off-by-one difference at the first policy step.
        self._arm_history: deque[np.ndarray] | None = None
        self._arm_applied_action = np.zeros(6, dtype=np.float32)
        # IsaacLab's ObservationManager keeps a CircularBuffer *per term* when
        # the group sets ``history_length``.  Consequently the exported input
        # is term-major (all 3 base-ang-vel samples, then all 3 gravity
        # samples, ...), rather than three complete 82-D frames.  This detail
        # is easy to miss because both layouts have the same 246-D shape.
        self._term_slices = (
            slice(0, 3),      # base_ang_vel (already scaled by 0.2)
            slice(3, 6),      # projected_gravity
            slice(6, 28),     # joint_pos
            slice(28, 50),    # joint_vel
            slice(50, 72),    # last_action
            slice(72, 75),    # velocity command
            slice(75, 82),    # end-effector pose command
        )
        self._term_history: list[np.ndarray | None] = [None] * len(self._term_slices)
        self.history: np.ndarray | None = None
        # Exact unflattened frame used for the most recent policy evaluation.
        # Keeping this separately from ``history`` makes it possible to
        # export a policy-rate trace and compare every ABI element with an
        # IsaacLab recording.
        self.last_frame: np.ndarray | None = None
        self.tau = np.zeros(22, dtype=np.float32)
        # The IsaacLab actuator owns PD damping/friction.  The source DDT
        # MJCF also contains joint damping/frictionloss values, which would
        # otherwise be applied a second time by MuJoCo.  Runtime scaling keeps
        # the generated description useful for visualization while making the
        # actuator split explicit and configurable.
        self._configure_passive_joint_dynamics(passive_damping_scale,
                                                passive_frictionloss_scale)
        self._initial_state = initial_state
        self._set_initial_pose()

    @staticmethod
    def validate_model(model: mujoco.MjModel) -> None:
        """Fail early when a description drifts from the 22-DoF policy ABI."""
        missing_joints = [n for n in JOINT_NAMES
                          if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
        missing_actuators = [n for n in ACTUATOR_NAMES
                             if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) < 0]
        if missing_joints or missing_actuators:
            raise RuntimeError(f"model ABI mismatch; missing joints={missing_joints}, "
                               f"actuators={missing_actuators}")

    def _configure_passive_joint_dynamics(self, damping_scale: float,
                                          frictionloss_scale: float) -> None:
        if not np.isfinite(damping_scale) or damping_scale < 0:
            raise ValueError("passive damping scale must be finite and non-negative")
        if not np.isfinite(frictionloss_scale) or frictionloss_scale < 0:
            raise ValueError("passive frictionloss scale must be finite and non-negative")
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                continue
            # Damping/friction are stored per DoF in MuJoCo (a hinge has one
            # DoF; free joints are not part of JOINT_NAMES here).
            did = int(self.model.jnt_dofadr[jid])
            self.model.dof_damping[did] *= float(damping_scale)
            self.model.dof_frictionloss[did] *= float(frictionloss_scale)

    def reset_history(self) -> None:
        """Reset policy/action histories exactly as an IsaacLab env reset."""
        self._term_history = [None] * len(self._term_slices)
        self.history = None
        self.last_frame = None
        self.action.fill(0.0)
        self.target_action.fill(0.0)
        self._arm_history = None
        self._arm_applied_action.fill(0.0)

    def _joint_qadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint not present in MuJoCo model: {name}")
        return int(self.model.jnt_qposadr[jid])

    def _joint_vadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_dofadr[jid])

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise RuntimeError(f"actuator not present in MuJoCo model: {name}")
        return int(aid)

    def _set_initial_pose(self) -> None:
        # The XML default is retained as a description-level hint, but the
        # rollout must be able to compare the training reset (0.45 m) with a
        # collision-corrected reset (~0.4567 m).
        self.data.qpos[2] = self.initial_height
        # MuJoCo free joints store orientation as (w, x, y, z).  MjData is
        # zero-initialised, which is not a valid rotation quaternion; set the
        # identity explicitly so the constructor path has the same reset
        # semantics as ``reset()`` and IsaacLab's upright reset.
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        for address, value in zip(self.qadr, self.default_q):
            self.data.qpos[address] = value
        self.data.qvel[:] = 0.0
        if self._initial_state is not None:
            state = self._initial_state
            self.data.qpos[0:3] = state["root_pos"]
            self.data.qpos[3:7] = state["root_quat"]
            for address, value in zip(self.qadr, state["q"]):
                self.data.qpos[address] = value
            # Free-joint velocity order in MuJoCo is linear xyz then angular
            # xyz.  IsaacLab exports body-frame angular velocity; use it only
            # when explicitly supplied, otherwise leave reset velocities zero.
            if "root_lin_vel" in state:
                self.data.qvel[0:3] = state["root_lin_vel"]
            if "root_ang_vel" in state:
                self.data.qvel[3:6] = state["root_ang_vel"]
            for address, value in zip(self.vadr, state["dq"]):
                self.data.qvel[address] = value
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        """Reset state and policy buffers to the configured training pose."""
        self.data.qpos[:] = 0.0
        self.data.qpos[3] = 1.0
        self.reset_history()
        self._set_initial_pose()

    def _gyro(self) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "trunk_gyro")
        if sid >= 0:
            adr = self.model.sensor_adr[sid]
            return self.data.sensordata[adr:adr + 3].copy()
        return self.data.qvel[3:6].copy()

    @staticmethod
    def _projected_gravity(quat: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = quat
        rotation = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ], dtype=np.float32)
        return rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def frame(self) -> np.ndarray:
        q = np.asarray([self.data.qpos[address] for address in self.qadr], dtype=np.float32)
        dq = np.asarray([self.data.qvel[address] for address in self.vadr], dtype=np.float32)
        q_rel = q - self.default_q
        q_rel[self.wheel_indices] = 0.0
        # IsaacLab's root_ang_vel_b is body-frame angular velocity.  The
        # MuJoCo gyro sensor is already expressed in the sensor/site frame;
        # the site is fixed to base_link, so no world-to-body conversion is
        # needed here.
        terms = [
            self._gyro(),
            self._projected_gravity(self.data.qpos[3:7]),
            q_rel,
            dq,
            self.action,
            self.command_velocity,
            self.ee_pose,
        ]
        frame = np.concatenate([
            np.asarray(term, dtype=np.float32) * self.observation_scales[index]
            for index, term in enumerate(terms)
        ]).astype(np.float32)
        # IsaacLab applies additive uniform corruption to each observation
        # term before it enters the history buffer.  Keep this disabled by
        # default for deterministic sim2sim calibration, but expose the exact
        # training ranges for a deployment-parity experiment.
        if self.observation_noise:
            noise = np.zeros(self.FRAME_DIM, dtype=np.float32)
            # ``base_ang_vel`` has an observation scale of 0.2 in IsaacLab;
            # its configured +/-0.2 corruption is therefore +/-0.04 in the
            # exported frame (noise is injected before the term scale).
            # base_ang_vel is already scaled by 0.2 in ``frame``.
            for index, term_slice in enumerate(self._term_slices):
                lower, upper = self.observation_noise_ranges[index]
                if lower == upper:
                    continue
                width = term_slice.stop - term_slice.start
                noise[term_slice] = self._noise_rng.uniform(lower, upper, width)
                # Noise is specified before the observation term scale.
                noise[term_slice] *= self.observation_scales[index]
            frame += noise.astype(np.float32)
        if frame.size != self.FRAME_DIM:
            raise RuntimeError(f"observation frame has {frame.size} values, expected {self.FRAME_DIM}")
        return np.clip(frame, -self.config.observation_clip, self.config.observation_clip)

    def policy_step(self, frame: np.ndarray | None = None,
                    replay_action: np.ndarray | None = None) -> None:
        """Evaluate one policy update and append one sample per term.

        ``frame`` is injectable so reset-dump tooling can serialize exactly
        the sample that was used to initialize the history buffer (important
        when observation noise is enabled).
        """
        if frame is None:
            frame = self.frame()
        term_blocks: list[np.ndarray] = []
        for index, term_slice in enumerate(self._term_slices):
            term = frame[term_slice]
            old = self._term_history[index]
            if old is None:
                # CircularBuffer fills all entries with the first sample on
                # first append, which is also the reset behavior in IsaacLab.
                old = np.tile(term, (self.history_length, 1))
            else:
                old = np.concatenate([old[1:], term[None, :]], axis=0)
            self._term_history[index] = old
            term_blocks.append(old.reshape(-1))
        history = np.concatenate(term_blocks).astype(np.float32)
        self.history = history
        self.last_frame = np.asarray(frame, dtype=np.float32).copy()
        if replay_action is None:
            self.target_action = self.policy(frame, history)
        else:
            replay_action = np.asarray(replay_action, dtype=np.float32).reshape(-1)
            if replay_action.size != 22 or not np.isfinite(replay_action).all():
                raise ValueError("replay action must be a finite 22-element vector")
            # A replay trace is already in the policy's raw action space.  Do
            # not run it through ONNX again or apply action scales twice.
            self.target_action = np.clip(replay_action, -self.policy.action_clip,
                                         self.policy.action_clip)
        self.action = self.target_action.copy()
        # The arm command is sampled by advance_actuator_delay() at every
        # physics step.  For zero delay it is available immediately.
        if self.piper_delay_steps == 0:
            self._arm_applied_action[:] = self.target_action[16:22]

    def advance_actuator_delay(self) -> None:
        """Advance the arm command delay by one MuJoCo simulation step."""
        if self.piper_delay_steps <= 0:
            return
        command = self.target_action[16:22].copy()
        if self._arm_history is None:
            # CircularBuffer fills all slots at its first append.  This is
            # important when the initial policy command is non-zero.
            self._arm_history = deque(
                (command.copy() for _ in range(self.piper_delay_steps + 1)),
                maxlen=self.piper_delay_steps + 1,
            )
        else:
            self._arm_history.append(command)
        self._arm_applied_action[:] = self._arm_history[0]

    def update_torque(self) -> None:
        """Evaluate the PD actuator at the current simulation state."""
        q = np.asarray([self.data.qpos[address] for address in self.qadr], dtype=np.float32)
        dq = np.asarray([self.data.qvel[address] for address in self.vadr], dtype=np.float32)
        effective_action = self.target_action.copy()
        effective_action[16:22] = self._arm_applied_action
        target_q = self.default_q + effective_action * self.action_scale
        target_dq = np.zeros(22, dtype=np.float32)
        target_dq[self.wheel_indices] = effective_action[self.wheel_indices] * self.action_scale[self.wheel_indices]
        computed = self.kp * (target_q - q) + self.kd * (target_dq - dq)
        if self.dc_motor_curve:
            # Match IsaacLab's DCMotor._clip_effort() exactly.  In particular,
            # the velocity is first clipped to the corner speed
            # ``velocity_limit * (1 + effort_limit / saturation_effort)``;
            # then only the *upper* positive bound and *lower* negative bound
            # are clipped to the configured effort limit.  IsaacLab does not
            # force max_effort >= 0 or min_effort <= 0, so at/above no-load
            # speed the envelope can legitimately cross zero.  Preserving
            # this detail is required for bitwise-compatible sim2sim traces.
            max_effort = self.torque_limit.copy()
            min_effort = -self.torque_limit.copy()
            dc_indices = np.arange(16, dtype=np.int64)
            saturation = self.dc_saturation_effort[dc_indices]
            velocity_limit = self.dc_velocity_limit[dc_indices]
            effort_limit = self.torque_limit[dc_indices]
            corner_speed = velocity_limit * (1.0 + effort_limit / saturation)
            speed = np.clip(dq[dc_indices], -corner_speed, corner_speed)
            top = saturation * (1.0 - speed / velocity_limit)
            bottom = saturation * (-1.0 - speed / velocity_limit)
            max_effort[dc_indices] = np.minimum(top, effort_limit)
            min_effort[dc_indices] = np.maximum(bottom, -effort_limit)
            self.tau = np.minimum(np.maximum(computed, min_effort), max_effort)
        else:
            self.tau = np.clip(computed, -self.torque_limit, self.torque_limit)

    def diagnostics(self) -> dict[str, float]:
        """Return cheap rollout diagnostics useful for sim2sim calibration."""
        gravity = self._projected_gravity(np.asarray(self.data.qpos[3:7], dtype=np.float32))
        tilt = np.degrees(np.arccos(np.clip(-float(gravity[2]), -1.0, 1.0)))
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        return {
            "base_z": float(self.data.qpos[2]),
            "tilt_deg": float(tilt),
            "max_abs_action": float(np.max(np.abs(self.action))),
            "max_abs_tau": float(np.max(np.abs(self.tau))),
            "torque_sat_frac": float(np.mean(np.isclose(np.abs(self.tau), self.torque_limit, atol=1e-3))),
            "contacts": float(self.data.ncon),
            "ee_x": float(self.data.xpos[ee_id, 0]) if ee_id >= 0 else float("nan"),
            "ee_y": float(self.data.xpos[ee_id, 1]) if ee_id >= 0 else float("nan"),
            "ee_z": float(self.data.xpos[ee_id, 2]) if ee_id >= 0 else float("nan"),
        }

    def observation_trace(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the latest frame and flattened policy history."""
        if self.last_frame is None or self.history is None:
            raise RuntimeError("policy_step() must be called before requesting an observation trace")
        return self.last_frame.copy(), self.history.copy()

    def policy_state_trace(self) -> dict[str, np.ndarray]:
        """Return simulator state fields at the policy sample instant.

        These fields deliberately use the same conventions as the IsaacLab
        exporter: quaternions are ``wxyz`` and angular velocity is the raw
        body-frame value (the policy frame applies the additional 0.2 scale).
        Keeping the conversion here makes the trace useful for diagnosing a
        dynamics mismatch even when the policy action itself looks sensible.
        """
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
        gravity = self._projected_gravity(np.asarray(self.data.qpos[3:7], dtype=np.float32))
        if ee_id >= 0:
            ee_pos = np.asarray(self.data.xpos[ee_id], dtype=np.float32).copy()
            ee_quat = np.asarray(self.data.xquat[ee_id], dtype=np.float32).copy()
        else:  # pragma: no cover - model validation normally catches this
            ee_pos = np.full(3, np.nan, dtype=np.float32)
            ee_quat = np.full(4, np.nan, dtype=np.float32)
        return {
            "root_pos": np.asarray(self.data.qpos[0:3], dtype=np.float32).copy(),
            "root_quat": np.asarray(self.data.qpos[3:7], dtype=np.float32).copy(),
            "root_ang_vel": np.asarray(self._gyro(), dtype=np.float32).copy(),
            "projected_gravity": gravity.astype(np.float32, copy=True),
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "command_velocity": self.command_velocity.copy(),
            "command_ee_pose": self.ee_pose.copy(),
        }

    def apply(self) -> None:
        self.data.ctrl[:] = 0.0
        for i, aid in enumerate(self.aids):
            self.data.ctrl[aid] = self.tau[i]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    # Prefer the ament-installed description when the script is launched from
    # an installed ROS2 workspace.  Fall back to the source-tree location so
    # the same file remains directly executable before a colcon build.
    default_xml = here.parents[2] / "loco_mani_description/mujoco/scene.xml"
    try:
        from ament_index_python.packages import get_package_share_directory
        installed_xml = Path(get_package_share_directory("loco_mani_description")) / "mujoco" / "scene.xml"
        if installed_xml.is_file():
            default_xml = installed_xml
    except Exception:
        pass
    parser.set_defaults(_default_xml=str(default_xml))
    parser.add_argument("--xml-path", default="")
    parser.add_argument("--config", default="",
                        help="YAML policy/controller config; values are used unless overridden below")
    parser.add_argument("--policy-path", default="", help="ONNX or TorchScript policy; empty enables zero-action mode")
    parser.add_argument("--action-replay", default="",
                        help="fixed raw 22-D action trace (.npy/.npz/.json/.csv); supersedes policy inference")
    parser.add_argument("--command-replay", default="",
                        help="policy-rate command replay (.npz/.json); updates velocity and EE pose observations")
    parser.add_argument("--command-file", default="",
                        help="live JSON command file written atomically by the loco_mani keyboard node")
    parser.add_argument("--command-file-timeout", type=float, default=0.5,
                        help="ignore live command-file updates older than this many seconds")
    parser.add_argument("--initial-state", default="",
                        help="optional IsaacLab reset-state JSON (q/dq/root pose) for deterministic replay")
    parser.add_argument("--replay-rate-hz", type=float, default=50.0,
                        help="policy-rate of --action-replay (default 50 Hz; used for trace diagnostics)")
    parser.add_argument("--action-clip", type=float, default=100.0,
                        help="absolute raw-action clip (training terms use 100)")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="simulation seconds; 0 runs until the viewer is closed/Ctrl-C")
    parser.add_argument("--sim-dt", type=float, default=0.005)
    parser.add_argument("--control-decimation", type=int, default=4)
    parser.add_argument("--initial-height", type=float, default=0.45,
                        help="root z at reset; 0.4567 avoids the MJCF wheel penetration")
    parser.add_argument("--piper-delay-steps", type=int, default=0,
                        help="Piper DelayedPD delay in simulation steps (training max=4)")
    parser.add_argument("--passive-damping-scale", type=float, default=0.0,
                        help="scale MJCF joint damping in addition to explicit PD (default 0)")
    parser.add_argument("--passive-frictionloss-scale", type=float, default=1.0,
                        help="scale MJCF joint frictionloss (default 1; use 0 for ablation)")
    parser.add_argument("--disable-dc-motor-curve", action="store_true",
                        help="disable IsaacLab D1 DCMotor torque-speed clipping")
    parser.add_argument("--diagnostics-path", default="",
                        help="optional CSV path for per-step state/action/torque diagnostics")
    parser.add_argument("--policy-trace-path", default="",
                        help="optional CSV path for policy-rate frame/history/action trace")
    parser.add_argument("--command-velocity", default="0,0,0", help="vx,vy,yaw-rate")
    parser.add_argument("--ee-pose", default="0.425,0,0.5,1,0,0,0", help="x,y,z,qw,qx,qy,qz")
    parser.add_argument("--headless", dest="headless", action="store_true", default=None)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--real-time", dest="real_time", action="store_true", default=None)
    parser.add_argument("--no-real-time", dest="real_time", action="store_false")
    parser.add_argument("--print-interval", type=float, default=1.0,
                        help="Print state statistics every N simulated seconds; 0 disables")
    parser.add_argument("--stop-on-fall", action="store_true",
                        help="Stop when base height/tilt indicates a fall")
    parser.add_argument("--fall-height", type=float, default=0.25,
                        help="base height threshold for --stop-on-fall (m)")
    parser.add_argument("--fall-tilt-deg", type=float, default=35.0,
                        help="tilt threshold for --stop-on-fall (degrees)")
    parser.add_argument("--observation-noise", action="store_true",
                        help="enable IsaacLab training-style additive observation noise")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="seed for --observation-noise (default: entropy source)")
    parser.add_argument("--observation-dump", default="",
                        help="write reset frame/history and ABI metadata as JSON")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    resolved_config = load_resolved_config(args.config) if args.config else PolicyConfig()
    if not args.xml_path:
        args.xml_path = getattr(args, "_default_xml", "")
    if args.config:
        if not args.policy_path:
            args.policy_path = resolved_config.path
        if resolved_config.xml_path and args.xml_path == getattr(args, "_default_xml", ""):
            args.xml_path = resolved_config.xml_path
        if args.command_velocity == "0,0,0":
            args.command_velocity = ",".join(str(float(v)) for v in resolved_config.command_velocity)
        if args.ee_pose == "0.425,0,0.5,1,0,0,0":
            args.ee_pose = ",".join(str(float(v)) for v in resolved_config.ee_pose)
        if args.duration == 30.0:
            args.duration = resolved_config.duration
        if args.sim_dt == 0.005:
            args.sim_dt = resolved_config.sim_dt
        if args.control_decimation == 4:
            args.control_decimation = resolved_config.decimation
        if args.initial_height == 0.45:
            args.initial_height = resolved_config.initial_height
        if args.piper_delay_steps == 0:
            args.piper_delay_steps = resolved_config.piper_delay_steps
        if args.passive_damping_scale == 0.0:
            args.passive_damping_scale = resolved_config.passive_damping_scale
        if args.passive_frictionloss_scale == 1.0:
            args.passive_frictionloss_scale = resolved_config.passive_frictionloss_scale
        if not args.disable_dc_motor_curve and not resolved_config.dc_motor_curve:
            args.disable_dc_motor_curve = True
        if not args.observation_noise and resolved_config.observation_noise:
            args.observation_noise = True
        if args.headless is None:
            args.headless = resolved_config.headless
        if args.real_time is None:
            args.real_time = resolved_config.real_time
    else:
        if args.headless is None:
            args.headless = False
        if args.real_time is None:
            args.real_time = False
    velocity = _parse_csv(args.command_velocity, 3, "command_velocity")
    ee_pose = _parse_csv(args.ee_pose, 7, "ee_pose")
    norm = np.linalg.norm(ee_pose[3:])
    if norm < 1e-6:
        raise ValueError("ee_pose quaternion must be non-zero")
    ee_pose[3:] /= norm
    velocity, ee_pose = sanitize_commands(velocity, ee_pose, resolved_config)
    if not np.isfinite(args.sim_dt) or args.sim_dt <= 0:
        raise ValueError("sim_dt must be a finite positive value")
    if not np.isfinite(args.duration) or args.duration < 0:
        raise ValueError("duration must be finite and non-negative (0 means unlimited)")
    if args.control_decimation <= 0:
        raise ValueError("control_decimation must be a positive integer")
    if not np.isfinite(args.fall_height) or args.fall_height <= 0:
        raise ValueError("fall_height must be a finite positive value")
    if not np.isfinite(args.fall_tilt_deg) or args.fall_tilt_deg <= 0:
        raise ValueError("fall_tilt_deg must be a finite positive value")
    if not np.isfinite(args.replay_rate_hz) or args.replay_rate_hz <= 0:
        raise ValueError("replay_rate_hz must be a finite positive value")
    replay_actions = _load_action_replay(args.action_replay) if args.action_replay else None
    replay_commands = _load_command_replay(args.command_replay) if args.command_replay else None
    if replay_commands is not None and replay_commands[2] is not None:
        args.replay_rate_hz = replay_commands[2]
    initial_state = _load_initial_state(args.initial_state) if args.initial_state else None
    command_file = CommandFileReader(args.command_file, args.command_file_timeout) if args.command_file else None
    if command_file is not None and replay_commands is None:
        live_command = command_file.read()
        if live_command is not None:
            velocity, ee_pose = sanitize_commands(*live_command, resolved_config)

    model = mujoco.MjModel.from_xml_path(args.xml_path)
    model.opt.timestep = args.sim_dt
    D1PiperRollout.validate_model(model)
    data = mujoco.MjData(model)
    # Fixed replay is deliberately independent of the inference runtime.  In
    # that mode do not even instantiate ``Policy``: this lets a workstation
    # without onnxruntime replay an IsaacLab trace and avoids loading/running
    # the network twice.
    policy_path = "" if replay_actions is not None else args.policy_path
    rollout = D1PiperRollout(
        model, data, Policy(policy_path, action_clip=resolved_config.action_clip if args.config else args.action_clip), velocity, ee_pose,
        initial_height=args.initial_height,
        piper_delay_steps=args.piper_delay_steps,
        passive_damping_scale=args.passive_damping_scale,
        passive_frictionloss_scale=args.passive_frictionloss_scale,
        dc_motor_curve=not args.disable_dc_motor_curve,
        observation_noise=args.observation_noise,
        noise_seed=args.noise_seed,
        initial_state=initial_state,
        config=resolved_config,
    )
    decimation = max(1, args.control_decimation)
    # ``duration=0`` is an explicit interactive mode.  It is particularly
    # useful with a live keyboard command file because the simulation should
    # not stop after the 30 s diagnostic default.
    steps = None if args.duration == 0.0 else int(args.duration / args.sim_dt)
    next_report = 0.0

    if args.observation_dump:
        # Capture the exact reset observation before the first action is
        # evaluated.  This is the artifact used for element-wise comparison
        # against IsaacLab's first reset observation.
        if replay_commands is not None:
            rollout.command_velocity = replay_commands[0][0].copy()
            rollout.ee_pose = replay_commands[1][0].copy()
        reset_frame_array = rollout.frame()
        if replay_actions is None:
            rollout.policy_step(reset_frame_array)
        else:
            rollout.policy_step(reset_frame_array, replay_action=replay_actions[0])
        reset_frame = reset_frame_array.tolist()
        reset_history = rollout.history.tolist() if rollout.history is not None else []
        dump = {
            "policy_path": str(args.policy_path),
            "xml_path": str(args.xml_path),
            "frame_dim": rollout.FRAME_DIM,
            "history_length": rollout.history_length,
            "history_layout": "term-major",
            "joint_names": JOINT_NAMES,
            "frame": reset_frame,
            "history": reset_history,
            "action": rollout.action.tolist(),
            "action_source": "replay" if replay_actions is not None else "policy",
            "replay_rate_hz": args.replay_rate_hz if replay_actions is not None else None,
            "initial_height": args.initial_height,
            "observation_noise": args.observation_noise,
        }
        dump_path = Path(args.observation_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
        # policy_step above is the first policy update at t=0; the loop must
        # not evaluate it twice.
        first_policy_done = True
    else:
        first_policy_done = False

    diagnostics_file = None
    diagnostics_writer = None
    if args.diagnostics_path:
        diagnostics_file = open(args.diagnostics_path, "w", newline="", encoding="utf-8")
        diagnostics_writer = csv.writer(diagnostics_file)
        diagnostics_writer.writerow([
            "time", "base_x", "base_y", "base_z", "tilt_deg", "contacts",
            *[f"action_{n}" for n in JOINT_NAMES],
            *[f"tau_{n}" for n in JOINT_NAMES],
            *[f"q_{n}" for n in JOINT_NAMES],
            *[f"dq_{n}" for n in JOINT_NAMES],
        ])

    # A separate policy-rate trace is intentionally wider than the regular
    # physics diagnostics.  It captures exactly what was fed to ONNX (the
    # 82-D frame and 246-D term-major history), together with the resulting
    # action and pre-integration state/torque.  This is the primary artifact
    # for element-wise IsaacLab -> MuJoCo comparison.
    policy_trace_file = None
    policy_trace_writer = None
    if args.policy_trace_path:
        policy_trace_path = Path(args.policy_trace_path)
        policy_trace_path.parent.mkdir(parents=True, exist_ok=True)
        policy_trace_file = policy_trace_path.open("w", newline="", encoding="utf-8")
        policy_trace_writer = csv.writer(policy_trace_file)
        policy_trace_writer.writerow([
            "time", "physics_step",
            "root_x", "root_y", "root_z",
            "root_qw", "root_qx", "root_qy", "root_qz",
            "root_wx", "root_wy", "root_wz",
            "gravity_x", "gravity_y", "gravity_z",
            "ee_x", "ee_y", "ee_z",
            "ee_qw", "ee_qx", "ee_qy", "ee_qz",
            "cmd_vx", "cmd_vy", "cmd_wz",
            "cmd_ee_x", "cmd_ee_y", "cmd_ee_z",
            "cmd_ee_qw", "cmd_ee_qx", "cmd_ee_qy", "cmd_ee_qz",
            *[f"frame_{i}" for i in range(rollout.FRAME_DIM)],
            *[f"history_{i}" for i in range(rollout.FRAME_DIM * rollout.history_length)],
            *[f"action_{n}" for n in JOINT_NAMES],
            *[f"tau_{n}" for n in JOINT_NAMES],
            *[f"q_{n}" for n in JOINT_NAMES],
            *[f"dq_{n}" for n in JOINT_NAMES],
        ])

    def loop(viewer: Any = None) -> None:
        nonlocal next_report
        step = 0
        while steps is None or step < steps:
            started = time.monotonic()
            policy_updated = False
            if step % decimation == 0 and not (step == 0 and first_policy_done):
                if replay_commands is not None:
                    command_index = int(np.floor(
                        step * args.sim_dt * args.replay_rate_hz + 1.0e-9
                    ))
                    command_index = min(command_index, replay_commands[0].shape[0] - 1)
                    rollout.command_velocity, rollout.ee_pose = sanitize_commands(
                        replay_commands[0][command_index], replay_commands[1][command_index],
                        resolved_config)
                elif command_file is not None:
                    live_command = command_file.read()
                    if live_command is not None:
                        rollout.command_velocity, rollout.ee_pose = sanitize_commands(
                            *live_command, resolved_config)
                if replay_actions is None:
                    rollout.policy_step()
                else:
                    # ``step`` is a physics-step index.  Convert it through
                    # the trace rate instead of assuming every replay file is
                    # exactly one row per controller update.
                    replay_index = int(np.floor(
                        step * args.sim_dt * args.replay_rate_hz + 1.0e-9
                    ))
                    if replay_index >= replay_actions.shape[0]:
                        # Holding the final command makes a shorter trace
                        # useful for checking a settling response while still
                        # making the truncation explicit in the console.
                        replay_index = replay_actions.shape[0] - 1
                    rollout.policy_step(replay_action=replay_actions[replay_index])
                policy_updated = True
            rollout.advance_actuator_delay()
            # PD is evaluated every simulation step, matching IsaacLab's
            # actuator update instead of holding a stale torque for decimation.
            rollout.update_torque()
            rollout.apply()
            if policy_trace_writer is not None and (policy_updated or (step == 0 and first_policy_done)):
                if rollout.last_frame is None or rollout.history is None:
                    raise RuntimeError("policy trace requested but no frame/history is available")
                q_trace = np.asarray([data.qpos[address] for address in rollout.qadr], dtype=np.float32)
                dq_trace = np.asarray([data.qvel[address] for address in rollout.vadr], dtype=np.float32)
                state_trace = rollout.policy_state_trace()
                policy_trace_writer.writerow([
                    float(data.time), int(step),
                    *state_trace["root_pos"].tolist(),
                    *state_trace["root_quat"].tolist(),
                    *state_trace["root_ang_vel"].tolist(),
                    *state_trace["projected_gravity"].tolist(),
                    *state_trace["ee_pos"].tolist(),
                    *state_trace["ee_quat"].tolist(),
                    *state_trace["command_velocity"].tolist(),
                    *state_trace["command_ee_pose"].tolist(),
                    *rollout.last_frame.tolist(), *rollout.history.tolist(),
                    *rollout.action.tolist(), *rollout.tau.tolist(),
                    *q_trace.tolist(), *dq_trace.tolist(),
                ])
            mujoco.mj_step(model, data)
            gravity = rollout._projected_gravity(np.asarray(data.qpos[3:7], dtype=np.float32))
            tilt = np.degrees(np.arccos(np.clip(-float(gravity[2]), -1.0, 1.0)))
            if args.print_interval > 0 and data.time >= next_report:
                diag = rollout.diagnostics()
                q = np.asarray([data.qpos[address] for address in rollout.qadr])
                print(
                    f"[loco_mani] t={data.time:7.3f}s z={diag['base_z']:+.3f} "
                    f"tilt={diag['tilt_deg']:5.1f}deg contacts={int(diag['contacts'])} "
                    f"|q|={np.linalg.norm(q):.3f} max_tau={diag['max_abs_tau']:.2f} "
                    f"sat={diag['torque_sat_frac']:.2f} ee=({diag['ee_x']:.2f},{diag['ee_y']:.2f},{diag['ee_z']:.2f})"
                )
                next_report += args.print_interval
            if diagnostics_writer is not None:
                q = np.asarray([data.qpos[address] for address in rollout.qadr], dtype=np.float32)
                dq = np.asarray([data.qvel[address] for address in rollout.vadr], dtype=np.float32)
                diagnostics_writer.writerow([
                    float(data.time), float(data.qpos[0]), float(data.qpos[1]),
                    float(data.qpos[2]), float(tilt), int(data.ncon),
                    *rollout.action.tolist(), *rollout.tau.tolist(),
                    *q.tolist(), *dq.tolist(),
                ])
            if args.stop_on_fall:
                if data.qpos[2] < args.fall_height or tilt > args.fall_tilt_deg:
                    print(f"[loco_mani] stopping after fall detection at t={data.time:.3f}s")
                    return
            if viewer is not None:
                viewer.sync()
                if not viewer.is_running():
                    return
            if args.real_time:
                remaining = args.sim_dt - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
            step += 1

    if args.headless:
        loop()
    else:
        try:
            from mujoco import viewer as mujoco_viewer
            # ``launch_passive`` owns a daemon UI thread.  Let that thread
            # observe the exit request before Python tears down MuJoCo/GLFW;
            # closing the context and immediately terminating the process can
            # otherwise race in GLFW cleanup (typically reported as
            # ``exit code -11`` or ``malloc(): unsorted ... corrupted``).
            viewer = mujoco_viewer.launch_passive(model, data)
            try:
                loop(viewer)
            finally:
                try:
                    viewer.close()
                except Exception:
                    pass
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    try:
                        if not viewer.is_running():
                            break
                    except Exception:
                        break
                    time.sleep(0.01)
        except RuntimeError as exc:
            raise SystemExit(f"viewer failed; retry with --headless: {exc}") from exc
    if np.isfinite(data.qpos).all():
        gravity = rollout._projected_gravity(np.asarray(data.qpos[3:7], dtype=np.float32))
        tilt = np.degrees(np.arccos(np.clip(-float(gravity[2]), -1.0, 1.0)))
        print(f"[loco_mani] finished t={data.time:.3f}s z={data.qpos[2]:+.3f} tilt={tilt:.1f}deg")
    if diagnostics_file is not None:
        diagnostics_file.close()
    if policy_trace_file is not None:
        policy_trace_file.close()


if __name__ == "__main__":
    main()
