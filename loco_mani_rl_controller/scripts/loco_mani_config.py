#!/usr/bin/env python3
"""Shared YAML configuration and policy-contract defaults.

The sim2sim and hardware adapters intentionally use the same resolved
configuration.  Keeping the policy order, gains, action scales and
observation semantics in one file prevents a newly trained checkpoint from
silently running with stale constants embedded in one of the two runtimes.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]
ACTUATOR_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf", "FL_foot",
    "FR_hip", "FR_thigh", "FR_calf", "FR_foot",
    "RL_hip", "RL_thigh", "RL_calf", "RL_foot",
    "RR_hip", "RR_thigh", "RR_calf", "RR_foot",
    "piper_joint1", "piper_joint2", "piper_joint3", "piper_joint4",
    "piper_joint5", "piper_joint6",
]
JOINT_COUNT = len(JOINT_NAMES)
D1_COUNT = 16
PIPER_COUNT = 6
WHEEL_INDICES = np.asarray([3, 7, 11, 15], dtype=np.int64)

DEFAULT_Q = np.asarray([0.0, 0.8, -1.5, 0.0] * 4 + [0.0] * 6, np.float32)
ACTION_SCALE = np.asarray([0.25, 0.25, 0.25, 5.0] * 4 + [0.25] * 6, np.float32)
KP = np.asarray([60.0, 60.0, 60.0, 0.0] * 4
                 + [50.0, 50.0, 80.0, 30.0, 30.0, 20.0], np.float32)
KD = np.asarray([3.0, 3.0, 3.0, 0.5] * 4
                 + [3.0, 2.0, 3.0, 3.0, 2.5, 1.0], np.float32)
TORQUE_LIMIT = np.asarray([90.0, 90.0, 90.0, 12.0] * 4
                           + [20.0, 20.0, 15.0, 7.0, 5.0, 5.0], np.float32)
DC_SATURATION_EFFORT = np.asarray([90.0, 90.0, 90.0, 12.0] * 4
                                   + [0.0] * 6, np.float32)
DC_VELOCITY_LIMIT = np.asarray([20.0, 20.0, 20.0, 30.0] * 4
                                + [np.inf] * 6, np.float32)

# Frame terms are concatenated in this fixed ABI order.  The sizes are part
# of the exported policy contract, while scales/noise ranges are configurable.
OBSERVATION_TERM_NAMES = (
    "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
    "last_action", "command_velocity", "ee_pose",
)
OBSERVATION_TERM_SIZES = (3, 3, 22, 22, 22, 3, 7)
OBSERVATION_SCALES = np.ones(7, np.float32)
OBSERVATION_SCALES[0] = 0.2
OBS_NOISE_RANGES = (
    (-0.2, 0.2), (-0.05, 0.05), (-0.01, 0.01), (-1.5, 1.5),
    (0.0, 0.0), (0.0, 0.0), (0.0, 0.0),
)


@dataclasses.dataclass
class PolicyConfig:
    """Resolved values consumed by both the MuJoCo and CAN adapters."""

    path: str = ""
    sha256: str | None = None
    input_names: tuple[str, ...] = ("obs",)
    output_name: str = "actions"
    frequency_hz: float = 50.0
    history_length: int = 3
    frame_dim: int = 82
    action_dim: int = 22
    sim_dt: float = 0.005
    decimation: int = 4
    history_layout: str = "term-major"
    action_clip: float = 100.0
    default_q: np.ndarray = dataclasses.field(default_factory=lambda: DEFAULT_Q.copy())
    action_scale: np.ndarray = dataclasses.field(default_factory=lambda: ACTION_SCALE.copy())
    kp: np.ndarray = dataclasses.field(default_factory=lambda: KP.copy())
    kd: np.ndarray = dataclasses.field(default_factory=lambda: KD.copy())
    torque_limit: np.ndarray = dataclasses.field(default_factory=lambda: TORQUE_LIMIT.copy())
    wheel_indices: np.ndarray = dataclasses.field(default_factory=lambda: WHEEL_INDICES.copy())
    dc_saturation_effort: np.ndarray = dataclasses.field(
        default_factory=lambda: DC_SATURATION_EFFORT.copy())
    dc_velocity_limit: np.ndarray = dataclasses.field(
        default_factory=lambda: DC_VELOCITY_LIMIT.copy())
    observation_scales: np.ndarray = dataclasses.field(
        default_factory=lambda: OBSERVATION_SCALES.copy())
    observation_noise_ranges: tuple[tuple[float, float], ...] = OBS_NOISE_RANGES
    observation_clip: float = 100.0
    piper_delay_steps: int = 0
    initial_height: float = 0.45
    command_velocity: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(3, np.float32))
    ee_pose: np.ndarray = dataclasses.field(
        default_factory=lambda: np.asarray([0.425, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], np.float32))
    xml_path: str = ""
    duration: float = 30.0
    passive_damping_scale: float = 0.0
    passive_frictionloss_scale: float = 1.0
    dc_motor_curve: bool = True
    headless: bool = True
    real_time: bool = False
    print_interval: float = 1.0
    stop_on_fall: bool = False
    fall_height: float = 0.25
    fall_tilt_deg: float = 35.0
    feedback_timeout_s: float = 0.020
    command_hz: float = 500.0
    state_reader_hz: float = 500.0
    command_timeout_s: float = 0.040
    command_file_timeout_s: float = 0.500
    observation_noise: bool = False
    joint_names: tuple[str, ...] = tuple(JOINT_NAMES)
    actuator_names: tuple[str, ...] = tuple(ACTUATOR_NAMES)
    position_limits: tuple[tuple[float, float] | None, ...] = tuple([None] * JOINT_COUNT)
    velocity_limits: np.ndarray = dataclasses.field(
        default_factory=lambda: np.full(JOINT_COUNT, np.inf, np.float32))
    command_velocity_limits: np.ndarray = dataclasses.field(
        default_factory=lambda: np.asarray([1.0, 1.0, 1.5], np.float32))
    # Command limits are kept in the same frame/units as the trained policy.
    # EE position is [x, y, z] (x/y relative to piper_base_link, z world
    # height for the current WBC export); orientation limits are XYZ fixed-axis
    # RPY limits in radians.
    ee_position_limits: tuple[tuple[float, float], ...] = (
        (0.15, 0.75), (-0.45, 0.45), (0.20, 0.90))
    ee_orientation_limits_rpy: tuple[tuple[float, float], ...] = (
        (-0.6, 0.6), (-0.6, 0.6), (-0.6, 0.6))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def parse_bool(value: Any, default: bool = False, *, name: str = "value") -> bool:
    """Parse a YAML/CLI boolean without treating ``"false"`` as true.

    Safety-related configuration is often edited by hand or generated by a
    deployment script.  Python's ``bool(str)`` is unsafe for that use because
    every non-empty string, including ``"false"``, evaluates to ``True``.
    Accept the common YAML spellings explicitly and reject ambiguous values.
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean, got {value!r}")
    # YAML may produce integer 0/1.  Do not silently accept arbitrary numbers
    # because a typo such as 2 should fail closed rather than enable a gate.
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _vector(value: Any, default: Sequence[float], size: int, name: str,
            *, finite: bool = True) -> np.ndarray:
    if value is None:
        value = default
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain {size} values, got {array.size}")
    if finite and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _resolve_path(value: Any, base_dir: Path) -> str:
    if value is None or str(value).strip() == "":
        return ""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load loco-mani configuration") from exc
    config_path = Path(path).expanduser()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must contain a YAML mapping: {config_path}")
    return payload


def validate_hardware_config(payload: Mapping[str, Any], *, require_complete: bool = False) -> bool:
    """Validate the deployment-only parts of the YAML without touching CAN.

    ``resolve_config`` validates the neural-network ABI.  This companion
    check validates the two-bus contract before a vendor factory is imported:
    D1 is exactly 16 joints on ``can0`` and Piper-L is exactly 6 joints on
    ``can1``.  When ``require_complete`` is true, every physical calibration
    field must be explicitly filled and unique.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("hardware configuration must be a YAML mapping")
    joints = tuple(str(item) for item in payload.get("joints", JOINT_NAMES))
    if joints != tuple(JOINT_NAMES):
        raise ValueError("joints must match the fixed D1+Piper-L policy order")
    buses = _mapping(payload.get("buses"), "buses")
    d1 = _mapping(buses.get("d1"), "buses.d1")
    piper = _mapping(buses.get("piper"), "buses.piper")
    if str(d1.get("interface", "can0")) != "can0":
        raise ValueError("buses.d1.interface must be can0")
    if int(d1.get("joint_count", D1_COUNT)) != D1_COUNT:
        raise ValueError("buses.d1.joint_count must be 16")
    if str(piper.get("interface", "can1")) != "can1":
        raise ValueError("buses.piper.interface must be can1")
    if int(piper.get("joint_count", PIPER_COUNT)) != PIPER_COUNT:
        raise ValueError("buses.piper.joint_count must be 6")
    if str(piper.get("robot", "piper_l")) != "piper_l":
        raise ValueError("buses.piper.robot must be piper_l")
    if int(piper.get("bitrate", 1_000_000)) <= 0:
        raise ValueError("buses.piper.bitrate must be positive")
    calibration = _mapping(payload.get("joint_calibration"), "joint_calibration")
    if set(calibration) != set(JOINT_NAMES):
        raise ValueError("joint_calibration must contain exactly the 22 policy joints")
    bus_indices = {"d1": [], "piper": []}
    can_ids = {"d1": [], "piper": []}
    for policy_index, name in enumerate(JOINT_NAMES):
        entry = _mapping(calibration.get(name), f"joint_calibration.{name}")
        bus = entry.get("bus")
        expected = "d1" if policy_index < D1_COUNT else "piper"
        if bus != expected:
            raise ValueError(f"{name}: bus must be {expected}")
        if not require_complete:
            continue
        index = entry.get("bus_index")
        can_id = entry.get("can_id")
        direction = entry.get("direction")
        offset = entry.get("position_offset_rad")
        max_index = D1_COUNT if bus == "d1" else PIPER_COUNT
        if not isinstance(index, int) or not 0 <= index < max_index:
            raise ValueError(f"{name}: bus_index must be in [0, {max_index - 1}]")
        if not isinstance(can_id, int) or can_id < 0:
            raise ValueError(f"{name}: can_id must be a non-negative integer")
        if direction not in (-1, 1):
            raise ValueError(f"{name}: direction must be +1 or -1")
        if not isinstance(offset, (int, float)) or not np.isfinite(float(offset)):
            raise ValueError(f"{name}: position_offset_rad must be finite")
        bus_indices[bus].append(index)
        can_ids[bus].append(can_id)
    if require_complete:
        if sorted(bus_indices["d1"]) != list(range(D1_COUNT)):
            raise ValueError("D1 bus_index values must be a complete permutation of [0, 15]")
        if sorted(bus_indices["piper"]) != list(range(PIPER_COUNT)):
            raise ValueError("Piper bus_index values must be a complete permutation of [0, 5]")
        if len(set(can_ids["d1"])) != D1_COUNT or len(set(can_ids["piper"])) != PIPER_COUNT:
            raise ValueError("CAN IDs must be unique within each bus")
    return require_complete


def resolve_config(payload: Mapping[str, Any], *, base_dir: str | Path = ".") -> PolicyConfig:
    """Resolve the YAML schema, accepting legacy ``policy`` keys as aliases."""
    policy = _mapping(payload.get("policy"), "policy")
    control = _mapping(payload.get("control"), "control")
    actions = _mapping(payload.get("actions", payload.get("action")), "actions")
    robot = _mapping(payload.get("robot"), "robot")
    observations = _mapping(payload.get("observations", payload.get("observation")), "observations")
    simulation = _mapping(payload.get("simulation"), "simulation")
    runtime = _mapping(payload.get("runtime"), "runtime")
    commands = _mapping(payload.get("commands"), "commands")
    base = Path(base_dir).expanduser()

    default_pose = [0.425, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]
    command_velocity = _mapping(commands.get("velocity"), "commands.velocity")
    ee_command = _mapping(commands.get("ee_pose"), "commands.ee_pose")

    ee_position_limits = []
    configured_position_limits = ee_command.get("position_limits")
    default_ee_position_limits = ((0.15, 0.75), (-0.45, 0.45), (0.20, 0.90))
    if configured_position_limits is None:
        configured_position_limits = {
            "x": default_ee_position_limits[0],
            "y": default_ee_position_limits[1],
            "z": default_ee_position_limits[2],
        }
    if isinstance(configured_position_limits, Mapping):
        position_keys = ("x", "y", "z")
        configured_position_limits = [configured_position_limits.get(k) for k in position_keys]
    if len(configured_position_limits) != 3:
        raise ValueError("commands.ee_pose.position_limits must contain x/y/z pairs")
    for index, limits in enumerate(configured_position_limits):
        if limits is None or len(limits) != 2:
            raise ValueError(f"commands.ee_pose.position_limits[{index}] must be [lower, upper]")
        lower, upper = float(limits[0]), float(limits[1])
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"invalid commands.ee_pose.position_limits[{index}]={limits}")
        ee_position_limits.append((lower, upper))

    rpy_limits = ee_command.get("orientation_limits_rpy", (-0.6, 0.6))
    if isinstance(rpy_limits, Mapping):
        rpy_limits = [rpy_limits.get(k, (-0.6, 0.6)) for k in ("roll", "pitch", "yaw")]
    elif len(rpy_limits) == 2 and all(np.isscalar(v) for v in rpy_limits):
        rpy_limits = [rpy_limits] * 3
    if len(rpy_limits) != 3:
        raise ValueError("commands.ee_pose.orientation_limits_rpy must contain roll/pitch/yaw pairs")
    ee_orientation_limits = []
    for index, limits in enumerate(rpy_limits):
        if len(limits) != 2:
            raise ValueError(f"commands.ee_pose.orientation_limits_rpy[{index}] must be [lower, upper]")
        lower, upper = float(limits[0]), float(limits[1])
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"invalid commands.ee_pose.orientation_limits_rpy[{index}]={limits}")
        ee_orientation_limits.append((lower, upper))

    scales = observations.get("scales", observations.get("term_scales"))
    if isinstance(scales, Mapping):
        scales = [scales.get(name, OBSERVATION_SCALES[i])
                  for i, name in enumerate(OBSERVATION_TERM_NAMES)]
    obs_scales = _vector(scales, OBSERVATION_SCALES, 7, "observations.scales")
    noise = observations.get("noise_ranges", observations.get("noise"))
    if isinstance(noise, Mapping):
        noise = [noise.get(name, OBS_NOISE_RANGES[i])
                 for i, name in enumerate(OBSERVATION_TERM_NAMES)]
    if noise is None:
        noise = OBS_NOISE_RANGES
    if len(noise) != 7:
        raise ValueError("observations.noise_ranges must contain 7 [lower, upper] pairs")
    noise_ranges: list[tuple[float, float]] = []
    for index, item in enumerate(noise):
        if len(item) != 2:
            raise ValueError(f"observations.noise_ranges[{index}] must be [lower, upper]")
        lower, upper = float(item[0]), float(item[1])
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"invalid observations.noise_ranges[{index}]={item}")
        noise_ranges.append((lower, upper))

    joint_names = tuple(str(item) for item in robot.get("joints", payload.get("joints", JOINT_NAMES)))
    if joint_names != tuple(JOINT_NAMES):
        raise ValueError("robot.joints must match the fixed D1+Piper-L policy order")
    calibration = _mapping(payload.get("joint_calibration"), "joint_calibration")
    position_limits: list[tuple[float, float] | None] = []
    velocity_limits: list[float] = []
    for name in joint_names:
        entry = _mapping(calibration.get(name), f"joint_calibration.{name}")
        limits = entry.get("position_limits")
        if limits is None:
            position_limits.append(None)
        else:
            if len(limits) != 2:
                raise ValueError(f"joint_calibration.{name}.position_limits must be [lower, upper]")
            position_limits.append((float(limits[0]), float(limits[1])))
        velocity_limits.append(float(entry.get("velocity_limit_rad_s", np.inf)))
    dc_motor = _mapping(control.get("dc_motor"), "control.dc_motor")
    def _bool(value: Any, default: bool, name: str) -> bool:
        return parse_bool(value, default, name=name)

    cfg = PolicyConfig(
        path=_resolve_path(policy.get("path"), base),
        sha256=str(policy["sha256"]) if policy.get("sha256") else None,
        input_names=tuple(str(item) for item in (
            [policy.get("input_names")] if isinstance(policy.get("input_names"), str)
            else policy.get("input_names", [policy.get("input_name", "obs")])
        )),
        output_name=str(policy.get("output_name", "actions")),
        frequency_hz=float(policy.get("frequency_hz", 50.0)),
        history_length=int(policy.get("history_length", observations.get("history_length", 3))),
        frame_dim=int(policy.get("frame_dim", observations.get("frame_dim", 82))),
        action_dim=int(policy.get("action_dim", 22)),
        sim_dt=float(simulation.get("sim_dt", policy.get("sim_dt", 0.005))),
        decimation=int(simulation.get("decimation", policy.get("decimation", 4))),
        history_layout=str(policy.get("history_layout", "term-major")),
        action_clip=float(policy.get("action_clip", 100.0)),
        default_q=_vector(control.get("default_joint_angles", actions.get("default_joint_angles", policy.get("default_joint_angles"))), DEFAULT_Q, 22, "control.default_joint_angles"),
        action_scale=_vector(control.get("action_scales", actions.get("scales", policy.get("action_scale"))), ACTION_SCALE, 22, "control.action_scales"),
        kp=_vector(control.get("joint_kp", control.get("kp")), KP, 22, "control.joint_kp"),
        kd=_vector(control.get("joint_kd", control.get("kd")), KD, 22, "control.joint_kd"),
        torque_limit=_vector(control.get("torque_limit", control.get("torque_limits")), TORQUE_LIMIT, 22, "control.torque_limit"),
        wheel_indices=np.asarray(control.get("wheel_indices", WHEEL_INDICES), dtype=np.int64).reshape(-1),
        dc_saturation_effort=_vector(dc_motor.get("saturation_effort"), DC_SATURATION_EFFORT, 22, "control.dc_motor.saturation_effort", finite=False),
        dc_velocity_limit=_vector(dc_motor.get("velocity_limit"), DC_VELOCITY_LIMIT, 22, "control.dc_motor.velocity_limit", finite=False),
        observation_scales=obs_scales,
        observation_noise_ranges=tuple(noise_ranges),
        observation_clip=float(observations.get("clip", 100.0)),
        piper_delay_steps=int(control.get("piper_delay_steps", simulation.get("piper_delay_steps", 0))),
        initial_height=float(simulation.get("initial_height", 0.45)),
        command_velocity=_vector(command_velocity.get("default"), [0.0, 0.0, 0.0], 3, "commands.velocity.default"),
        ee_pose=_vector(ee_command.get("default"), default_pose, 7, "commands.ee_pose.default"),
        xml_path=_resolve_path(simulation.get("xml_path"), base),
        duration=float(simulation.get("duration", 30.0)),
        passive_damping_scale=float(simulation.get("passive_damping_scale", 0.0)),
        passive_frictionloss_scale=float(simulation.get("passive_frictionloss_scale", 1.0)),
        dc_motor_curve=_bool(simulation.get("dc_motor_curve", dc_motor.get("enabled")), True, "simulation.dc_motor_curve"),
        headless=_bool(simulation.get("headless"), True, "simulation.headless"),
        real_time=_bool(simulation.get("real_time"), False, "simulation.real_time"),
        print_interval=float(simulation.get("print_interval", 1.0)),
        stop_on_fall=_bool(simulation.get("stop_on_fall"), False, "simulation.stop_on_fall"),
        fall_height=float(simulation.get("fall_height", 0.25)),
        fall_tilt_deg=float(simulation.get("fall_tilt_deg", 35.0)),
        feedback_timeout_s=float(runtime.get("feedback_timeout_ms", 20.0)) / 1000.0,
        command_hz=float(runtime.get("command_hz", 500.0)),
        state_reader_hz=float(runtime.get("state_reader_hz", 500.0)),
        command_timeout_s=float(runtime.get("policy_watchdog_timeout_ms", 40.0)) / 1000.0,
        command_file_timeout_s=float(runtime.get("command_file_timeout_s", 0.5)),
        observation_noise=_bool(observations.get("enabled"), False, "observations.enabled"),
        joint_names=joint_names,
        actuator_names=tuple(str(item) for item in robot.get("actuators", ACTUATOR_NAMES)),
        position_limits=tuple(position_limits),
        velocity_limits=np.asarray(velocity_limits, np.float32),
        command_velocity_limits=_vector(commands.get("velocity_limits"), [1.0, 1.0, 1.5], 3, "commands.velocity_limits"),
        ee_position_limits=tuple(ee_position_limits),
        ee_orientation_limits_rpy=tuple(ee_orientation_limits),
    )
    if cfg.frame_dim != 82 or cfg.action_dim != 22:
        raise ValueError("D1+Piper-L adapter currently requires frame_dim=82 and action_dim=22")
    if cfg.history_length <= 0 or cfg.history_layout != "term-major":
        raise ValueError("history_length must be positive and history_layout must be term-major")
    if not np.isfinite([cfg.frequency_hz, cfg.sim_dt, cfg.action_clip,
                        cfg.feedback_timeout_s, cfg.command_hz,
                        cfg.state_reader_hz, cfg.command_timeout_s,
                        cfg.command_file_timeout_s]).all() or any(value <= 0 for value in (
                            cfg.frequency_hz, cfg.sim_dt, cfg.action_clip,
                            cfg.feedback_timeout_s, cfg.command_hz,
                            cfg.state_reader_hz, cfg.command_timeout_s,
                            cfg.command_file_timeout_s)):
        raise ValueError("policy frequency_hz, sim_dt and action_clip must be finite and positive")
    if cfg.wheel_indices.size != 4 or np.any(cfg.wheel_indices < 0) or np.any(cfg.wheel_indices >= 16):
        raise ValueError("control.wheel_indices must contain four D1 indices")
    if cfg.actuator_names != tuple(ACTUATOR_NAMES):
        raise ValueError("robot.actuators must match the fixed MuJoCo actuator order")
    if np.any(np.isnan(cfg.velocity_limits)) or np.any(cfg.velocity_limits < 0):
        raise ValueError("joint velocity limits must be non-negative (inf is allowed)")
    piper_bus = _mapping(_mapping(payload.get("buses"), "buses").get("piper"), "buses.piper")
    piper_bitrate = int(piper_bus.get("bitrate", 1_000_000))
    if piper_bitrate <= 0:
        raise ValueError("buses.piper.bitrate must be positive")
    if cfg.ee_pose[3:].dot(cfg.ee_pose[3:]) < 1.0e-12:
        raise ValueError("commands.ee_pose.default quaternion must be non-zero")
    cfg.ee_pose[3:] /= np.linalg.norm(cfg.ee_pose[3:])
    return cfg


def load_resolved_config(path: str | Path) -> PolicyConfig:
    config_path = Path(path).expanduser()
    return resolve_config(load_yaml(config_path), base_dir=config_path.parent)


def sanitize_commands(command_velocity: Sequence[float], ee_pose: Sequence[float],
                      config: PolicyConfig) -> tuple[np.ndarray, np.ndarray]:
    """Clamp live commands to the ranges used by the trained policy.

    The keyboard is only one possible command source.  ROS topics, JSON
    writers, and future teleoperation nodes must pass through the same limit
    layer so an out-of-range command cannot reach either the ONNX observation
    or a real actuator.  The returned quaternion is scalar-first ``wxyz`` and
    is rebuilt after clamping XYZ fixed-axis RPY limits.
    """
    velocity = np.asarray(command_velocity, dtype=np.float32).reshape(-1)
    pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
    if velocity.shape != (3,) or pose.shape != (7,):
        raise ValueError("command_velocity must be (3,) and ee_pose must be (7,)")
    if not np.isfinite(velocity).all() or not np.isfinite(pose).all():
        raise ValueError("commands must contain finite values")
    velocity = np.clip(velocity, -np.asarray(config.command_velocity_limits, np.float32),
                       np.asarray(config.command_velocity_limits, np.float32))
    position_limits = np.asarray(config.ee_position_limits, dtype=np.float32)
    pose[:3] = np.clip(pose[:3], position_limits[:, 0], position_limits[:, 1])
    quat = pose[3:].astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12:
        raise ValueError("ee_pose quaternion must be non-zero")
    w, x, y, z = quat / norm
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    rpy = np.asarray([roll, pitch, yaw], dtype=np.float64)
    rpy_limits = np.asarray(config.ee_orientation_limits_rpy, dtype=np.float64)
    rpy = np.clip(rpy, rpy_limits[:, 0], rpy_limits[:, 1])
    cr, cp, cy = np.cos(rpy / 2.0)
    sr, sp, sy = np.sin(rpy / 2.0)
    pose[3:] = np.asarray([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)
    pose[3:] /= np.linalg.norm(pose[3:])
    return velocity.astype(np.float32), pose.astype(np.float32)


DEFAULT_CONFIG = PolicyConfig()
