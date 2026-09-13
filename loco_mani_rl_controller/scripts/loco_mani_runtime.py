#!/usr/bin/env python3
"""D1 + Piper-L policy runtime with a deliberately latched hardware gate.

The module is the small amount of orchestration that is intentionally kept
out of :mod:`d1_piper_hardware`: vendor APIs are read from independent worker
threads, the policy is evaluated at 50 Hz, and the command path is serviced at
500 Hz.  Importing this file, constructing a runtime, and running it in the
default mode never opens ``can0`` or ``can1``.  A real D1 adapter must be
injected by the deployment image because ``CanfdApi`` is a C++ ABI.

The command is useful today for dry-run/shadow tests and is the reference
loop for the future ROS 2 component.  It is intentionally not a replacement
for a certified emergency-stop circuit.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

try:
    from command_file import CommandFileReader
except ImportError:  # source-tree import and installed script fallback
    _command_file_path = Path(__file__).with_name("command_file.py")
    _spec = importlib.util.spec_from_file_location("loco_mani_command_file", _command_file_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot load {_command_file_path}")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    CommandFileReader = _module.CommandFileReader

try:  # direct execution from the source tree
    from d1_piper_hardware import (
        D1Backend,
        HardwarePolicyRuntime,
        JointCalibration,
        JointCommand,
        OnnxPolicy,
        PiperBackend,
        SafetySupervisor,
        StateSnapshot,
        merge_snapshots,
    )
except ImportError:  # pragma: no cover - installed package path
    from loco_mani_rl_controller.scripts.d1_piper_hardware import (
        D1Backend,
        HardwarePolicyRuntime,
        JointCalibration,
        JointCommand,
        OnnxPolicy,
        PiperBackend,
        SafetySupervisor,
        StateSnapshot,
        merge_snapshots,
    )

try:
    from loco_mani_config import load_resolved_config, validate_hardware_config, parse_bool
except ImportError:  # installed/source script fallback
    _config_path = Path(__file__).with_name("loco_mani_config.py")
    _config_spec = importlib.util.spec_from_file_location("loco_mani_config", _config_path)
    if _config_spec is None or _config_spec.loader is None:
        raise ImportError(f"cannot load {_config_path}")
    _config_module = importlib.util.module_from_spec(_config_spec)
    sys.modules[_config_spec.name] = _config_module
    _config_spec.loader.exec_module(_config_module)
    load_resolved_config = _config_module.load_resolved_config
    validate_hardware_config = _config_module.validate_hardware_config
    parse_bool = _config_module.parse_bool


@dataclasses.dataclass
class _SnapshotBox:
    snapshot: StateSnapshot = dataclasses.field(default_factory=StateSnapshot)
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    def put(self, snapshot: StateSnapshot) -> None:
        with self.lock:
            self.snapshot = snapshot

    def get(self) -> StateSnapshot:
        with self.lock:
            return self.snapshot.copy()


class _Reader(threading.Thread):
    def __init__(self, backend: Any, box: _SnapshotBox, stop: threading.Event,
                 period_s: float) -> None:
        super().__init__(daemon=True)
        self.backend, self.box, self.stop, self.period_s = backend, box, stop, period_s
        self.exception: BaseException | None = None

    def run(self) -> None:  # noqa: D401 - thread entry point
        next_tick = time.monotonic()
        while not self.stop.is_set():
            try:
                self.box.put(self.backend.read())
            except BaseException as exc:  # preserve the failure for the owner
                self.exception = exc
                self.box.put(StateSnapshot(monotonic_ns=time.monotonic_ns(), error=str(exc)))
            next_tick += self.period_s
            delay = next_tick - time.monotonic()
            if delay > 0:
                self.stop.wait(delay)
            else:
                next_tick = time.monotonic()


@dataclasses.dataclass
class RuntimeStats:
    policy_steps: int = 0
    command_steps: int = 0
    send_failures: int = 0
    max_loop_time_us: float = 0.0
    last_fault: str = ""


class StartupManager:
    """Optional safe preparation phase before enabling the learned policy.

    The original D1 controller enters RL only after its transform-up/joint-PD
    sequence.  A combined D1+Piper system needs the same boundary, but the
    exact stand pose is hardware-specific.  This class therefore implements a
    configurable, conservative ramp to the policy default pose and remains
    disabled unless ``startup.enabled`` is explicitly set in YAML.  It never
    sends commands by itself; the normal safety and backend gates still apply.
    """

    def __init__(self, config: dict[str, Any], resolved: Any) -> None:
        startup = config.get("startup", {})
        if not isinstance(startup, dict):
            raise ValueError("startup must be a YAML mapping")
        self.enabled = parse_bool(startup.get("enabled"), False, name="startup.enabled")
        self.mode = str(startup.get("mode", "policy")).strip().lower()
        if self.mode not in {"policy", "hold", "ramp_default"}:
            raise ValueError("startup.mode must be policy, hold or ramp_default")
        if self.mode == "policy":
            self.enabled = False
        self.ramp_s = float(startup.get("ramp_s", 3.0))
        self.hold_s = float(startup.get("hold_s", 1.0))
        if not np.isfinite([self.ramp_s, self.hold_s]).all() or self.ramp_s < 0.0 or self.hold_s < 0.0:
            raise ValueError("startup.ramp_s and startup.hold_s must be finite and non-negative")
        self.default_q = np.asarray(resolved.default_q, np.float32).copy()
        self.startup_kp = np.asarray(
            startup.get("kp", resolved.kp), np.float32
        ).reshape(-1)
        self.startup_kd = np.asarray(
            startup.get("kd", resolved.kd), np.float32
        ).reshape(-1)
        if self.default_q.shape != (22,) or self.startup_kp.shape != (22,) or self.startup_kd.shape != (22,):
            raise ValueError("startup default_q/kp/kd must contain 22 values")
        if not np.isfinite(self.default_q).all() or not np.isfinite(self.startup_kp).all() or not np.isfinite(self.startup_kd).all():
            raise ValueError("startup default_q/kp/kd must be finite")
        self.wheel_indices = np.asarray(resolved.wheel_indices, np.int64)
        self.reset()

    def reset(self) -> None:
        self.phase = "wait_state" if self.enabled else "policy"
        self._initial_q: np.ndarray | None = None
        self._phase_start = 0.0

    @property
    def active(self) -> bool:
        return self.phase != "policy"

    def update(self, snapshot: StateSnapshot, now: float) -> JointCommand | None:
        """Return a preparation command, or ``None`` once policy may run."""
        if not self.enabled:
            return None
        if not snapshot.valid:
            raise RuntimeError("startup cannot run with invalid state")
        if self.phase == "wait_state":
            self._initial_q = snapshot.q.copy()
            self._phase_start = float(now)
            self.phase = "hold" if self.mode == "hold" or self.ramp_s <= 0.0 else "ramp_default"
        assert self._initial_q is not None
        elapsed = max(0.0, float(now) - self._phase_start)
        if self.phase == "ramp_default":
            alpha = min(1.0, elapsed / max(self.ramp_s, 1.0e-9))
            target = self._initial_q + alpha * (self.default_q - self._initial_q)
            if alpha >= 1.0:
                self.phase = "hold"
                self._phase_start = float(now)
        elif self.phase == "hold":
            target = self.default_q.copy() if self.mode == "ramp_default" else self._initial_q.copy()
            if elapsed >= self.hold_s:
                self.phase = "policy"
        else:
            return None
        # Wheels are velocity actuators in the learned policy.  During the
        # preparation phase hold their measured angle and command zero speed.
        target[self.wheel_indices] = snapshot.q[self.wheel_indices]
        return JointCommand(
            position=target.astype(np.float32),
            velocity=np.zeros(22, np.float32),
            kp=self.startup_kp.copy(),
            kd=self.startup_kd.copy(),
            torque=np.zeros(22, np.float32),
        )


class HardwarePolicyRuntimeLoop:
    """Two-rate policy/command loop shared by dry-run and real deployments."""

    def __init__(self, d1: D1Backend, piper: PiperBackend,
                 policy_runtime: HardwarePolicyRuntime,
                 safety: SafetySupervisor, *, policy_hz: float = 50.0,
                 command_hz: float = 500.0, state_reader_hz: float | None = None,
                 hardware_gate: bool = False,
                 send_enabled: bool = False,
                 enable_piper: bool = False,
                 d1_force_direct: bool = False,
                 startup: StartupManager | None = None,
                 hardware_access: bool = False,
                 read_only: bool = False) -> None:
        if not np.isfinite(policy_hz) or policy_hz <= 0:
            raise ValueError("policy_hz must be finite and positive")
        if not np.isfinite(command_hz) or command_hz <= 0:
            raise ValueError("command_hz must be finite and positive")
        if state_reader_hz is None:
            state_reader_hz = command_hz
        if not np.isfinite(state_reader_hz) or state_reader_hz <= 0:
            raise ValueError("state_reader_hz must be finite and positive")
        self.d1, self.piper = d1, piper
        self.policy_runtime, self.safety = policy_runtime, safety
        self.policy_period = 1.0 / float(policy_hz)
        self.command_period = 1.0 / float(command_hz)
        self.state_reader_period = 1.0 / float(state_reader_hz)
        self.hardware_gate = bool(hardware_gate)
        self.send_enabled = bool(send_enabled)
        self.enable_piper = bool(enable_piper)
        self.d1_force_direct = bool(d1_force_direct)
        self.hardware_access = bool(hardware_access or send_enabled)
        self.read_only = bool(read_only)
        if self.read_only and self.send_enabled:
            raise ValueError("read-only runtime cannot send actuator commands")
        # Store the startup manager before validating the real-send path.  The
        # validation below intentionally refuses a hardware run without an
        # explicit preparation phase; referring to ``self.startup`` before it
        # is assigned would turn that safety check into an AttributeError.
        self.startup = startup
        if self.send_enabled and not self.enable_piper:
            raise ValueError(
                "real send requires explicit enable_piper=True; refusing to assume Piper motor enable"
            )
        if self.send_enabled and (self.startup is None or not self.startup.enabled):
            raise ValueError(
                "real send requires an explicitly enabled startup preparation phase"
            )
        if self.d1_force_direct and not self.send_enabled:
            raise ValueError("d1_force_direct requires send_enabled=True")
        self.stop = threading.Event()
        self.d1_box, self.piper_box = _SnapshotBox(), _SnapshotBox()
        self.stats = RuntimeStats()
        self._readers: list[_Reader] = []
        self._last_command = JointCommand()
        self._last_policy_ns = 0

    def _safe_send(self) -> None:
        """Best-effort zero command; never hide the original fault."""
        if not self.send_enabled:
            return
        # Keep the buses independent.  If can0 teardown is already failing,
        # Piper must still receive its zero/stop attempt, and vice versa.
        try:
            self.d1.send(JointCommand(), hardware_gate=self.hardware_gate)
        except Exception as exc:  # pragma: no cover - hardware-only path
            self.stats.send_failures += 1
            self.stats.last_fault = f"D1 safe command failed: {exc}"
        try:
            if self.piper.enabled:
                self.piper.send(JointCommand(), hardware_gate=self.hardware_gate)
        except Exception as exc:  # pragma: no cover - hardware-only path
            self.stats.send_failures += 1
            if not self.stats.last_fault:
                self.stats.last_fault = f"Piper safe command failed: {exc}"

    def start(self) -> None:
        # Backends enforce their own gate/calibration checks.  Calling connect
        # here is safe in dry-run and fails closed for an incomplete real setup.
        try:
            self.d1.connect(hardware_gate=self.hardware_gate, read_only=self.read_only)
            self.piper.connect(hardware_gate=self.hardware_gate, read_only=self.read_only)
        except Exception:
            # If the second backend fails during startup, release the first
            # one immediately.  Otherwise a failed launch could leave can0
            # open even though no control loop was ever entered.
            try:
                self.d1.disconnect()
            finally:
                self.piper.disconnect()
            raise
        self._readers = [
            _Reader(self.d1, self.d1_box, self.stop, self.state_reader_period),
            _Reader(self.piper, self.piper_box, self.stop, self.state_reader_period),
        ]
        for reader in self._readers:
            reader.start()

    def _wait_for_initial_state(self, timeout_s: float = 0.5) -> None:
        """Require valid, timely feedback from *both* buses before enable."""
        deadline = time.monotonic() + float(timeout_s)
        last_error = "waiting for D1/Piper feedback"
        while time.monotonic() < deadline and not self.stop.is_set():
            snapshot = merge_snapshots(self.d1_box.get(), self.piper_box.get())
            if snapshot.valid:
                self.safety.check_state(snapshot, enforce_joint_limits=not self.read_only)
                if not self.safety.fault:
                    return
                last_error = self.safety.fault
            elif snapshot.error:
                last_error = snapshot.error
            self.stop.wait(0.002)
        raise RuntimeError(f"initial hardware feedback gate failed: {last_error}")

    def stop_and_disconnect(self) -> None:
        self.stop.set()
        for reader in self._readers:
            reader.join(timeout=0.5)
        self._safe_send()
        # Piper provides a documented damped E-stop.  Use it whenever the
        # process owned enabled real actuators, including Ctrl-C.  D1 requires
        # a deployment-specific adapter method; do not guess its RPC ABI.
        if self.send_enabled:
            try:
                self.piper.emergency_stop(hardware_gate=self.hardware_gate)
            except Exception as exc:  # pragma: no cover - hardware-only path
                self.stats.send_failures += 1
                if not self.stats.last_fault:
                    self.stats.last_fault = f"Piper emergency stop failed: {exc}"
            try:
                self.d1.emergency_stop(hardware_gate=self.hardware_gate)
            except Exception:
                # A zero D1 command was already attempted above.  Lack of an
                # optional adapter E-stop must not mask the primary fault.
                pass
        # Cleanup is deliberately independent: a vendor destructor must not
        # prevent the other CAN interface from being closed.
        try:
            self.d1.disconnect()
        except Exception as exc:  # pragma: no cover - hardware-only path
            self.stats.last_fault = self.stats.last_fault or f"D1 disconnect failed: {exc}"
        try:
            self.piper.disconnect()
        except Exception as exc:  # pragma: no cover - hardware-only path
            self.stats.last_fault = self.stats.last_fault or f"Piper disconnect failed: {exc}"

    def run(self, duration_s: float, command_velocity: Sequence[float],
            ee_pose: Sequence[float], command_file: CommandFileReader | None = None) -> RuntimeStats:
        if not np.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and non-negative (0 means unlimited)")
        velocity = np.asarray(command_velocity, np.float32).reshape(-1)
        pose = np.asarray(ee_pose, np.float32).reshape(-1)
        if velocity.size != 3 or pose.size != 7 or not np.isfinite(velocity).all() or not np.isfinite(pose).all():
            raise ValueError("command_velocity must be 3 finite values and ee_pose 7 finite values")
        self.policy_runtime.reset()
        if self.startup is not None:
            self.startup.reset()
        self._last_policy_ns = 0
        try:
            self.start()
            # The reader threads start asynchronously, including in dry-run.
            # Do not let the first policy tick observe the intentionally empty
            # snapshot and latch a false safety fault.  Waiting for one
            # complete D1+Piper sample is also the correct startup contract
            # for real hardware; the backend-specific gates have already
            # decided whether opening the buses was allowed.
            self._wait_for_initial_state()
            if self.send_enabled and self.d1_force_direct:
                # This is the only place where the runtime can change the D1
                # MCU mode.  It remains behind both --send and
                # --hardware-gate, and is never used by shadow/read-only runs.
                self.d1.set_force_direct(hardware_gate=self.hardware_gate)
            if self.send_enabled:
                self.piper.enable(hardware_gate=self.hardware_gate)
            started = time.monotonic()
            next_policy = started
            next_command = started
            while not self.stop.is_set() and (duration_s == 0.0 or time.monotonic() - started < duration_s):
                now = time.monotonic()
                if command_file is not None:
                    live_command = command_file.read()
                    if live_command is not None:
                        velocity, pose = live_command
                if now >= next_policy:
                    snapshot = merge_snapshots(self.d1_box.get(), self.piper_box.get())
                    self.safety.check_state(snapshot, enforce_joint_limits=not self.read_only)
                    if self.safety.fault:
                        self.stats.last_fault = self.safety.fault
                        self.stop.set()
                        break
                    startup_command = (
                        self.startup.update(snapshot, now) if self.startup is not None else None
                    )
                    if startup_command is None:
                        _, _, command = self.policy_runtime.step(snapshot, velocity, pose)
                        self._last_command = command
                    else:
                        self._last_command = startup_command
                    self._last_policy_ns = time.monotonic_ns()
                    self.stats.policy_steps += 1
                    next_policy += self.policy_period
                    if next_policy < now:
                        next_policy = now + self.policy_period

                if now >= next_command:
                    loop_start = time.monotonic_ns()
                    if self._last_policy_ns == 0 or time.monotonic_ns() - self._last_policy_ns > self.safety.command_timeout_ns:
                        self.safety.trigger("policy watchdog timeout")
                    # Re-read the merged bus snapshot at the command rate.
                    # The policy is intentionally only 50 Hz, but the
                    # feedback watchdog must not wait for the next policy
                    # tick: a CAN/IMU stream can stop between two inference
                    # updates.  Checking the current snapshot here bounds
                    # stale-state command output to one low-level period and
                    # also applies feedback joint/velocity limits to the
                    # state actually paired with this command.
                    command_snapshot = merge_snapshots(
                        self.d1_box.get(), self.piper_box.get())
                    self.safety.check_state(
                        command_snapshot,
                        enforce_joint_limits=not self.read_only,
                    )
                    command = self.safety.limit_command(
                        self._last_command, state=command_snapshot,
                        dt=self.command_period)
                    if self.safety.fault:
                        self.stats.last_fault = self.safety.fault
                        self.stop.set()
                        break
                    if self.send_enabled:
                        try:
                            ok_d1 = self.d1.send(command, hardware_gate=self.hardware_gate)
                            ok_piper = self.piper.send(command, hardware_gate=self.hardware_gate)
                            if not (ok_d1 and ok_piper):
                                raise RuntimeError("vendor backend rejected command")
                        except Exception as exc:  # pragma: no cover - hardware-only path
                            self.stats.send_failures += 1
                            self.safety.trigger(f"command send failed: {exc}")
                            self.stats.last_fault = self.safety.fault
                            self.stop.set()
                            break
                    self.stats.command_steps += 1
                    elapsed_us = (time.monotonic_ns() - loop_start) / 1e3
                    self.stats.max_loop_time_us = max(self.stats.max_loop_time_us, elapsed_us)
                    next_command += self.command_period
                    if next_command < now:
                        next_command = now + self.command_period
                delay = min(next_policy, next_command) - time.monotonic()
                if delay > 0:
                    self.stop.wait(min(delay, 0.002))
        finally:
            self.stop_and_disconnect()
        return self.stats


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("runtime config must contain a YAML mapping")
    return payload


def _load_symbol(spec: str) -> Callable[..., Any]:
    """Load ``package.module:factory`` without making it a hard dependency."""
    if ":" not in spec:
        raise ValueError(f"factory must use module:callable syntax, got {spec!r}")
    module_name, symbol_name = spec.split(":", 1)
    if not module_name or not symbol_name:
        raise ValueError(f"factory must use module:callable syntax, got {spec!r}")
    symbol: Any = importlib.import_module(module_name)
    for part in symbol_name.split("."):
        symbol = getattr(symbol, part)
    if not callable(symbol):
        raise TypeError(f"factory symbol is not callable: {spec}")
    return symbol


def _build_safety(config: dict[str, Any], resolved=None) -> SafetySupervisor:
    entries = config.get("joint_calibration", {})
    names = config.get("joints", [])
    position_limits = [entries[name].get("position_limits") for name in names]
    velocity_limits = [float(entries[name].get("velocity_limit_rad_s", np.inf)) for name in names]
    torque_limits = [float(entries[name].get("torque_limit_nm", np.inf)) for name in names]
    safety = config.get("safety", {})
    if resolved is not None:
        position_limits = [entries[name].get("position_limits") for name in names]
        velocity_limits = resolved.velocity_limits if hasattr(resolved, "velocity_limits") else velocity_limits
        # Enforce both the policy envelope and the per-joint physical envelope
        # from calibration.  A newly adapted policy must never raise a motor's
        # hardware limit merely by changing control.torque_limit.
        configured_torque = np.asarray(torque_limits, dtype=np.float32)
        policy_torque = np.asarray(resolved.torque_limit, dtype=np.float32)
        if configured_torque.shape == policy_torque.shape:
            torque_limits = np.minimum(configured_torque, policy_torque)
        else:
            torque_limits = policy_torque
    return SafetySupervisor(
        state_timeout_ms=float(safety.get("state_timeout_ms", 20.0)),
        command_timeout_ms=float(safety.get("command_timeout_ms", 40.0)),
        max_tilt_deg=float(safety.get("max_tilt_deg", 35.0)),
        min_base_height_m=float(safety.get("min_base_height_m", 0.25)),
        max_torque_rate_nm_s=float(safety.get("max_torque_rate_nm_s", 300.0)),
        require_base_height=parse_bool(
            safety.get("base_height_check_enabled"), False,
            name="safety.base_height_check_enabled"),
        torque_limits=torque_limits,
        position_limits=position_limits,
        velocity_limits=velocity_limits,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[1] / "config/d1_piper_l.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--policy-path", default="", help="empty uses the zero-action policy")
    # ``None`` is intentional here.  A concrete value (including ``5`` or
    # ``0,0,0``) supplied on the command line must remain an explicit user
    # override; using those values as sentinels silently replaced them with
    # the YAML defaults in the old implementation.
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--command-velocity", default=None)
    parser.add_argument("--ee-pose", default=None)
    parser.add_argument("--command-file", default="",
                        help="live JSON command file written by loco_mani_keyboard")
    parser.add_argument("--command-file-timeout", type=float, default=0.5)
    parser.add_argument("--send", action="store_true", help="send commands; disabled by default")
    parser.add_argument(
        "--shadow", action="store_true",
        help="open both real buses and run read/policy checks without enabling or sending motors",
    )
    parser.add_argument("--hardware-gate", action="store_true", help="explicitly permit vendor connect/send")
    parser.add_argument("--enable-piper", action="store_true",
                        help="explicit third gate to enable Piper motors before MIT sending")
    parser.add_argument(
        "--d1-force-direct", action="store_true",
        help="explicitly request D1 SET_READY_NEXT=FORCE_DIRECT before sending",
    )
    parser.add_argument(
        "--d1-api-factory", default="d1_tita_adapter:create_d1_api",
        help="module:factory returning the validated flat-C D1 adapter")
    parser.add_argument(
        "--d1-motor-out-factory", default="d1_tita_adapter:make_motor_out",
        help="module:factory constructing D1 command records")
    args = parser.parse_args()
    config = _load_yaml(args.config)
    validate_hardware_config(config, require_complete=args.send)
    resolved = load_resolved_config(args.config)
    policy_cfg = config.get("policy", {})
    # Keep the no-argument command offline and dependency-light.  Passing an
    # explicit --policy-path opts into ONNX verification; --send also requires
    # the configured policy so a real actuator cannot run a zero stub.
    hardware_access = bool(args.send or args.shadow)
    if args.send and args.shadow:
        raise SystemExit("--send and --shadow are mutually exclusive")
    # An explicitly supplied checkpoint is safe to load in dry-run mode and
    # is useful for validating the complete policy/observation/action chain
    # without opening either CAN bus.  When no path is supplied, retain the
    # dependency-light zero-action dry-run; real/shadow hardware still uses
    # the configured policy path and refuses to run a stub.
    policy_path = args.policy_path or (resolved.path if hardware_access else "")
    if hardware_access and not policy_path:
        raise SystemExit("hardware mode requires a policy path; refusing zero-action control")
    expected_sha = resolved.sha256

    # YAML supplies values only when the corresponding CLI option was omitted.
    # Do not compare against a valid command value as a sentinel: zero velocity
    # and a five-second run are both legitimate explicit requests.
    if args.duration is None:
        args.duration = float(config.get("simulation", {}).get("duration", 5.0))
    if args.command_velocity is None:
        args.command_velocity = ",".join(str(float(v)) for v in resolved.command_velocity)
    if args.ee_pose is None:
        args.ee_pose = ",".join(str(float(v)) for v in resolved.ee_pose)

    if policy_path:
        policy_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = OnnxPolicy(
            policy_path, expected_sha256=expected_sha, config=resolved
        )
    else:
        policy_fn = lambda _frame, _history: np.zeros(resolved.action_dim, np.float32)
    runtime = HardwarePolicyRuntime(
        policy_fn, max_action_abs=float(config.get("safety", {}).get("max_action_abs", resolved.action_clip)),
        config=resolved,
    )
    if hardware_access and not args.hardware_gate:
        raise SystemExit("hardware mode requires --hardware-gate; no CAN access is implied by config alone")
    if args.send and not args.enable_piper:
        raise SystemExit("--send requires --enable-piper; refusing to enable/command Piper implicitly")
    startup_cfg = config.get("startup", {})
    safety_cfg = config.get("safety", {})
    # ``--send`` is intentionally a two-sided opt-in: the command line gate
    # is not enough to override a YAML file that still declares workstation
    # dry-run mode.  This prevents copying a known-safe config to a Jetson and
    # accidentally making it live by adding only launch flags.
    dry_run = parse_bool(safety_cfg.get("dry_run"), True, name="safety.dry_run")
    enable_on_start = parse_bool(
        safety_cfg.get("enable_on_start"), False, name="safety.enable_on_start")
    startup_enabled = parse_bool(
        startup_cfg.get("enabled"), False, name="startup.enabled")
    if args.send and dry_run:
        raise SystemExit(
            "--send requires safety.dry_run=false in the YAML after the physical preparation checks"
        )
    if args.send and enable_on_start:
        raise SystemExit(
            "safety.enable_on_start must remain false; use the explicit --enable-piper gate"
        )
    if args.send and not startup_enabled:
        raise SystemExit("--send requires startup.enabled=true in the YAML after the physical preparation pose is verified")
    # A partial/null calibration is intentionally converted to identity only
    # for dry-run.  When --send is requested, from_config rejects every null
    # field before either CAN interface is opened.
    # Shadow/inspection opens the buses only for feedback and therefore does
    # not require the policy-to-CAN calibration yet.  A send run remains
    # blocked until every ID/sign/offset is explicitly filled.
    calibration = JointCalibration.from_config(config, require_complete=args.send)
    # Never allow a real send to rely on the identity fallback returned for a
    # partially populated YAML.  The backend repeats this check at connect,
    # but failing before factory loading gives a clearer operator error and
    # prevents any future adapter from opening a bus too early.
    if args.send and not calibration.complete:
        raise SystemExit(
            "--send requires a complete joint_calibration mapping: fill every "
            "bus_index/can_id/direction/position_offset_rad explicitly"
        )
    runtime_cfg = config.get("runtime", {})
    feedback_timeout_s = float(resolved.feedback_timeout_s)

    d1 = D1Backend(
        interface=config.get("buses", {}).get("d1", {}).get("interface", "can0"),
        dry_run=not hardware_access,
        calibration=calibration,
        config=resolved,
        feedback_timeout_s=feedback_timeout_s,
        api_factory=_load_symbol(args.d1_api_factory) if args.d1_api_factory else None,
        motor_out_factory=_load_symbol(args.d1_motor_out_factory) if args.d1_motor_out_factory else None,
    )
    piper_cfg = config.get("buses", {}).get("piper", {})
    piper = PiperBackend(
        interface=piper_cfg.get("interface", "can1"), robot=piper_cfg.get("robot", "piper_l"),
        bitrate=int(piper_cfg.get("bitrate", 1_000_000)),
        firmware_profile=piper_cfg.get("firmware_profile", "auto"),
        dry_run=not hardware_access, calibration=calibration, config=resolved,
        feedback_timeout_s=feedback_timeout_s,
    )
    loop = HardwarePolicyRuntimeLoop(
        d1, piper, runtime, _build_safety(config, resolved),
        policy_hz=resolved.frequency_hz,
        command_hz=float(resolved.command_hz),
        state_reader_hz=float(resolved.state_reader_hz),
        hardware_gate=args.hardware_gate, send_enabled=args.send, enable_piper=args.enable_piper,
        d1_force_direct=args.d1_force_direct,
        startup=StartupManager(config, resolved), hardware_access=hardware_access,
        read_only=args.shadow,
    )
    signal.signal(signal.SIGINT, lambda _sig, _frame: loop.stop.set())
    signal.signal(signal.SIGTERM, lambda _sig, _frame: loop.stop.set())
    stats = loop.run(
        args.duration,
        [float(v) for v in args.command_velocity.split(",")],
        [float(v) for v in args.ee_pose.split(",")],
        CommandFileReader(args.command_file, args.command_file_timeout) if args.command_file else None,
    )
    print(json.dumps(dataclasses.asdict(stats), indent=2))
    return 0 if not stats.last_fault else 2


if __name__ == "__main__":
    raise SystemExit(main())
