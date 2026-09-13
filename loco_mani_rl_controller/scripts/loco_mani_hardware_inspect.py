#!/usr/bin/env python3
"""Explicit, read-only D1/Piper feedback inspection tool.

This command can open a real CAN interface only with ``--hardware-gate``.  It
never enables Piper, never calls a motion command, and never sends a D1 packet.
It is intended to collect raw bus-order feedback before filling calibration.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from d1_piper_hardware import D1Backend, JointCalibration, PiperBackend
    from loco_mani_config import load_resolved_config, validate_hardware_config
except ImportError:  # installed-script fallback
    from loco_mani_rl_controller.scripts.d1_piper_hardware import D1Backend, JointCalibration, PiperBackend
    from loco_mani_rl_controller.scripts.loco_mani_config import load_resolved_config, validate_hardware_config


def _factory(spec: str):
    if ":" not in spec:
        raise ValueError("factory must use module:callable syntax")
    module, symbol = spec.split(":", 1)
    value = importlib.import_module(module)
    for part in symbol.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"factory is not callable: {spec}")
    return value


def _finite(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).resolve().parents[1] / "config/d1_piper_l.yaml"
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--bus", choices=("d1", "piper", "both"), default="both")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--period", type=float, default=0.1)
    parser.add_argument("--hardware-gate", action="store_true",
                        help="explicitly allow opening the selected CAN bus(es)")
    parser.add_argument("--d1-api-factory", default="d1_tita_adapter:create_d1_api")
    parser.add_argument("--d1-motor-out-factory", default="")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.samples <= 0 or not np.isfinite(args.period) or args.period <= 0:
        raise SystemExit("samples must be positive and period must be finite/positive")
    if not args.hardware_gate:
        raise SystemExit("read-only hardware inspection requires --hardware-gate; no CAN is opened by default")

    payload = __import__("yaml").safe_load(args.config.read_text(encoding="utf-8"))
    validate_hardware_config(payload, require_complete=False)
    resolved = load_resolved_config(args.config)
    calibration = JointCalibration.from_config(payload, require_complete=False)
    buses = payload.get("buses", {})
    d1 = piper = None
    firmware = None
    try:
        if args.bus in {"d1", "both"}:
            d1_cfg = buses.get("d1", {})
            d1 = D1Backend(
                interface=d1_cfg.get("interface", "can0"), dry_run=False,
                calibration=calibration, config=resolved,
                api_factory=_factory(args.d1_api_factory),
            )
            d1.connect(hardware_gate=True, read_only=True)
        if args.bus in {"piper", "both"}:
            piper_cfg = buses.get("piper", {})
            piper = PiperBackend(
                interface=piper_cfg.get("interface", "can1"),
                robot=piper_cfg.get("robot", "piper_l"),
                bitrate=int(piper_cfg.get("bitrate", 1_000_000)),
                firmware_profile=piper_cfg.get("firmware_profile", "auto"),
                dry_run=False, calibration=calibration, config=resolved,
            )
            piper.connect(hardware_gate=True, read_only=True)
            # ``connect()`` already performs the SDK's firmware probe when
            # firmware_profile=auto.  Querying it again here is useful for a
            # report, but it is still read-only and must never be mistaken
            # for a motion/enable operation.
            firmware = piper.firmware_info()
        else:
            firmware = None

        samples = []
        for index in range(args.samples):
            item = {"sample": index, "monotonic_ns": time.monotonic_ns()}
            if d1 is not None:
                state = d1.read()
                item["d1"] = {
                    "valid": bool(state.valid), "error": state.error,
                    "q_bus_or_policy": _finite(state.q[:16]),
                    "dq_bus_or_policy": _finite(state.dq[:16]),
                    "tau_bus_or_policy": _finite(state.tau[:16]),
                    "imu_quat_wxyz": _finite(state.quat_wxyz),
                    "gyro": _finite(state.gyro), "accel": _finite(state.accel),
                }
            if piper is not None:
                state = piper.read()
                item["piper"] = {
                    "valid": bool(state.valid), "error": state.error,
                    "q_bus_or_policy": _finite(state.q[16:]),
                    "dq_bus_or_policy": _finite(state.dq[16:]),
                    "tau_bus_or_policy": _finite(state.tau[16:]),
                }
            print(json.dumps(item, ensure_ascii=False), flush=True)
            samples.append(item)
            if index + 1 < args.samples:
                time.sleep(args.period)
        report = {
            "read_only": True, "hardware_gate": True,
            "calibration_complete": bool(calibration.complete),
            "samples": samples,
            "piper_firmware": firmware,
            "note": "Values are raw bus order when calibration is incomplete; do not copy them into policy calibration without manual mapping.",
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finally:
        if d1 is not None:
            try:
                d1.disconnect()
            except Exception as exc:
                print(f"D1 inspection disconnect failed: {exc}", file=sys.stderr)
        if piper is not None:
            try:
                piper.disconnect()
            except Exception as exc:
                print(f"Piper inspection disconnect failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
