#!/usr/bin/env python3
"""Offline preflight for the D1 + Piper-L deployment configuration.

The default invocation only reads YAML and optionally hashes/inspects ONNX. It
does not import a CAN driver, construct a vendor object, enable an actuator, or
open ``can0``/``can1``.  ``--for-send`` validates the physical calibration and
the explicit dry-run gate, but still does not open hardware.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from loco_mani_config import (
        load_yaml, load_resolved_config, validate_hardware_config, parse_bool)
    from d1_piper_hardware import OnnxPolicy
except ImportError:  # installed-script fallback
    from loco_mani_rl_controller.scripts.loco_mani_config import (
        load_yaml, load_resolved_config, validate_hardware_config, parse_bool)
    from loco_mani_rl_controller.scripts.d1_piper_hardware import OnnxPolicy


def main() -> int:
    default_config = Path(__file__).resolve().parents[1] / "config/d1_piper_l.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--check-policy", action="store_true",
                        help="also verify the configured ONNX ABI and SHA256")
    parser.add_argument("--for-send", action="store_true",
                        help="validate complete calibration and live safety gates without opening CAN")
    args = parser.parse_args()

    try:
        payload = load_yaml(args.config)
        validate_hardware_config(payload, require_complete=args.for_send)
        resolved = load_resolved_config(args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        # Keep an operator-facing preflight refusal concise.  In particular,
        # the default YAML intentionally contains null calibration fields;
        # ``--for-send`` must reject that state without a traceback while
        # still returning a non-zero status and never opening CAN.
        raise SystemExit(f"[loco_mani] preflight failed: {exc}") from None
    safety = payload.get("safety", {})
    startup = payload.get("startup", {})
    if not isinstance(startup, dict):
        raise SystemExit("startup must be a YAML mapping")
    if not isinstance(safety, dict):
        raise SystemExit("safety must be a YAML mapping")
    try:
        dry_run = parse_bool(safety.get("dry_run"), True, name="safety.dry_run")
        enable_on_start = parse_bool(
            safety.get("enable_on_start"), False, name="safety.enable_on_start")
        base_height_check_enabled = parse_bool(
            safety.get("base_height_check_enabled"), False,
            name="safety.base_height_check_enabled")
        startup_enabled = parse_bool(
            startup.get("enabled"), False, name="startup.enabled")
    except ValueError as exc:
        raise SystemExit(f"[loco_mani] preflight failed: {exc}") from None
    report = {
        "config": str(args.config),
        "can_accessed": False,
        "calibration_complete": bool(args.for_send),
        "dry_run": dry_run,
        "enable_on_start": enable_on_start,
        "base_height_check_enabled": base_height_check_enabled,
        "startup_enabled": startup_enabled,
        "policy_path": resolved.path,
        "policy_checked": False,
        "feedback_timeout_ms": resolved.feedback_timeout_s * 1000.0,
        "command_hz": resolved.command_hz,
        "state_reader_hz": resolved.state_reader_hz,
        "policy_watchdog_timeout_ms": resolved.command_timeout_s * 1000.0,
    }
    if args.for_send and report["dry_run"]:
        raise SystemExit("--for-send requires safety.dry_run=false; no CAN was opened")
    if args.for_send and report["enable_on_start"]:
        raise SystemExit("safety.enable_on_start must remain false; use the explicit runtime gate")
    if args.for_send and not report["startup_enabled"]:
        raise SystemExit(
            "--for-send requires startup.enabled=true after the physical preparation pose "
            "has been verified; no CAN was opened"
        )
    if args.for_send and not resolved.path:
        raise SystemExit("--for-send requires policy.path; refusing a zero-action hardware run")
    if args.check_policy:
        if not resolved.path:
            raise SystemExit("policy.path is empty")
        policy = OnnxPolicy(resolved.path, expected_sha256=resolved.sha256, config=resolved)
        report.update({
            "policy_checked": True,
            "policy_sha256": policy.sha256,
            "policy_inputs": policy.input_names,
            "policy_output": policy.output_name,
            "policy_input_widths": policy.input_widths,
        })
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
