#!/usr/bin/env python3
"""Check the supplied ONNX policy against the D1+Piper-L deployment ABI.

This is deliberately read-only: it never opens CAN and never enables a motor.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=Path)
    args = parser.parse_args()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required") from exc
    if not args.policy.is_file():
        raise SystemExit(f"policy not found: {args.policy}")
    session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) < 1:
        raise SystemExit(f"expected one input and at least one output; got {len(inputs)}/{len(outputs)}")
    shape = inputs[0].shape
    out_shape = outputs[0].shape
    input_type = getattr(inputs[0], "type", "")
    output_type = getattr(outputs[0], "type", "")
    dim = shape[-1] if shape else None
    out_dim = out_shape[-1] if out_shape else None
    # Dynamic batch dimensions are common in exports; only the feature width
    # is part of this deployment ABI.
    if dim != 246 or out_dim != 22:
        raise SystemExit(f"ABI mismatch: input={shape}, output={out_shape}; expected [1,246] -> [1,22]")
    if input_type and input_type != "tensor(float)":
        raise SystemExit(f"ABI mismatch: input type={input_type}; expected tensor(float)")
    if output_type and output_type != "tensor(float)":
        raise SystemExit(f"ABI mismatch: output type={output_type}; expected tensor(float)")
    sample = np.zeros((1, 246), dtype=np.float32)
    action = np.asarray(session.run(None, {inputs[0].name: sample})[0])
    if action.shape != (1, 22) or not np.isfinite(action).all():
        raise SystemExit(f"invalid inference output: shape={action.shape}")
    digest = hashlib.sha256(args.policy.read_bytes()).hexdigest()
    print(f"OK: {args.policy} input={inputs[0].name}{shape} output={outputs[0].name}{out_shape}")
    print(f"types: input={input_type or 'unknown'} output={output_type or 'unknown'}")
    print(f"sha256: {digest}")
    print(f"zero-observation output range=[{action.min():.6g}, {action.max():.6g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
