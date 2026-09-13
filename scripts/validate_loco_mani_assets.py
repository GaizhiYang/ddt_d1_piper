#!/usr/bin/env python3
"""Read-only validation for the D1 + Piper-L deployment contract.

This command intentionally does not open CAN devices or enable actuators.  It
checks the things that can be checked on a workstation: the ONNX ABI, the
named MuJoCo joints/actuators, reset observation dimensions and finite model
state.  Hardware IDs, signs and zero offsets remain explicit field checks in
the YAML and must be validated on a supported test stand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

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
DEFAULT_Q = [0.0, 0.8, -1.5, 0.0] * 4 + [0.0] * 6


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--dump", type=Path,
                        help="optional JSON report path")
    args = parser.parse_args()

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    missing_joints = [
        n for n in JOINT_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) < 0
    ]
    missing_actuators = [
        n for n in ACTUATOR_NAMES
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) < 0
    ]
    if missing_joints or missing_actuators:
        raise SystemExit(
            f"model ABI mismatch: missing joints={missing_joints}, "
            f"actuators={missing_actuators}"
        )

    data = mujoco.MjData(model)
    # Evaluate contacts at the same reset pose used by the rollout rather
    # than MuJoCo's all-zero joint pose (which puts the calves through the
    # floor and makes a contact count meaningless).
    if model.nq >= 7:
        data.qpos[2] = 0.45
        # A free joint stores a quaternion in wxyz order.  Leaving the
        # zero-initialized quaternion untouched makes the diagnostic pose
        # invalid even though all values are finite.
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, value in zip(JOINT_NAMES, DEFAULT_Q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[int(model.jnt_qposadr[jid])] = value
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise SystemExit("model reset produced non-finite qpos/qvel")

    report: dict[str, object] = {
        "xml": str(args.xml),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "contacts_after_forward": int(data.ncon),
        "max_contact_penetration_m": float(max(
            (max(0.0, -float(data.contact[i].dist)) for i in range(data.ncon)),
            default=0.0,
        )),
        "finite_reset": True,
        "joint_names": JOINT_NAMES,
        "actuator_names": ACTUATOR_NAMES,
    }
    if args.policy:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("onnxruntime is required when --policy is supplied") from exc
        if not args.policy.is_file():
            raise SystemExit(f"policy not found: {args.policy}")
        session = ort.InferenceSession(str(args.policy), providers=["CPUExecutionProvider"])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        shape = inputs[0].shape if inputs else []
        out_shape = outputs[0].shape if outputs else []
        if len(inputs) != 1 or not outputs or shape[-1] != 246 or out_shape[-1] != 22:
            raise SystemExit(f"policy ABI mismatch: input={shape}, output={out_shape}")
        sample = np.zeros((1, 246), dtype=np.float32)
        action = np.asarray(session.run(None, {inputs[0].name: sample})[0])
        if action.shape != (1, 22) or not np.isfinite(action).all():
            raise SystemExit(f"policy inference invalid: shape={action.shape}")
        report["policy"] = str(args.policy)
        report["policy_sha256"] = hashlib.sha256(args.policy.read_bytes()).hexdigest()
        report["policy_input"] = shape
        report["policy_output"] = out_shape

    print(json.dumps(report, indent=2))
    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
