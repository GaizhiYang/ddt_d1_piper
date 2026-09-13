#!/usr/bin/env python3
"""Export a policy-rate IsaacLab trace for D1 + Piper-L sim2sim calibration.

Run this script with the IsaacLab Python launcher (the same environment used
by ``loco-mani/scripts/rsl_rl/play.py``).  It records the actual observation
history delivered to the policy, rather than rebuilding it from a guessed
joint order.  The resulting CSV is intentionally compatible with
``compare_policy_traces.py`` and the MuJoCo ``--policy-trace-path`` output.

Example (from the IsaacLab repository):

    ./isaaclab.sh -p /home/hh/loco_mani_ws/src/ddt_controller/scripts/\
export_isaaclab_trace.py --checkpoint \
      /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/model_8000.pt \
      --output /tmp/isaac_policy_trace.csv --duration 10

The script does not send CAN frames and does not modify the training files.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="D1-PIPER-WBC-Play")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--initial-state-output", type=Path, default=None,
                    help="optional JSON path for the exact post-reset state used by replay")
parser.add_argument("--actions-output", type=Path, default=None,
                    help="optional .npy path for raw 22-D policy actions")
parser.add_argument("--commands-output", type=Path, default=None,
                    help="optional .npz path for policy-rate velocity/EE command replay")
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--disable-observation-corruption", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import LeggedManip_Lab.tasks  # noqa: F401,E402


JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint",
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
]
TERM_DIMS = (3, 3, 22, 22, 22, 3, 7)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _one(value: Any, width: int | None = None) -> np.ndarray:
    result = _numpy(value)
    if result.ndim >= 2:
        result = result[0]
    result = result.astype(np.float32, copy=False).reshape(-1)
    if width is not None and result.size != width:
        raise RuntimeError(f"unexpected field width {result.size}, expected {width}")
    return result.copy()


def _latest_frame(history: np.ndarray) -> np.ndarray:
    """Recover the current 82-D frame from IsaacLab term-major history."""
    history = np.asarray(history, dtype=np.float32).reshape(-1)
    if history.size != 246:
        raise RuntimeError(f"policy history has {history.size} values, expected 246")
    blocks: list[np.ndarray] = []
    offset = 0
    for width in TERM_DIMS:
        block = history[offset:offset + width * 3]
        blocks.append(block[2 * width:3 * width])
        offset += width * 3
    return np.concatenate(blocks).astype(np.float32)


def _robot_trace(env: Any, joint_ids: Any, ee_id: int, command_manager: Any) -> dict[str, np.ndarray]:
    robot = env.scene["robot"]
    data = robot.data
    q = _one(data.joint_pos[:, joint_ids], 22)
    dq = _one(data.joint_vel[:, joint_ids], 22)
    tau = _one(data.applied_torque[:, joint_ids], 22)
    root_pos = _one(data.root_pos_w, 3)
    root_quat = _one(data.root_quat_w, 4)
    root_w = _one(data.root_ang_vel_b, 3)
    gravity = _one(data.projected_gravity_b, 3)
    ee_pos = _one(data.body_pos_w[:, ee_id], 3)
    ee_quat = _one(data.body_quat_w[:, ee_id], 4)
    cmd_v = _one(command_manager.get_command("base_velocity"), 3)
    cmd_ee = _one(command_manager.get_command("ee_pose"), 7)
    return {
        "root_pos": root_pos, "root_quat": root_quat, "root_ang_vel": root_w,
        "projected_gravity": gravity, "ee_pos": ee_pos, "ee_quat": ee_quat,
        "command_velocity": cmd_v, "command_ee_pose": cmd_ee,
        "q": q, "dq": dq, "tau": tau,
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
         agent_cfg: RslRlBaseRunnerCfg) -> None:
    if not np.isfinite(args_cli.duration) or args_cli.duration <= 0:
        raise ValueError("--duration must be finite and positive")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    if args_cli.device:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_observation_corruption:
        env_cfg.observations.policy.enable_corruption = False
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, metadata.version("rsl-rl-lib")
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = OnPolicyRunner if agent_cfg.class_name == "OnPolicyRunner" else DistillationRunner
    runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args_cli.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
    ee_ids, _ = robot.find_bodies("end_effector")
    if len(ee_ids) != 1:
        raise RuntimeError(f"expected one end_effector body, got {ee_ids}")
    command_manager = unwrapped.command_manager
    obs = env.get_observations()
    # ``RslRlVecEnvWrapper`` may expose observations as a torch tensor or a
    # dict depending on the IsaacLab/RSL-RL release.  The policy group must
    # be the flattened 246-D history in either case.
    if isinstance(obs, dict):
        policy_obs = obs["policy"]
    else:
        policy_obs = obs
    duration_steps = int(args_cli.duration / float(unwrapped.step_dt))
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)

    # Save the state immediately after reset, before evaluating the first
    # policy action.  This removes ambiguity from reset randomization when a
    # trace is replayed in MuJoCo.  All vectors use policy joint order and the
    # free-base quaternion convention wxyz.
    if args_cli.initial_state_output is not None:
        root_pos = _one(robot.data.root_pos_w, 3)
        root_quat = _one(robot.data.root_quat_w, 4)
        root_lin_vel = _one(robot.data.root_lin_vel_w, 3)
        root_ang_vel = _one(robot.data.root_ang_vel_b, 3)
        q0 = _one(robot.data.joint_pos[:, joint_ids], 22)
        dq0 = _one(robot.data.joint_vel[:, joint_ids], 22)
        reset_payload = {
            "root_pos": root_pos.tolist(),
            "root_quat": root_quat.tolist(),
            "root_lin_vel": root_lin_vel.tolist(),
            "root_ang_vel": root_ang_vel.tolist(),
            "q": q0.tolist(),
            "dq": dq0.tolist(),
            "joint_names": JOINT_NAMES,
            "source": "IsaacLab post-reset, pre-policy-step",
            "task": args_cli.task,
            "seed": args_cli.seed,
        }
        args_cli.initial_state_output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.initial_state_output.write_text(
            json.dumps(reset_payload, indent=2) + "\n",
            encoding="utf-8",
        )

    header = ["time", "physics_step",
              "root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz",
              "root_wx", "root_wy", "root_wz", "gravity_x", "gravity_y", "gravity_z",
              "ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz",
              "cmd_vx", "cmd_vy", "cmd_wz", "cmd_ee_x", "cmd_ee_y", "cmd_ee_z",
              "cmd_ee_qw", "cmd_ee_qx", "cmd_ee_qy", "cmd_ee_qz",
              *[f"frame_{i}" for i in range(82)],
              *[f"history_{i}" for i in range(246)],
              *[f"action_{n}" for n in JOINT_NAMES], *[f"tau_{n}" for n in JOINT_NAMES],
              *[f"q_{n}" for n in JOINT_NAMES], *[f"dq_{n}" for n in JOINT_NAMES]]

    actions: list[np.ndarray] = []
    command_velocity: list[np.ndarray] = []
    command_ee_pose: list[np.ndarray] = []
    with args_cli.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for step in range(duration_steps):
            with torch.inference_mode():
                if isinstance(obs, dict):
                    policy_obs = obs["policy"]
                else:
                    policy_obs = obs
                history = _one(policy_obs, 246)
                frame = _latest_frame(history)
                action = _one(policy(policy_obs), 22)
            actions.append(action.copy())
            state = _robot_trace(unwrapped, joint_ids, int(ee_ids[0]), command_manager)
            command_velocity.append(state["command_velocity"].copy())
            command_ee_pose.append(state["command_ee_pose"].copy())
            # ``data.applied_torque`` sampled before env.step() is the torque
            # from the previous physics update in IsaacLab.  Advance once and
            # sample it afterwards so the trace's tau row corresponds to the
            # action recorded on this row.  q/dq and the rest of the state stay
            # at the pre-step policy sample instant, matching MuJoCo's trace.
            obs, _, dones, _ = env.step(torch.as_tensor(action[None, :], device=unwrapped.device))
            applied_tau = _one(robot.data.applied_torque[:, joint_ids], 22)
            writer.writerow([
                float(step * unwrapped.step_dt), step,
                *state["root_pos"].tolist(), *state["root_quat"].tolist(),
                *state["root_ang_vel"].tolist(), *state["projected_gravity"].tolist(),
                *state["ee_pos"].tolist(), *state["ee_quat"].tolist(),
                *state["command_velocity"].tolist(), *state["command_ee_pose"].tolist(),
                *frame.tolist(), *history.tolist(), *action.tolist(), *applied_tau.tolist(),
                *state["q"].tolist(), *state["dq"].tolist(),
            ])
            if bool(_numpy(dones).reshape(-1)[0]):
                break
            if hasattr(policy, "reset"):
                policy.reset(dones)
    if args_cli.actions_output is not None:
        args_cli.actions_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args_cli.actions_output, np.asarray(actions, dtype=np.float32))
    if args_cli.commands_output is not None:
        args_cli.commands_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args_cli.commands_output,
            command_velocity=np.asarray(command_velocity, dtype=np.float32),
            command_ee_pose=np.asarray(command_ee_pose, dtype=np.float32),
            rate_hz=np.asarray(1.0 / float(unwrapped.step_dt), dtype=np.float32),
        )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
