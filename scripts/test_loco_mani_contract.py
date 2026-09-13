#!/usr/bin/env python3
"""Self-contained contract tests for the D1+Piper-L MuJoCo adapter.

These tests do not open CAN devices and do not load the neural network.  They
lock down the parts of the deployment ABI that are easy to regress while
editing the model or controller: named 22-DoF lookup, 82-D frame composition,
term-major history, action scaling and the IsaacLab DelayBuffer semantics.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

import mujoco


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py"
HARDWARE = ROOT / "loco_mani_rl_controller/scripts/d1_piper_hardware.py"
COMMAND_FILE = ROOT / "loco_mani_rl_controller/scripts/command_file.py"
KEYBOARD = ROOT / "loco_mani_rl_controller/scripts/loco_mani_keyboard.py"
CONFIG = ROOT / "loco_mani_rl_controller/scripts/loco_mani_config.py"
RUNTIME = ROOT / "loco_mani_rl_controller/scripts/loco_mani_runtime.py"
D1_ADAPTER = ROOT / "loco_mani_rl_controller/scripts/d1_tita_adapter.py"
SCENE = ROOT / "loco_mani_description/mujoco/scene.xml"


def _load_module():
    spec = importlib.util.spec_from_file_location("d1_piper_l_mujoco_contract", CONTROLLER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CONTROLLER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = _load_module()


def _load_hardware_module():
    spec = importlib.util.spec_from_file_location("d1_piper_hardware_contract", HARDWARE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {HARDWARE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


H = _load_hardware_module()


def _load_command_file_module():
    spec = importlib.util.spec_from_file_location("command_file_contract", COMMAND_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {COMMAND_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CF = _load_command_file_module()


def _load_keyboard_module():
    spec = importlib.util.spec_from_file_location("loco_mani_keyboard_contract", KEYBOARD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {KEYBOARD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


K = _load_keyboard_module()


def _load_config_module():
    spec = importlib.util.spec_from_file_location("loco_mani_config_contract", CONFIG)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {CONFIG}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CFG = _load_config_module()


def _load_runtime_module():
    # Use the already imported contract modules so this remains a pure
    # offline test and does not need a ROS installation or any CAN driver.
    sys.modules.setdefault("d1_piper_hardware", H)
    sys.modules.setdefault("command_file", CF)
    spec = importlib.util.spec_from_file_location("loco_mani_runtime_contract", RUNTIME)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNTIME}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = _load_runtime_module()


def _load_d1_adapter_module():
    spec = importlib.util.spec_from_file_location("d1_tita_adapter_contract", D1_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {D1_ADAPTER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


D1A = _load_d1_adapter_module()


class LocoManiContractTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.rollout = M.D1PiperRollout(
            self.model,
            self.data,
            M.Policy(""),
            np.zeros(3, dtype=np.float32),
            np.asarray([0.425, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            initial_height=0.4567,
            piper_delay_steps=4,
        )

    def test_strict_boolean_parser_is_fail_closed(self):
        self.assertFalse(CFG.parse_bool("false", True, name="test"))
        self.assertTrue(CFG.parse_bool("true", False, name="test"))
        self.assertFalse(CFG.parse_bool("0", True, name="test"))
        self.assertTrue(CFG.parse_bool("yes", False, name="test"))
        with self.assertRaises(ValueError):
            CFG.parse_bool("perhaps", False, name="test")
        with self.assertRaises(ValueError):
            CFG.parse_bool(2, False, name="test")

    def test_string_false_safety_values_do_not_enable_live_path(self):
        payload = CFG.load_yaml(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        payload["safety"]["dry_run"] = "false"
        payload["safety"]["enable_on_start"] = "false"
        payload["safety"]["base_height_check_enabled"] = "false"
        payload["startup"]["enabled"] = "false"
        self.assertFalse(CFG.parse_bool(payload["safety"]["dry_run"], True))
        self.assertFalse(CFG.parse_bool(payload["safety"]["enable_on_start"], True))
        self.assertFalse(CFG.parse_bool(payload["safety"]["base_height_check_enabled"], True))
        self.assertFalse(CFG.parse_bool(payload["startup"]["enabled"], True))

    def test_model_abi(self):
        self.assertEqual(self.model.nq, 32)
        self.assertEqual(self.model.nv, 31)
        self.assertEqual(self.model.nu, 22)
        self.rollout.validate_model(self.model)
        self.assertEqual(
            [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                       for n in M.JOINT_NAMES]],
            M.JOINT_NAMES,
        )

    def test_ros2_control_is_split_from_plain_urdf(self):
        """The model URDF stays plugin-free; xacro owns ros2_control."""
        import xml.etree.ElementTree as ET

        plain = ET.parse(ROOT / "loco_mani_description/urdf/robot.urdf").getroot()
        self.assertIsNone(plain.find("ros2_control"))
        wrapper = (ROOT / "loco_mani_description/xacro/robot.xacro").read_text(
            encoding="utf-8"
        )
        control = (ROOT / "loco_mani_description/xacro/ros2control.xacro").read_text(
            encoding="utf-8"
        )
        self.assertIn("xacro:include", wrapper)
        self.assertIn("ros2control.xacro", wrapper)
        self.assertIn("xacro:loco_mani_ros2_control", wrapper)
        self.assertIn('<xacro:macro name="loco_mani_ros2_control"', control)
        self.assertEqual(control.count('<joint name="'), len(M.JOINT_NAMES))

        # Expand the wrapper once with an explicit plugin/name to verify that
        # launch-time substitutions reach the emitted ros2_control element.
        import xacro

        expanded = xacro.process_file(
            str(ROOT / "loco_mani_description/xacro/robot.xacro"),
            mappings={"hardware_plugin": "example/Hardware", "ros2_control_name": "hw"},
        )
        # xacro returns an xml.dom.minidom Document; convert through ElementTree
        # so the assertions above and below use the same API.
        expanded = ET.fromstring(expanded.toxml())
        ros2_control = expanded.find("ros2_control")
        self.assertIsNotNone(ros2_control)
        self.assertEqual(ros2_control.get("name"), "hw")
        self.assertEqual(ros2_control.find("./hardware/plugin").text, "example/Hardware")

    def test_mujoco_sensor_vectors_use_filtered_indices(self):
        """Mixed IMU/F-T sensor lists must not index filtered vectors globally."""
        source = (ROOT / "simulation/mujoco_bridge/mujoco_ros2_control/src/mujoco_system.cpp").read_text(
            encoding="utf-8"
        )
        self.assertGreaterEqual(source.count("ft_sensor_data_.back()"), 1)
        self.assertGreaterEqual(source.count("imu_sensor_data_.back()"), 1)
        self.assertNotIn("ft_sensor_data_.at(sensor_index)", source)
        self.assertNotIn("imu_sensor_data_.at(sensor_index)", source)

    def test_live_command_file_reader(self):
        import json
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "command.json"
            path.write_text(json.dumps({
                "timestamp_unix": time.time(),
                "command_velocity": [0.2, -0.1, 0.3],
                "ee_pose": [0.4, 0.01, 0.5, 2.0, 0.0, 0.0, 0.0],
            }), encoding="utf-8")
            reader = CF.CommandFileReader(path, timeout_s=0.5)
            command = reader.read()
            self.assertIsNotNone(command)
            velocity, pose = command
            self.assertTrue(np.allclose(velocity, [0.2, -0.1, 0.3]))
            self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0)

            # A dead keyboard must stop locomotion after the heartbeat
            # timeout while preserving the last valid end-effector target.
            path.write_text(json.dumps({
                "timestamp_unix": time.time() - 2.0,
                "command_velocity": [0.8, 0.0, 0.0],
                "ee_pose": [0.4, 0.01, 0.5, 1.0, 0.0, 0.0, 0.0],
            }), encoding="utf-8")
            time.sleep(0.001)
            velocity, pose = reader.read()
            self.assertTrue(np.array_equal(velocity, np.zeros(3, dtype=np.float32)))
            self.assertTrue(np.allclose(pose[:3], [0.4, 0.01, 0.5]))

    def test_keyboard_orientation_command(self):
        """Keyboard RPY controls emit normalized scalar-first quaternions."""
        keyboard = K.LocoManiKeyboard.__new__(K.LocoManiKeyboard)
        keyboard._closed = False
        keyboard.velocity = np.zeros(3, dtype=np.float64)
        keyboard.ee_pose = K.DEFAULT_EE.copy()
        keyboard.ee_rpy = np.zeros(3, dtype=np.float64)
        keyboard.ee_step = 0.01
        keyboard.orientation_step = 0.05
        keyboard.orientation_limit = 0.6
        keyboard.velocity_step = 0.1
        keyboard.yaw_step = 0.15
        keyboard._publish = lambda: None
        keyboard._print_state = lambda: None

        self.assertTrue(np.allclose(keyboard.ee_pose[3:], [1.0, 0.0, 0.0, 0.0]))
        keyboard._handle_key("t")
        self.assertAlmostEqual(float(keyboard.ee_rpy[0]), 0.05)
        self.assertAlmostEqual(float(np.linalg.norm(keyboard.ee_pose[3:])), 1.0)
        self.assertGreater(float(keyboard.ee_pose[4]), 0.0)  # positive roll

        for _ in range(100):
            keyboard._handle_key("f")
        self.assertLessEqual(float(keyboard.ee_rpy[1]), 0.6)
        self.assertAlmostEqual(float(np.linalg.norm(keyboard.ee_pose[3:])), 1.0)

        keyboard._handle_key("0")
        self.assertTrue(np.allclose(keyboard.ee_rpy, np.zeros(3)))
        self.assertTrue(np.allclose(keyboard.ee_pose, K.DEFAULT_EE))

    def test_yaml_resolves_full_policy_configuration(self):
        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        self.assertEqual(config.frame_dim, 82)
        self.assertEqual(config.action_dim, 22)
        self.assertEqual(config.history_length, 3)
        self.assertEqual(config.history_layout, "term-major")
        self.assertEqual(config.action_scale.shape, (22,))
        self.assertEqual(config.kp.shape, (22,))
        self.assertEqual(config.kd.shape, (22,))
        self.assertEqual(config.torque_limit.shape, (22,))
        self.assertEqual(config.velocity_limits.shape, (22,))
        self.assertEqual(config.wheel_indices.tolist(), [3, 7, 11, 15])
        self.assertTrue(np.allclose(config.ee_pose[3:], [1, 0, 0, 0]))

    def test_live_command_limits_are_shared_with_hardware_runtime(self):
        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        velocity, pose = CFG.sanitize_commands(
            [99.0, -99.0, 99.0],
            [99.0, -99.0, -99.0, 0.0, 1.0, 0.0, 0.0],
            config,
        )
        self.assertTrue(np.allclose(velocity, [1.0, -1.0, 1.5]))
        self.assertTrue(np.allclose(pose[:3], [0.75, -0.45, 0.20]))
        self.assertAlmostEqual(float(np.linalg.norm(pose[3:])), 1.0, places=6)
        # A 90-degree pitch is clipped to the configured orientation range;
        # this test also exercises quaternion -> RPY -> quaternion conversion.
        _, clipped = CFG.sanitize_commands(
            [0.0, 0.0, 0.0], [0.4, 0.0, 0.5, 0.7071068, 0.0, 0.7071068, 0.0], config)
        self.assertLessEqual(abs(float(clipped[4])), 0.6)
        self.assertAlmostEqual(float(np.linalg.norm(clipped[3:])), 1.0, places=6)

    def test_piper_urdf_rpy_matches_mujoco_body_quaternions(self):
        """Guard the URDF fixed-axis-RPY -> MuJoCo quaternion conversion."""
        import math
        import xml.etree.ElementTree as ET

        urdf = ET.parse(ROOT / "loco_mani_description/urdf/robot.urdf").getroot()
        expected = {
            "link2": "1.5707963 -0.1357866 -3.1415926",
            "link3": "0 0 -1.7938494",
            "link4": "1.5707963 0 0",
            "link5": "-1.5707963 0 0",
            "link6": "1.5707963 0 0",
        }
        for name, rpy in expected.items():
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            # Find the joint whose child link is the body under test.
            urdf_joint = next(j for j in urdf.findall("joint")
                              if j.find("child") is not None and j.find("child").get("link") == name)
            origin = urdf_joint.find("origin")
            self.assertEqual(origin.get("rpy"), rpy)
            roll, pitch, yaw = (float(x) for x in rpy.split())
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            expected_q = np.asarray([
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ])
            # ``model.body_quat`` is the local body transform.  ``data.xquat``
            # is in world coordinates and includes all preceding Piper joint
            # transforms, so it is not the right quantity for this check.
            self.assertTrue(np.allclose(self.model.body_quat[body_id], expected_q, atol=1e-6)
                            or np.allclose(self.model.body_quat[body_id], -expected_q, atol=1e-6))

    def test_frame_and_term_major_history(self):
        frame = self.rollout.frame()
        self.assertEqual(frame.shape, (82,))
        self.rollout.policy_step(frame)
        self.assertEqual(self.rollout.history.shape, (246,))
        # Each observation term is buffered independently.  The first append
        # fills all three slots with the same sample, then terms are flattened
        # one after another (not three complete frames interleaved).
        history_offset = 0
        for term_slice in self.rollout._term_slices:
            term = frame[term_slice]
            width = term_slice.stop - term_slice.start
            self.assertTrue(np.array_equal(
                self.rollout.history[history_offset:history_offset + width * 3],
                np.tile(term, 3),
            ))
            history_offset += width * 3

    def test_delay_buffer_transition(self):
        self.rollout.policy_step(replay_action=np.zeros(22, dtype=np.float32))
        self.rollout.advance_actuator_delay()
        self.assertTrue(np.array_equal(self.rollout._arm_applied_action, np.zeros(6)))
        command = np.ones(22, dtype=np.float32)
        self.rollout.target_action = command.copy()
        # max_delay=4 means the new command is observed after four stale
        # samples have been consumed; the fifth append returns it.
        for _ in range(4):
            self.rollout.advance_actuator_delay()
            self.assertTrue(np.array_equal(self.rollout._arm_applied_action, np.zeros(6)))
        self.rollout.advance_actuator_delay()
        self.assertTrue(np.array_equal(self.rollout._arm_applied_action, np.ones(6)))

    def test_wheel_position_observation_is_masked(self):
        # Deliberately move a wheel while leaving all other active joints at
        # their defaults.  The policy position term must remain exactly zero
        # for all four wheels.
        for index in M.WHEEL_INDICES:
            self.data.qpos[self.rollout.qadr[index]] = 1.234
        mujoco.mj_forward(self.model, self.data)
        frame = self.rollout.frame()
        self.assertTrue(np.array_equal(frame[6:28][M.WHEEL_INDICES], np.zeros(4)))

    def test_dc_motor_speed_envelope_matches_corner_speed_semantics(self):
        # IsaacLab clips velocity to the corner speed (v_lim * (1 + e/s)) and
        # only clips the upper/lower effort bounds to +/- effort_limit.  Above
        # the no-load speed the positive bound is therefore allowed to become
        # negative; this is intentional four-quadrant motor behavior.
        hip = 0
        self.rollout.target_action.fill(0.0)
        # A large enough position error keeps the raw PD effort positive even
        # after the damping term at this test velocity is subtracted.
        self.rollout.target_action[hip] = 10.0
        self.data.qpos[self.rollout.qadr[hip]] = M.DEFAULT_Q[hip]
        self.data.qvel[self.rollout.vadr[hip]] = 25.0  # > 20, below 40 corner speed
        mujoco.mj_forward(self.model, self.data)
        self.rollout.update_torque()
        self.assertAlmostEqual(float(self.rollout.tau[hip]), -22.5, places=5)

        # At the corner speed the positive envelope reaches -saturation_effort
        # while the negative bound remains at -effort_limit.
        self.data.qvel[self.rollout.vadr[hip]] = 40.0
        mujoco.mj_forward(self.model, self.data)
        self.rollout.update_torque()
        self.assertAlmostEqual(float(self.rollout.tau[hip]), -90.0, places=5)

    def test_hardware_backends_are_dry_run_and_gated(self):
        d1 = H.D1Backend(dry_run=True)
        piper = H.PiperBackend(dry_run=True)
        d1.connect()
        piper.connect()
        self.assertTrue(d1.read().valid)
        self.assertTrue(piper.read().valid)
        self.assertTrue(d1.send(H.JointCommand()))
        self.assertTrue(piper.send(H.JointCommand()))
        # A non-dry backend cannot even connect without an explicit gate or a
        # vendor adapter.  This test never opens can0/can1.
        with self.assertRaises(PermissionError):
            H.PiperBackend(dry_run=False).connect()

    def test_read_only_backend_rejects_every_write_path(self):
        """Inspection mode may read feedback but must never emit a command."""
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22),
            direction=np.ones(22, np.float32),
            position_offset_rad=np.zeros(22, np.float32),
        )

        class D1Api:
            def get_motors_in(self):
                return [{"position": 0.0, "velocity": 0.0, "torque": 0.0}] * 16
            def get_motors_status(self):
                return [0] * 16
            def get_imu_data(self):
                return {"quaternion": [0.0, 0.0, 0.0, 1.0],
                        "gyro": [0.0] * 3, "accl": [0.0, 0.0, 9.81]}
            def send_motors_can(self, _records):
                raise AssertionError("read-only inspection attempted to send")

        d1 = H.D1Backend(
            api_factory=D1Api, motor_out_factory=lambda **kwargs: kwargs,
            dry_run=False, calibration=calibration,
        )
        d1.connect(hardware_gate=True, read_only=True)
        self.assertTrue(d1.read().valid)
        with self.assertRaises(PermissionError):
            d1.send(H.JointCommand(), hardware_gate=True)
        with self.assertRaises(PermissionError):
            d1.emergency_stop(hardware_gate=True)
        d1.disconnect()

        piper = H.PiperBackend(dry_run=False, calibration=calibration)
        piper.connected = True
        piper.read_only = True
        piper.arm = object()
        with self.assertRaises(PermissionError):
            piper.enable(hardware_gate=True)
        with self.assertRaises(PermissionError):
            piper.send(H.JointCommand(), hardware_gate=True)

    def test_piper_real_mit_requires_explicit_enable(self):
        """Connecting a real Piper must not silently make MIT sends legal."""
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22),
            direction=np.ones(22, np.float32),
            position_offset_rad=np.zeros(22, np.float32),
        )
        backend = H.PiperBackend(dry_run=False, calibration=calibration)
        backend.connected = True
        backend.arm = object()
        with self.assertRaises(RuntimeError):
            backend.send(H.JointCommand(), hardware_gate=True)

    def test_piper_policy_limits_are_checked_before_calibrated_bus_offset(self):
        """A valid policy target must survive a non-zero encoder offset."""
        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22), direction=np.ones(22, np.float32),
            position_offset_rad=np.r_[np.zeros(16, np.float32), np.full(6, 1.0, np.float32)],
        )
        backend = H.PiperBackend(
            dry_run=False, firmware_profile="v188", calibration=calibration, config=config)
        class ArmStub:
            def set_motion_mode(self, _mode):
                pass
            def move_mit(self, **_kwargs):
                pass
        backend.arm, backend.connected, backend.enabled = ArmStub(), True, True
        backend._motion_mode_set = False
        command = H.JointCommand(
            position=config.default_q.copy(), velocity=np.zeros(22, np.float32),
            kp=config.kp.copy(), kd=config.kd.copy(), torque=np.zeros(22, np.float32))
        self.assertTrue(backend.send(command, hardware_gate=True))

    def test_piper_mit_feedback_code_is_profile_specific(self):
        """v188 must not accept the legacy 0x04 MIT feedback code."""
        class JointMessage:
            joint_1, joint_2, joint_3 = 1.0, 2.0, 3.0
            joint_4, joint_5, joint_6 = 4.0, 5.0, 6.0
        class MotorMessage:
            def __init__(self, index):
                self.position, self.velocity, self.torque = index * 0.1, 0.0, 0.0
        class Status:
            arm_status = 0
            err_status = object()
            err_code = 0
            ctrl_mode = 1
            mode_feedback = 0x04
        class ArmStub:
            def get_joint_angles(self):
                return type("F", (), {"msg": JointMessage(), "timestamp": 1.0, "hz": 100.0})()
            def get_motor_states(self, index):
                return type("F", (), {"msg": MotorMessage(index), "hz": 100.0, "timestamp": float(index)})()
            def get_arm_status(self):
                return type("F", (), {"msg": Status(), "hz": 100.0, "timestamp": 1.0})()
        backend = H.PiperBackend(dry_run=False, firmware_profile="v188")
        backend.arm, backend.connected, backend.enabled = ArmStub(), True, True
        backend._motion_mode_set = True
        snapshot = backend.read()
        self.assertFalse(snapshot.valid)
        self.assertIn("does not match", snapshot.error)

    def test_piper_mit_rejects_sdk_clamping_and_disables_auto_mode_switch(self):
        """The backend must validate MIT fields before any motion frame."""
        class ArmStub:
            def __init__(self):
                self.calls = []
                self.auto_mode = True

            def set_auto_set_motion_mode_enabled(self, enabled):
                self.calls.append(("auto_mode", enabled))
                self.auto_mode = enabled

            def set_motion_mode(self, mode):
                self.calls.append(("mode", mode))

            def move_mit(self, **kwargs):
                self.calls.append(("move", kwargs))

        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22), direction=np.ones(22, np.float32),
            position_offset_rad=np.zeros(22, np.float32),
        )
        backend = H.PiperBackend(
            dry_run=False, firmware_profile="v188", calibration=calibration, config=config)
        arm = ArmStub()
        backend.arm, backend.connected, backend.enabled = arm, True, True
        backend._auto_motion_mode_disabled = False
        command = H.JointCommand(
            position=config.default_q.copy(), velocity=np.zeros(22, np.float32),
            kp=config.kp.copy(), kd=config.kd.copy(), torque=np.zeros(22, np.float32))
        self.assertTrue(backend.send(command, hardware_gate=True))
        self.assertEqual(arm.calls[0], ("mode", "mit"))
        self.assertEqual(sum(call[0] == "move" for call in arm.calls), 6)

        bad = H.JointCommand(
            position=config.default_q.copy(), velocity=np.zeros(22, np.float32),
            kp=config.kp.copy(), kd=config.kd.copy(), torque=np.zeros(22, np.float32))
        bad.position[16] = 99.0
        with self.assertRaises(ValueError):
            backend.send(bad, hardware_gate=True)
        self.assertEqual(sum(call[0] == "move" for call in arm.calls), 6)

    def test_piper_v183_mit_uses_uniform_eight_nm_limit(self):
        """v183 is +/-8 N*m on every joint, unlike DEFAULT's 8*b table."""
        class ArmStub:
            def __init__(self):
                self.calls = []

            def set_motion_mode(self, mode):
                self.calls.append(("mode", mode))

            def move_mit(self, **kwargs):
                self.calls.append(("move", kwargs))

        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22), direction=np.ones(22, np.float32),
            position_offset_rad=np.zeros(22, np.float32),
        )
        backend = H.PiperBackend(
            dry_run=False, firmware_profile="v183", calibration=calibration, config=config)
        arm = ArmStub()
        backend.arm, backend.connected, backend.enabled = arm, True, True
        command = H.JointCommand(
            position=config.default_q.copy(), velocity=np.zeros(22, np.float32),
            kp=config.kp.copy(), kd=config.kd.copy(), torque=np.zeros(22, np.float32))
        command.torque[16 + 1] = 8.0
        self.assertTrue(backend.send(command, hardware_gate=True))
        self.assertEqual(sum(call[0] == "move" for call in arm.calls), 6)
        command.torque[16 + 1] = 8.01
        with self.assertRaises(ValueError):
            backend.send(command, hardware_gate=True)
        self.assertEqual(sum(call[0] == "move" for call in arm.calls), 6)

    def test_startup_manager_ramps_then_releases_policy(self):
        """The optional preparation stage holds wheels and releases RL only after hold."""
        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        startup = R.StartupManager(
            {"startup": {"enabled": True, "mode": "ramp_default", "ramp_s": 1.0, "hold_s": 0.1}},
            config,
        )
        snapshot = H.StateSnapshot(
            q=np.zeros(22, np.float32),
            dq=np.zeros(22, np.float32),
            tau=np.zeros(22, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            valid=True,
            monotonic_ns=__import__("time").monotonic_ns(),
        )
        initial = startup.update(snapshot, 10.0)
        self.assertIsNotNone(initial)
        self.assertTrue(np.array_equal(initial.position[H.WHEEL_INDICES], np.zeros(4)))
        halfway = startup.update(snapshot, 10.5)
        self.assertAlmostEqual(float(halfway.position[1]), float(config.default_q[1] * 0.5))
        complete = startup.update(snapshot, 11.0)
        self.assertIsNotNone(complete)
        self.assertEqual(startup.phase, "hold")
        # The first tick after hold duration switches phase; the following
        # one returns None, authorizing policy inference.
        self.assertIsNotNone(startup.update(snapshot, 11.2))
        self.assertIsNone(startup.update(snapshot, 11.21))
        self.assertEqual(startup.phase, "policy")

    def test_runtime_start_failure_disconnects_first_backend(self):
        """A failure opening can1 must close a previously opened can0 backend."""
        class First:
            def __init__(self):
                self.disconnected = False
            def connect(self, **_kwargs):
                return None
            def disconnect(self):
                self.disconnected = True
        class Second:
            def connect(self, **_kwargs):
                raise RuntimeError("can1 unavailable")
            def disconnect(self):
                return None
        first, second = First(), Second()
        loop = R.HardwarePolicyRuntimeLoop(
            first, second,
            H.HardwarePolicyRuntime(lambda _frame, _history: np.zeros(22, np.float32)),
            H.SafetySupervisor(),
        )
        with self.assertRaisesRegex(RuntimeError, "can1 unavailable"):
            loop.start()
        self.assertTrue(first.disconnected)

    def test_d1_backend_supports_all_motor_api_variant(self):
        """The current released libtita_robot exports send_motors_can()."""
        class Api:
            def __init__(self):
                self.sent = None

            def send_motors_can(self, records):
                self.sent = list(records)
                return True

        api = Api()
        backend = H.D1Backend(
            api_factory=lambda: api,
            motor_out_factory=lambda **kwargs: kwargs,
            dry_run=False,
            calibration=H.JointCalibration(
                bus_index=np.r_[np.arange(16), np.arange(6)],
                can_id=np.arange(22),
                direction=np.ones(22, np.float32),
                position_offset_rad=np.zeros(22, np.float32),
            ),
        )
        backend.connect(hardware_gate=True)
        self.assertTrue(backend.send(H.JointCommand(), hardware_gate=True))
        self.assertIsNotNone(api.sent)
        self.assertEqual(len(api.sent), 16)

    def test_d1_force_direct_is_explicitly_gated(self):
        class Api:
            def __init__(self):
                self.requested = False

            def set_force_direct(self):
                self.requested = True
                return True

        api = Api()
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22), direction=np.ones(22, np.float32),
            position_offset_rad=np.zeros(22, np.float32),
        )
        backend = H.D1Backend(
            api_factory=lambda: api, motor_out_factory=lambda **kwargs: kwargs,
            dry_run=False, calibration=calibration)
        backend.connect(hardware_gate=True)
        with self.assertRaises(PermissionError):
            backend.set_force_direct()
        self.assertFalse(api.requested)
        self.assertTrue(backend.set_force_direct(hardware_gate=True))
        self.assertTrue(api.requested)

    def test_d1_ctypes_feedback_record_is_not_a_command_record(self):
        """The flat adapter must expose a three-field read-only record.

        A previous implementation accidentally constructed the six-field
        command ``MotorRecord`` from feedback and therefore failed on the
        first real D1 read.  Keep this ABI boundary explicit and testable
        without constructing the adapter (which would open can0).
        """
        feedback = D1A.FeedbackMotorRecord(1.0, 2.0, 3.0)
        self.assertEqual((feedback.position, feedback.velocity, feedback.torque),
                         (1.0, 2.0, 3.0))
        command = D1A.make_motor_out(
            timestamp=7, position=1.0, velocity=2.0, kp=3.0, kd=4.0, torque=5.0)
        self.assertEqual(command.timestamp, 7)
        with self.assertRaises(TypeError):
            D1A.MotorRecord(1.0, 2.0, 3.0)

    def test_d1_ctypes_struct_layout_matches_vendor_header(self):
        """The flat adapter buffers cover every C feedback field."""
        import ctypes

        self.assertEqual(ctypes.sizeof(D1A._Motor), 16)
        # This is the repository's flat-C output struct, not the larger
        # vendor api_imu_data_t.  The adapter intentionally omits vendor
        # temperature because the flat ABI does not expose it.
        self.assertEqual(ctypes.sizeof(D1A._Imu), 44)

    def test_d1_backend_supports_per_leg_api_variant(self):
        class Api:
            def __init__(self):
                self.sent = []

            def send_leg_motors_can(self, records, leg_index):
                self.sent.append((leg_index, list(records)))
                return True

        api = Api()
        backend = H.D1Backend(
            api_factory=lambda: api,
            motor_out_factory=lambda **kwargs: kwargs,
            dry_run=False,
            calibration=H.JointCalibration(
                bus_index=np.r_[np.arange(16), np.arange(6)],
                can_id=np.arange(22),
                direction=np.ones(22, np.float32),
                position_offset_rad=np.zeros(22, np.float32),
            ),
        )
        backend.connect(hardware_gate=True)
        self.assertTrue(backend.send(H.JointCommand(), hardware_gate=True))
        self.assertEqual([index for index, _ in api.sent], [0, 1, 2, 3])
        self.assertTrue(all(len(records) == 4 for _, records in api.sent))

    def test_nonidentity_calibration_reorders_gains_with_commands(self):
        """Bus permutation must apply equally to p/v/tau and kp/kd."""
        class Api:
            def __init__(self):
                self.sent = None

            def send_motors_can(self, records):
                self.sent = list(records)
                return True

        api = Api()
        # Swap the first two D1 joints; keep Piper identity.  A policy-order
        # gain of 11/22 must follow the corresponding command to bus slots
        # 1/0, rather than remaining attached to the array index.
        bus_index = np.r_[np.asarray([1, 0] + list(range(2, 16))), np.arange(6)]
        calibration = H.JointCalibration(
            bus_index=bus_index,
            can_id=np.arange(22),
            direction=np.ones(22, np.float32),
            position_offset_rad=np.zeros(22, np.float32),
        )
        backend = H.D1Backend(
            api_factory=lambda: api,
            motor_out_factory=lambda **kwargs: kwargs,
            dry_run=False,
            calibration=calibration,
        )
        backend.connect(hardware_gate=True)
        command = H.JointCommand(
            position=np.arange(22, dtype=np.float32),
            velocity=np.zeros(22, np.float32),
            kp=np.arange(22, dtype=np.float32) + 10.0,
            kd=np.arange(22, dtype=np.float32) + 20.0,
            torque=np.zeros(22, np.float32),
        )
        self.assertTrue(backend.send(command, hardware_gate=True))
        self.assertEqual(api.sent[0]["position"], 1.0)
        self.assertEqual(api.sent[0]["kp"], 11.0)
        self.assertEqual(api.sent[0]["kd"], 21.0)
        self.assertEqual(api.sent[1]["position"], 0.0)
        self.assertEqual(api.sent[1]["kp"], 10.0)
        self.assertEqual(api.sent[1]["kd"], 20.0)

    def test_hardware_observation_and_action_adapter(self):
        snapshot = H.StateSnapshot(
            q=np.asarray(M.DEFAULT_Q, np.float32),
            dq=np.zeros(22, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            gyro=np.zeros(3, np.float32), valid=True,
            monotonic_ns=__import__("time").monotonic_ns(),
        )
        adapter = H.ObservationAdapter()
        frame = adapter.frame(snapshot, [0, 0, 0], [0.425, 0, 0.5, 1, 0, 0, 0])
        self.assertEqual(frame.shape, (82,))
        history = adapter.append(frame)
        self.assertEqual(history.shape, (246,))
        action = np.ones(22, np.float32)
        command = H.ActionAdapter.to_command(action, snapshot)
        self.assertTrue(np.array_equal(command.velocity[H.WHEEL_INDICES], np.full(4, 5.0)))
        self.assertTrue(np.array_equal(command.position[H.WHEEL_INDICES], snapshot.q[H.WHEEL_INDICES]))
        self.assertTrue(np.allclose(command.position[16:], 0.25))

    def test_piper_structured_joint_feedback_is_decoded(self):
        # pyAgxArm's real get_joint_angles() returns a MessageAbstract whose
        # ``msg`` has joint_1..joint_6 attributes rather than an iterable list.
        class JointMessage:
            joint_1, joint_2, joint_3 = 1.0, 2.0, 3.0
            joint_4, joint_5, joint_6 = 4.0, 5.0, 6.0

        class MotorMessage:
            def __init__(self, index):
                self.position = 0.1 * index
                self.velocity = 0.2 * index
                self.torque = 0.3 * index

        class ArmStub:
            def get_joint_angles(self):
                return type("Feedback", (), {"msg": JointMessage(), "timestamp": 1.0, "hz": 100.0})()

            def get_motor_states(self, index):
                return type("Feedback", (), {"msg": MotorMessage(index), "hz": 100.0})()

            def get_arm_status(self):
                return type("Feedback", (), {"msg": type("Status", (), {
                    "arm_status": 0, "err_status": object(), "err_code": 0,
                    "ctrl_mode": 0, "mode_feedback": 0})(), "hz": 100.0})()

        backend = H.PiperBackend(dry_run=False)
        backend.arm = ArmStub()
        backend.connected = True
        snapshot = backend.read()
        self.assertTrue(snapshot.valid)
        self.assertTrue(np.array_equal(snapshot.q[16:], np.arange(1.0, 7.0)))

    def test_piper_status_angle_limit_and_error_code_are_rejected(self):
        """All documented Piper status faults invalidate the policy state."""
        class JointMessage:
            joint_1, joint_2, joint_3 = 1.0, 2.0, 3.0
            joint_4, joint_5, joint_6 = 4.0, 5.0, 6.0

        class MotorMessage:
            def __init__(self, index):
                self.position = 0.1 * index
                self.velocity = 0.2 * index
                self.torque = 0.3 * index

        class ErrorStatus:
            communication_status_joint_1 = False
            communication_status_joint_2 = False
            communication_status_joint_3 = False
            communication_status_joint_4 = False
            communication_status_joint_5 = False
            communication_status_joint_6 = False
            joint_1_angle_limit = False
            joint_2_angle_limit = True
            joint_3_angle_limit = False
            joint_4_angle_limit = False
            joint_5_angle_limit = False
            joint_6_angle_limit = False

        class StatusMessage:
            arm_status = 0
            err_status = ErrorStatus()
            err_code = 1 << 9

        class ArmStub:
            def get_joint_angles(self):
                return type("Feedback", (), {"msg": JointMessage(), "timestamp": 1.0, "hz": 100.0})()

            def get_motor_states(self, index):
                return type("Feedback", (), {"msg": MotorMessage(index), "hz": 100.0})()

            def is_ok(self):
                return True

            def get_arm_status(self):
                return type("Feedback", (), {"msg": StatusMessage(), "hz": 100.0})()

        backend = H.PiperBackend(dry_run=False)
        backend.arm = ArmStub()
        backend.connected = True
        snapshot = backend.read()
        self.assertFalse(snapshot.valid)
        self.assertIn("angle-limit", snapshot.error)

        # A non-zero raw error bit must remain a fault even if the decoded
        # fields happen to be absent in a future SDK wrapper.
        StatusMessage.err_status = object()
        StatusMessage.err_code = 1 << 14
        snapshot = backend.read()
        self.assertFalse(snapshot.valid)
        self.assertIn("error code", snapshot.error)

    def test_piper_cached_feedback_timestamp_expires(self):
        """A positive SDK FPS must not hide a stalled cached CAN sample."""
        import time

        class JointMessage:
            joint_1, joint_2, joint_3 = 0.0, 0.0, 0.0
            joint_4, joint_5, joint_6 = 0.0, 0.0, 0.0

        class MotorMessage:
            position = velocity = torque = 0.0

        class ArmStub:
            def get_joint_angles(self):
                return type("Feedback", (), {
                    "msg": JointMessage(), "timestamp": 1.0, "hz": 100.0})()

            def get_motor_states(self, _index):
                return type("Feedback", (), {
                    "msg": MotorMessage(), "timestamp": 2.0, "hz": 100.0})()

            def get_arm_status(self):
                return type("Feedback", (), {"msg": type("Status", (), {
                    "arm_status": 0, "err_status": object(), "err_code": 0,
                    "ctrl_mode": 0, "mode_feedback": 0})(), "hz": 100.0})()

        backend = H.PiperBackend(dry_run=False, feedback_timeout_s=0.001)
        backend.arm = ArmStub()
        backend.connected = True
        self.assertTrue(backend.read().valid)
        time.sleep(0.003)
        stale = backend.read()
        self.assertFalse(stale.valid)
        self.assertIn("timestamp is stale", stale.error)

    def test_d1_cached_vendor_timestamp_is_not_treated_as_fresh(self):
        """A repeated D1 vendor frame must eventually trip the freshness gate."""
        class Api:
            def get_motors_in(self):
                return [{"position": 0.0, "velocity": 0.0, "torque": 0.0}] * 16

            def get_motors_status(self):
                return [0] * 16

            def get_imu_data(self):
                return {"timestamp": 1, "quaternion": [0.0, 0.0, 0.0, 1.0],
                        "gyro": [0.0] * 3, "accl": [0.0, 0.0, 9.81]}

        backend = H.D1Backend(api_factory=Api, dry_run=False,
                              feedback_timeout_s=0.001)
        backend.connect(hardware_gate=True, read_only=True)
        self.assertTrue(backend.read().valid)
        __import__("time").sleep(0.005)
        stale = backend.read()
        self.assertFalse(stale.valid)
        self.assertIn("stale", stale.error)
        backend.disconnect()

    def test_safety_accepts_explicit_infinite_velocity_envelope(self):
        """Unbounded velocity is allowed; NaN and negative values are not."""
        safety = H.SafetySupervisor(velocity_limits=[np.inf] * 22)
        self.assertTrue(np.isinf(safety.velocity_limits).all())
        with self.assertRaises(ValueError):
            H.SafetySupervisor(velocity_limits=[np.nan] * 22)
        with self.assertRaises(ValueError):
            H.SafetySupervisor(velocity_limits=[-1.0] * 22)

    def test_piper_explicit_firmware_profile_does_not_probe(self):
        """An explicit SDK profile skips the firmware query path."""
        import types

        calls = []

        class Arm:
            def connect(self):
                calls.append("connect")

            def disconnect(self):
                calls.append("disconnect")

            def is_connected(self):
                return True

        class Factory:
            @staticmethod
            def create_arm(_config):
                calls.append("create")
                return Arm()

        fake_sdk = types.SimpleNamespace(
            create_agx_arm_config=lambda **kwargs: kwargs,
            AgxArmFactory=Factory,
            resolve_firmware_profile=lambda *_args: (_ for _ in ()).throw(
                AssertionError("explicit firmware profile must not resolve")),
        )
        old_sdk = sys.modules.get("pyAgxArm")
        sys.modules["pyAgxArm"] = fake_sdk
        try:
            calibration = H.JointCalibration(
                bus_index=np.r_[np.arange(16), np.arange(6)],
                can_id=np.arange(22),
                direction=np.ones(22, np.float32),
                position_offset_rad=np.zeros(22, np.float32),
            )
            backend = H.PiperBackend(
                dry_run=False, firmware_profile="v188", calibration=calibration)
            backend.connect(hardware_gate=True)
            self.assertTrue(backend.connected)
            self.assertEqual(backend.firmware_profile, "v188")
            self.assertEqual(calls, ["create", "connect"])
            backend.disconnect()
        finally:
            if old_sdk is None:
                sys.modules.pop("pyAgxArm", None)
            else:
                sys.modules["pyAgxArm"] = old_sdk

    def test_piper_auto_firmware_profile_probes_then_reopens_selected_driver(self):
        """Auto profile performs one read-only probe and then reconnects v188."""
        import types

        calls = []

        class Arm:
            def __init__(self, profile):
                self.profile = profile

            def connect(self):
                calls.append(("connect", self.profile))

            def disconnect(self):
                calls.append(("disconnect", self.profile))

            def get_firmware(self, **_kwargs):
                calls.append(("firmware", self.profile))
                return {"software_version": "S-V1.8-8"}

            def is_connected(self):
                return True

        class Factory:
            @staticmethod
            def create_arm(config):
                profile = config["firmeware_version"]
                calls.append(("create", profile))
                return Arm(profile)

        fake_sdk = types.SimpleNamespace(
            create_agx_arm_config=lambda **kwargs: kwargs,
            AgxArmFactory=Factory,
            resolve_firmware_profile=lambda _robot, version: "v188" if version == "S-V1.8-8" else "default",
        )
        old_sdk = sys.modules.get("pyAgxArm")
        sys.modules["pyAgxArm"] = fake_sdk
        try:
            calibration = H.JointCalibration(
                bus_index=np.r_[np.arange(16), np.arange(6)],
                can_id=np.arange(22), direction=np.ones(22, np.float32),
                position_offset_rad=np.zeros(22, np.float32),
            )
            backend = H.PiperBackend(
                dry_run=False, firmware_profile="auto", calibration=calibration)
            backend.connect(hardware_gate=True)
            self.assertEqual(backend.firmware_profile, "v188")
            self.assertEqual(
                calls,
                [("create", "default"), ("connect", "default"),
                 ("firmware", "default"), ("disconnect", "default"),
                 ("create", "v188"), ("connect", "v188")],
            )
            backend.disconnect()
        finally:
            if old_sdk is None:
                sys.modules.pop("pyAgxArm", None)
            else:
                sys.modules["pyAgxArm"] = old_sdk

    def test_mock_buses_run_full_policy_safety_and_send_chain(self):
        """Exercise both reader threads and the gated command path offline."""
        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        snapshot = H.StateSnapshot(
            q=np.asarray(config.default_q, np.float32),
            dq=np.zeros(22, np.float32), tau=np.zeros(22, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            gyro=np.zeros(3, np.float32), accel=np.asarray([0, 0, 9.81], np.float32),
            valid=True, monotonic_ns=__import__("time").monotonic_ns())

        class MockBus:
            def __init__(self, offset):
                self.offset = offset
                self.connected = False
                self.enabled = False
                self.sent = []
                self.estopped = False

            def connect(self, **_kwargs):
                self.connected = True

            def disconnect(self):
                self.connected = False

            def read(self):
                value = snapshot.copy()
                value.q[:16] = 0.0 if self.offset else snapshot.q[:16]
                if self.offset:
                    value.q[:16] = 0.0
                    value.q[16:] = snapshot.q[16:]
                else:
                    value.q[16:] = 0.0
                value.monotonic_ns = __import__("time").monotonic_ns()
                return value

            def send(self, command, **_kwargs):
                self.sent.append(command)
                return True

            def enable(self, **_kwargs):
                self.enabled = True

            def emergency_stop(self, **_kwargs):
                self.estopped = True
                self.enabled = False

        d1, piper = MockBus(False), MockBus(True)
        startup = R.StartupManager(
            {"startup": {"enabled": True, "mode": "ramp_default", "ramp_s": 0.0, "hold_s": 0.0}},
            config)
        policy_calls = []

        def policy(frame, history):
            policy_calls.append((frame.copy(), history.copy()))
            return np.zeros(22, np.float32)

        loop = R.HardwarePolicyRuntimeLoop(
            d1, piper, H.HardwarePolicyRuntime(policy, config=config), H.SafetySupervisor(
                state_timeout_ms=100.0, command_timeout_ms=100.0,
                torque_limits=config.torque_limit, position_limits=config.position_limits,
                velocity_limits=config.velocity_limits),
            policy_hz=50.0, command_hz=200.0, hardware_gate=True,
            send_enabled=True, enable_piper=True, startup=startup,
            hardware_access=True)
        stats = loop.run(0.08, [0.0, 0.0, 0.0], config.ee_pose)
        self.assertGreater(stats.policy_steps, 0)
        self.assertGreater(stats.command_steps, 0)
        self.assertTrue(policy_calls)
        self.assertTrue(d1.sent and piper.sent)
        self.assertTrue(d1.estopped and piper.estopped)
        self.assertFalse(loop.safety.fault)

    def test_low_level_loop_rejects_feedback_stale_between_policy_ticks(self):
        """A stalled bus must trip the 500-Hz safety path before next inference."""
        import time

        config = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        seed = H.StateSnapshot(
            q=np.asarray(config.default_q, np.float32),
            dq=np.zeros(22, np.float32), tau=np.zeros(22, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            gyro=np.zeros(3, np.float32), accel=np.asarray([0, 0, 9.81], np.float32),
            valid=True, monotonic_ns=time.monotonic_ns())

        class StallingBus:
            def __init__(self):
                self.connected = False
                self.reads = 0

            def connect(self, **_kwargs):
                self.connected = True

            def disconnect(self):
                self.connected = False

            def read(self):
                self.reads += 1
                value = seed.copy()
                # Let the initial feedback gate observe fresh samples, then
                # emulate a cached vendor frame that never advances.
                value.monotonic_ns = (
                    time.monotonic_ns() if self.reads <= 8
                    else time.monotonic_ns() - 200_000_000)
                return value

            def send(self, _command, **_kwargs):
                raise AssertionError("send must not be reached after feedback timeout")

        d1, piper = StallingBus(), StallingBus()
        safety = H.SafetySupervisor(
            state_timeout_ms=5.0, command_timeout_ms=100.0,
            torque_limits=config.torque_limit, position_limits=config.position_limits,
            velocity_limits=config.velocity_limits)
        loop = R.HardwarePolicyRuntimeLoop(
            d1, piper,
            H.HardwarePolicyRuntime(lambda _f, _h: np.zeros(22, np.float32), config=config),
            safety, policy_hz=50.0, command_hz=500.0, state_reader_hz=500.0,
            hardware_gate=False, send_enabled=False, startup=None, hardware_access=False)
        stats = loop.run(0.08, [0.0, 0.0, 0.0], config.ee_pose)
        self.assertIn("timeout", stats.last_fault)
        self.assertLess(stats.command_steps, 40)

    def test_runtime_rejects_force_direct_without_real_send(self):
        with self.assertRaises(ValueError):
            R.HardwarePolicyRuntimeLoop(
                object(), object(),
                H.HardwarePolicyRuntime(lambda _f, _h: np.zeros(22, np.float32)),
                H.SafetySupervisor(), d1_force_direct=True)

    def test_command_replay_loader(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory) / "commands.npz"
            np.savez(
                replay_path,
                command_velocity=np.zeros((2, 3), np.float32),
                command_ee_pose=np.tile(np.asarray([0.4, 0, 0.5, 1, 0, 0, 0], np.float32), (2, 1)),
                rate_hz=np.asarray(50.0, np.float32),
            )
            velocity, pose, rate = M._load_command_replay(replay_path)
            self.assertEqual(velocity.shape, (2, 3))
            self.assertEqual(pose.shape, (2, 7))
            self.assertEqual(rate, 50.0)

    def test_hardware_runtime_applies_conservative_action_gate(self):
        snapshot = H.StateSnapshot(
            q=np.asarray(M.DEFAULT_Q, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            valid=True, monotonic_ns=__import__("time").monotonic_ns(),
        )
        runtime = H.HardwarePolicyRuntime(lambda _frame, _history: np.full(22, 100.0),
                                          max_action_abs=10.0)
        _frame, _history, command = runtime.step(snapshot, [0, 0, 0], [0.425, 0, 0.5, 1, 0, 0, 0])
        self.assertTrue(np.all(np.abs(runtime.action) <= 10.0))
        self.assertTrue(np.all(np.abs(command.velocity[H.WHEEL_INDICES]) <= 50.0))

    def test_merge_snapshots_uses_conservative_timestamp(self):
        now = __import__("time").monotonic_ns()
        d1 = H.StateSnapshot(valid=True, monotonic_ns=now - 10)
        piper = H.StateSnapshot(valid=True, monotonic_ns=now - 20)
        piper.q[16:] = 1.0
        merged = H.merge_snapshots(d1, piper)
        self.assertEqual(merged.monotonic_ns, now - 20)
        self.assertTrue(np.array_equal(merged.q[16:], np.ones(6)))

    def test_joint_calibration_schema(self):
        import yaml
        config = yaml.safe_load((ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml").read_text())
        H.validate_joint_calibration(config)
        calibration = H.JointCalibration.from_config(config)
        self.assertTrue(np.array_equal(calibration.bus_index[:16], np.arange(16)))
        self.assertTrue(np.array_equal(calibration.bus_index[16:], np.arange(6)))
        with self.assertRaises(ValueError):
            H.validate_joint_calibration(config, require_complete=True)

    def test_joint_calibration_sign_offset_round_trip(self):
        calibration = H.JointCalibration(
            bus_index=np.r_[np.arange(16), np.arange(6)],
            can_id=np.arange(22),
            direction=np.asarray([-1.0] + [1.0] * 21, np.float32),
            position_offset_rad=np.asarray([0.3] + [0.0] * 21, np.float32),
        )
        q_policy = np.arange(16, dtype=np.float32)
        dq_policy = np.ones(16, np.float32)
        tau_policy = np.full(16, 2.0, np.float32)
        q_bus, dq_bus, tau_bus = calibration.command(q_policy, dq_policy, tau_policy, bus="d1")
        self.assertAlmostEqual(float(q_bus[0]), 0.3)
        self.assertAlmostEqual(float(q_bus[1]), 1.0)
        q_back, dq_back, tau_back = calibration.feedback(q_bus, dq_bus, tau_bus, bus="d1")
        self.assertTrue(np.allclose(q_back, q_policy))
        self.assertTrue(np.allclose(dq_back, dq_policy))
        self.assertTrue(np.allclose(tau_back, tau_policy))

    def test_safety_latch_zeroes_commands(self):
        safety = H.SafetySupervisor()
        snapshot = H.StateSnapshot(valid=True, monotonic_ns=__import__("time").monotonic_ns())
        safety.check_state(snapshot, base_height_m=0.20)
        limited = safety.limit_command(H.JointCommand())
        self.assertTrue(np.array_equal(limited.torque, np.zeros(22, np.float32)))
        self.assertIn("base height", safety.fault)

    def test_valid_state_passes_safety_shape_check(self):
        # A valid 22-DoF snapshot must not be rejected merely because the
        # tuple shape is compared against a scalar.  This guards the startup
        # path used by the future real CAN runtime.
        safety = H.SafetySupervisor()
        snapshot = H.StateSnapshot(
            valid=True,
            monotonic_ns=__import__("time").monotonic_ns(),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
        )
        safety.check_state(snapshot, base_height_m=0.45)
        self.assertEqual(safety.fault, "")

    def test_safety_height_check_fails_closed_without_source(self):
        snapshot = H.StateSnapshot(
            q=np.asarray(M.DEFAULT_Q, np.float32),
            dq=np.zeros(22, np.float32), tau=np.zeros(22, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            gyro=np.zeros(3, np.float32), accel=np.asarray([0, 0, 9.81], np.float32),
            valid=True, monotonic_ns=__import__("time").monotonic_ns())
        safety = H.SafetySupervisor(require_base_height=True)
        safety.check_state(snapshot)
        self.assertIn("base height source", safety.fault)

    def test_feedback_timeout_is_resolved_from_runtime_yaml(self):
        """Feedback freshness must be a shared, editable deployment parameter."""
        resolved = CFG.load_resolved_config(ROOT / "loco_mani_rl_controller/config/d1_piper_l.yaml")
        self.assertAlmostEqual(resolved.feedback_timeout_s, 0.020)

    def test_safety_rejects_malformed_accelerometer(self):
        """All IMU vectors, including acceleration, are part of the state ABI."""
        safety = H.SafetySupervisor()
        snapshot = H.StateSnapshot(
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            monotonic_ns=__import__("time").monotonic_ns(), valid=True,
        )
        snapshot.accel = np.zeros(2, dtype=np.float32)
        safety.check_state(snapshot)
        self.assertIn("shape", safety.fault)

    def test_safety_limits_position_and_velocity_targets(self):
        position_limits = [None] * 22
        position_limits[16] = (-1.0, 1.0)
        velocity_limits = np.full(22, 2.0, dtype=np.float32)
        safety = H.SafetySupervisor(position_limits=position_limits,
                                    velocity_limits=velocity_limits)
        # Prime the watchdog; the first command is allowed immediately.
        command = H.JointCommand(
            position=np.full(22, 5.0, dtype=np.float32),
            velocity=np.full(22, 5.0, dtype=np.float32),
        )
        limited = safety.limit_command(command)
        self.assertAlmostEqual(float(limited.position[16]), 1.0)
        self.assertAlmostEqual(float(limited.position[0]), 5.0)
        self.assertTrue(np.all(np.abs(limited.velocity) <= 2.0))

    def test_safety_limits_effective_pd_torque(self):
        """The sent impedance command, not only t_ff, stays within limits."""
        safety = H.SafetySupervisor(
            torque_limits=np.full(22, 2.0, np.float32),
            position_limits=[None] * 22,
            velocity_limits=np.full(22, 10.0, np.float32),
            max_torque_rate_nm_s=1.0,
            command_timeout_ms=1000.0,
        )
        now = __import__("time").monotonic_ns()
        state = H.StateSnapshot(
            q=np.zeros(22, np.float32), dq=np.zeros(22, np.float32),
            quat_wxyz=np.asarray([1, 0, 0, 0], np.float32),
            valid=True, monotonic_ns=now,
        )
        command = H.JointCommand(
            position=np.ones(22, np.float32) * 10.0,
            velocity=np.zeros(22, np.float32),
            kp=np.ones(22, np.float32), kd=np.zeros(22, np.float32),
            torque=np.zeros(22, np.float32),
        )
        limited = safety.limit_command(command, state=state, now_ns=now, dt=0.1)
        effective = limited.torque + limited.kp * (limited.position - state.q)
        self.assertTrue(np.all(np.abs(effective) <= 2.0 + 1e-6))
        self.assertTrue(np.all(np.abs(effective) <= 0.1 + 1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
