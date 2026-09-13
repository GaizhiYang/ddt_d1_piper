# D1 + Piper-L policy sim2sim audit

This note compares the IsaacLab training asset/configuration with the current
MuJoCo description.  It is intentionally a calibration report; passing the
ONNX ABI check does not imply that the dynamics are equivalent.

## What is already aligned

The Piper chain in the training `d1_piper_l.urdf` and this package's generated
`urdf/robot.urdf` has matching mount transform, link/joint origins, axes and
parent-child relationships.  URDF fixed-axis XYZ `rpy` is converted to a
MuJoCo body quaternion by the asset generator; it must not be copied directly
to a MuJoCo `euler` attribute, whose rotation convention differs.

* The training policy export has one `float32` input of 246 values (three
  82-value frames) and one output of 22 actions.
* The active joint order is
  `FL, FR, RL, RR` (hip, thigh, calf, wheel), followed by Piper `joint1..6`.
* The default active-joint pose is `[0, .8, -1.5, 0]` per leg and six zero arm
  joints.  Wheel positions are masked to zero in the policy observation.
* IsaacLab uses `sim.dt=0.005`, `decimation=4`, hence a 50 Hz policy update;
  the controller script exposes the same values.
* The URDF and MJCF inertial masses agree to numerical precision for the
  physical links (about 54.97676 kg).  The empty `end_effector` frame remains
  massless; MuJoCo uses only a negligible epsilon mass for its movable passive
  `gripper_link` body.
* PD gains, action scales and effort limits in the controller correspond to
  the values in `d1_piper_articulation_cfg.py`.

## High-priority mismatches

### Contact and collision geometry

The generated USD is converted with `collider_type: convex_hull` and keeps the
URDF collision elements.  The MJCF uses manually authored boxes/cylinders for
the D1 and Piper URDF STL meshes as convex collision proxies.  Piper
self-collision remains disabled while floor contact is enabled.  The foot
cylinder is initially about 6.24 mm
below the plane when the training root height is 0.45 m.  This explains why
the current script needs approximately 0.4567 m merely to remove initial
penetration; it is not evidence that the training reset height is wrong.

Recommended calibration order:

1. Export/inspect the actual USD collider extents and contact points (not the
   visual meshes).  Confirm wheel radius, axis, and each wheel link origin.
2. Rebuild the MJCF collision geoms from the URDF collision meshes (convex
   hull or capsule) and enable Piper terrain collision.  Keep visual meshes
   non-colliding.
3. Run a static contact test at z=0.45 with zero action and record penetration,
   normal force, and contact point for all four wheels.  Only then choose a
   MuJoCo-only reset offset if PhysX deliberately starts in penetration.

### Actuator model

IsaacLab's leg `DCMotor` has `friction=0.589`, armature `0.0535`, and the
wheel group has velocity damping 0.5.  The arm uses
`DelayedPDActuator` with per-joint Coulomb friction 0.01 and 0--4 simulation
step delay.  The controller currently applies explicit PD and torque clipping,
but does not add the leg/arm actuator friction term.  The rollout now keeps
MJCF `frictionloss` enabled by default (and provides a zero-scale ablation),
while passive viscous damping remains disabled to avoid double counting.

The rollout maps the configured Coulomb coefficients to MuJoCo
`dof_frictionloss` (and exposes `--passive-frictionloss-scale` for an ablation),
but this is still an approximation of PhysX's static/dynamic friction
semantics.  Validate the sign/zero-velocity behavior with a one-joint step
response, retain armature in the model, and compare against a recorded
IsaacLab command trace.  Keep the Piper delay queue at simulation-step
resolution and verify its off-by-one behavior before claiming dynamic parity.

### Physics solver/contact parameters

Training uses PhysX solver position iterations 4, velocity iterations 0,
`max_depenetration_velocity=1`, zero rigid-body damping, plane friction 1/1,
and restitution 1 (with robot material randomized to friction 0.5--1.2 and
restitution 0--0.1).  The current MJCF defaults to MuJoCo Newton, 100
iterations, `solref=[0.02,1]`, `solimp=[0.9,0.95,...]`, and default geom
friction 0.4; these are not equivalent.  MuJoCo cannot reproduce PhysX
iterations one-to-one, so tune contact softness/iterations only after geometry
is corrected and document the chosen approximation.  At minimum set floor and
wheel tangential friction to 1.0 and expose `solref/solimp`, friction and
solver options as calibration parameters.

### Gripper and passive DOFs

The source URDF contains a passive `gripper` prismatic DOF plus two mimic
finger joints.  The generated MJCF retains all three joints and enforces the
`+/-0.5` mimic relations with equality constraints.  They are not in the 22-D
policy, but their reset state and negligible epsilon body mass remain part of
the asset comparison.

## Observation/action checks

The policy frame is

```
0.2 * base_ang_vel_b, projected_gravity_b,
joint_pos - default_joint_pos (wheel entries forced to 0),
joint_vel - default_joint_vel, last_action,
base_velocity_command, ee_pose_command
```

The training observation group enables noise (angular velocity ±0.2 before
the 0.2 scale, gravity ±0.05, position ±0.01, velocity ±1.5).  The current
sim2sim controller intentionally omits noise; that is useful for deterministic
calibration, but a hardware policy should either reproduce the trained noise
statistics or confirm that noise was disabled in the exported deployment
configuration.  Confirm that `last_action` is the action-manager output (raw
concatenated 22-vector), not the delayed Piper command.

WBC command semantics are frame-sensitive: x/y are relative to
`piper_base_link`, z is world height, and the quaternion is expressed in the
Piper base frame.  The mount `(0.20, 0, 0.09)` and `end_effector` fixed frame
must be measured on the physical bracket and checked against the USD; do not
silently change these values to make a rollout look stable.

## Required acceptance tests before CAN or hardware

1. Compare the first IsaacLab reset observation with the MuJoCo first frame,
   element by element (including quaternion convention and joint ordering).
2. With gravity disabled, command fixed joint targets and verify static PD
   error, torque limits, armature and friction for every joint.
3. With gravity enabled and policy disabled, verify wheel contact height,
   normal forces, base tilt and passive settling.
4. Replay an IsaacLab action trace in MuJoCo and compare q/dq/torque and end
   effector pose over at least 5--10 seconds.
5. Only proceed after stable standing, low-speed x/y/yaw motion, Piper end
   effector tracking, no persistent torque saturation, and long rollouts
   without falls.  Then repeat the same tests through the can0/can1 hardware
   adapters with torque disabled or a mechanical support.

### Fixed action replay

The MuJoCo controller accepts `--action-replay` for deterministic replay of an
IsaacLab action trace.  The file contains raw action-manager outputs at the
policy rate (normally 50 Hz), shaped `[N, 22]`; `.npy`, `.npz`, JSON and CSV
are supported.  This bypasses ONNX inference but preserves action scaling,
wheel velocity terms, Piper delay and torque clipping.  Use it to compare
IsaacLab and MuJoCo state/torque traces before tuning contact parameters:

```bash
python3 loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py \
  --action-replay /tmp/isaac_actions.npy --replay-rate-hz 50 \
  --diagnostics-path /tmp/mujoco_replay.csv \
  --headless --duration 10 --initial-height 0.4567
```

The replay path is intentionally separate from the policy path, so an action
trace cannot be accidentally interpreted as an observation or run through the
network twice.

### ROS 2 description split

`loco_mani_description/urdf/robot.urdf` is hardware-independent and contains
no `<ros2_control>` element.  The ROS 2 wrapper `xacro/robot.xacro` includes
`xacro/ros2control.xacro` and instantiates the `loco_mani_ros2_control` macro.
Its `hardware_plugin` and `ros2_control_name` arguments let simulation and
real-hardware launch files select their own plugin without modifying the
kinematic URDF.

### Policy-rate trace comparison

For an element-wise ABI and dynamics check, request a policy-rate trace in
addition to the per-physics-step diagnostics:

```bash
python3 loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py \
  --policy-trace-path /tmp/mujoco_policy_trace.csv \
  --diagnostics-path /tmp/mujoco_physics.csv \
  --headless --duration 10 --initial-height 0.4567
```

The policy trace records the exact 82-D frame, 246-D term-major history and
the action used at each 50 Hz update, plus pre-integration `q/dq/tau`.  Convert
the IsaacLab export to the same column names, then compare it with:

```bash
python3 scripts/compare_policy_traces.py \
  /path/to/isaac_policy_trace.csv /tmp/mujoco_policy_trace.csv \
  --report /tmp/trace_report.json
```

The tool fails on missing columns, non-finite values or unequal sample counts
(use `--align-time` only when timestamps are known to represent the same
samples).  This prevents a visually stable but ABI-misaligned rollout from
being accepted as sim2sim parity.

## Source references

* `LeggedManip_Lab/assets/d1_piper_l/d1_piper_articulation_cfg.py`
* `LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/d1_piper_l/d1_piper_common_env_cfg.py`
* `LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/config/d1_piper_l/wbc_env_cfg.py`
* `LeggedManip_Lab/assets/d1_piper_l/generated_usd/config.yaml`
* `loco_mani_description/mujoco/robot.xml`
