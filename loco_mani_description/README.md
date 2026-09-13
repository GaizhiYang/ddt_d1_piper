# loco_mani_description

This package is the single robot description used by the D1+Piper-L rollout.
The combined URDF is derived from
`/home/hh/loco-mani/source/LeggedManip_Lab/LeggedManip_Lab/assets/d1_piper_l/d1_piper_l.urdf`;
the D1 MuJoCo subtree is retained from `urdfs/d1_description` and the Piper-L
chain is appended at `base_link`.

Regenerate after changing either upstream description:

```bash
cd /home/hh/loco_mani_ws/src/ddt_controller
python3 scripts/generate_d1_piper_l_assets.py
```

生成器会将 Piper URDF 的固定轴 XYZ `rpy` 先转换为 MuJoCo `wxyz` 四元数；不要
把 URDF 的 `rpy` 字符串直接当作 MuJoCo `euler` 使用，否则 `joint2` 等关节的
连杆姿态会发生变化。

The policy joint order is fixed and must not be inferred from XML order:

```text
FL hip/thigh/calf/foot, FR hip/thigh/calf/foot,
RL hip/thigh/calf/foot, RR hip/thigh/calf/foot,
Piper joint1..joint6
```

The two gripper finger slides are passive and are intentionally excluded from
the 22-DoF policy interface.

`urdf/robot.urdf` and `xacro/robot.xacro` are generated from the same
authoritative training URDF.  Their Piper link/joint origins and axes are kept
identical; the MuJoCo generator converts URDF fixed-axis XYZ `rpy` to body
quaternions before writing `mujoco/robot.xml`.

`urdf/robot.urdf` is deliberately hardware-independent and contains no
`<ros2_control>` block.  The xacro wrapper includes
`xacro/ros2control.xacro`, whose `loco_mani_ros2_control` macro adds the 22
policy joints and IMU interfaces.  The default plugin is
`mujoco_ros2_control/MujocoSystem`; select another plugin without editing the
robot model:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
xacro loco_mani_description/xacro/robot.xacro \
  hardware_plugin:=your_ros2_control_plugin \
  ros2_control_name:=hardware >/tmp/d1_piper_l.urdf
```

This split keeps model-only tools (`check_urdf`, RViz and the MuJoCo viewer)
independent of a hardware plugin while allowing a ROS 2 control launch file to
inject the plugin it actually runs.

For compatibility with the original D1 packages, the standalone file also
exports a `ros2control` alias accepting `hw_env:=none|webots|gazebo|hw`.
New launch files should prefer the explicit `loco_mani_ros2_control` macro and
`hardware_plugin` argument.

The package also installs a model-only viewer for checking geometry without a
controller:

```bash
python3 loco_mani_description/scripts/view_mujoco_model.py
```

The WBC command is frame-sensitive: `x/y` are relative to `piper_base_link`,
`z` is an absolute world height, and the quaternion uses `[qw, qx, qy, qz]` in
the Piper-base frame.  The current mount transform (`piper_base_link` at
`(0.20, 0, 0.09)` in `base_link`) is copied from the training asset and must
be replaced only after measuring the real bracket.

The generated MuJoCo root height is 0.45 m, matching the training reset.  With
the legacy wheel collision primitive this produces about 6.6 mm initial
penetration; the rollout exposes `--initial-height 0.4567` for a collision-free
diagnostic comparison.  Confirm the generated USD contact/rest offsets before
choosing which value is used for policy validation.  The upstream DDT
standalone demo uses 0.8 m, so rerun the generator after changing the source
description rather than copying that older value back.
