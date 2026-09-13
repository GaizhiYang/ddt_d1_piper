# D1 vendor C ABI adapter

`libtita_robot.so` exposes a C++ API (`CanfdApi`, `std::vector` and packed
structures).  Python must not guess that ABI.  This package compiles a small
flat C adapter against the exact header and library used on the target image:

```text
Python/ctypes ── fixed arrays ──> loco_mani_d1_vendor_adapter
                                      └─ exact canfd_api.hpp + libtita_robot.so
```

The adapter supports exactly 16 D1 motors, `can0`, one coherent read snapshot,
the vendor `send_motors_can` API, motor/IMU timeout flags, a vendor feedback
timestamp and a diagnostic string.  Constructing a handle opens CAN; the
Python runtime therefore calls the factory only after `--hardware-gate`.
Importing the Python wrapper or loading this shared object does not construct
a handle.  The timestamp is used to reject repeated cached frames before they
can satisfy the policy watchdog.

## Build

Use matching header and library roots from the same vendor installation.  The
workstation fallback is only for compile checks:

```bash
cd /home/hh/loco_mani_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select loco_mani_d1_vendor_adapter \
  --cmake-args -DD1_TITA_ROBOT_ROOT=/home/hh/ddt_ros2_ws/ddt_ros2_control/hardware/tita_robot
```

On Jetson, pass the ARM64 vendor root and rebuild there.  CMake selects
`lib/arm64` automatically when the target architecture is `aarch64`; do not
copy the x86-64 `.so` to the Jetson.

## Python factory

`loco_mani_rl_controller/scripts/d1_tita_adapter.py` wraps this library and
implements the narrow object protocol consumed by `D1Backend`:

```bash
export LOCO_MANI_D1_ADAPTER_LIB=/path/to/libloco_mani_d1_vendor_adapter.so
ros2 launch loco_mani_bringup hardware.launch.py \
  shadow:=true hardware_gate:=true \
  d1_api_factory:=d1_tita_adapter:create_d1_api \
  d1_motor_out_factory:=d1_tita_adapter:make_motor_out
```

The wrapper has no default real-bus fallback.  If the library path is absent,
the factory fails before CAN is opened.

The adapter also exports `loco_mani_d1_set_force_direct()`. This is a
state-changing `SET_READY_NEXT=FORCE_DIRECT` RPC and is intentionally separate
from create/read/send. The runtime calls it only when the operator explicitly
sets `--send --hardware-gate --d1-force-direct` (ROS launch:
`d1_force_direct:=true`); it is never used in inspection or shadow mode.

The adapter does not infer joint IDs, signs, offsets, or policy order.  Those
remain a separate calibration/inspection task.  `loco_mani_hardware_inspect.py`
is the read-only entry point for collecting raw bus-order feedback.

## Read-only inspection

After manually confirming the interface names and installing the matching
adapter `.so`, use:

```bash
export LOCO_MANI_D1_ADAPTER_LIB=/path/to/libloco_mani_d1_vendor_adapter.so
python3 install/loco_mani_rl_controller/lib/loco_mani_rl_controller/loco_mani_hardware_inspect.py \
  --config src/ddt_controller/loco_mani_rl_controller/config/d1_piper_l.yaml \
  --bus d1 --samples 20 --period 0.1 --hardware-gate \
  --output /tmp/d1_feedback.json
```

Use `--bus piper` for Piper-only or `--bus both` for both buses.  The tool
does not call enable, MIT, D1 send, or automatic emergency-stop.  It still
opens the selected CAN interface, so `--hardware-gate` is an explicit manual
authorization.  Incomplete calibration means the printed arrays are raw bus
order and must not be copied into YAML without manual mapping.
