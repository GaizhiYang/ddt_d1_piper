# D1 + Piper-L 真机部署边界

这个仓库的 sim2sim 控制器与真机控制器共用 policy ABI，但不会在启动时自动使能电机。真机后端必须按下边界实现：

```text
StateProvider
  D1Backend(can0)      -> 16 x (q, dq, tau) + base IMU
  PiperBackend(can1)   -> 6 x (q, dq, tau)
ObservationAdapter     -> 82-D frame, history 3
ONNXPolicy             -> 246-D input, 22-D raw action
ActionAdapter          -> D1 PD/velocity + Piper MIT
SafetySupervisor       -> watchdog, timeout, tilt/height, limits, E-stop
```

仓库中的 `loco_mani_rl_controller/scripts/d1_piper_hardware.py` 已提供上述
边界的 Python 接口和安全门控，可用于 dry-run/shadow 测试。它不会因为导入或
启动而打开 CAN；D1 的 `CanfdApi` 需要现场提供一个经过 ABI 验证的适配器，Piper
才会在 `hardware_gate=True` 后调用 `pyAgxArm`。快速检查配置（纯只读）:

```bash
python3 loco_mani_rl_controller/scripts/d1_piper_hardware.py \
  --config loco_mani_rl_controller/config/d1_piper_l.yaml
```

输出中的 `hardware_opened` 必须为 `false`。不要把这个脚本当作已经完成的
真机控制节点；在双 CAN 发送前仍须完成下文的 trace、机械支撑和急停验收。

当前还提供参考编排循环 `loco_mani_runtime.py`：两个独立读线程、50 Hz 策略、
500 Hz 限幅/发送，并在状态超时、策略 watchdog 或发送失败时锁存故障。默认
运行是 dry-run，不会打开总线：

```bash
python3 loco_mani_rl_controller/scripts/loco_mani_runtime.py \
  --config loco_mani_rl_controller/config/d1_piper_l.yaml \
  --policy-path /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx \
  --duration 5
```

只有完成 D1 `api_factory`/`api_motor_out_factory`、逐关节标定和机械支撑验收后，
才可通过 `--send --hardware-gate` 进入发送路径；此选项不会替代外部急停。D1
直控模式默认不自动切换；现场若确认需要软件切换，必须另外使用
`--d1-force-direct`（ROS launch 对应 `d1_force_direct:=true`）。它只调用适配器中
明确实现的 `SET_READY_NEXT=FORCE_DIRECT` RPC，不会在 connect/read 或 shadow 模式执行。

## 只读硬件检查

仓库新增 `hardware/d1_vendor_adapter`，将 DDT C++ API 封装为固定数组的纯 C
ABI；Python 不直接猜测 `std::vector` 或 `api_motor_out_t` 布局。目标机必须用
同一份 `canfd_api.hpp` 和 `libtita_robot.so` 编译该适配器，然后采样：

```bash
export LOCO_MANI_D1_ADAPTER_LIB=/path/to/libloco_mani_d1_vendor_adapter.so
python3 install/loco_mani_rl_controller/lib/loco_mani_rl_controller/loco_mani_hardware_inspect.py \
  --config src/ddt_controller/loco_mani_rl_controller/config/d1_piper_l.yaml \
  --bus both --samples 20 --period 0.1 --hardware-gate \
  --output /tmp/d1_piper_feedback.json
```

工具只读取 D1 电机/IMU、Piper 固件和 6 轴状态，不调用 enable、MIT、D1 发送或
自动急停（Piper 固件查询只发送 SDK 的查询请求帧，不发送任何运动指令）；未完成
标定时只允许该只读模式，`--send` 仍拒绝所有空的 ID、方向和零位字段。

## 总线和驱动

* D1 的 `CanfdApi` 当前固定使用 `can0`，可复用其 16 电机和 IMU 接收接口；不要把 22 个关节传给现有 `hardware_bridge`，因为它会按 `mJoints.size() % 8/%6` 推断腿组，22 会被错误分成 3-DOF 组。发送符号需按现场库确认：当前已检查的 `libtita_robot.so` 只有 `send_motors_can`，旧版可能提供 `send_leg_motors_can`；本仓库后端兼容两者。
* Piper-L 必须用 `pyAgxArm` 的独立实例连接 USB2CAN 的 `can1`。先读取 `get_firmware()`，再用 `resolve_firmware_profile()` 选择 `default/v183/v188/v189`；不同固件的 MIT 编码和模式值不同。SDK 的配置参数名是既有拼写 `firmeware_version`，通信配置应明确使用 `comm="can"`、`interface="socketcan"`、`channel="can1"` 和 `bitrate=1000000`；不要沿用官方 ROS 节点默认的 `can0`。
  如果 SDK 只以源码形式存在，启动真实运行时的同一个 Python 环境必须包含
  `PYTHONPATH=/home/hh/pyAgxArm`（或先将 SDK 安装到该环境）；导入检查本身不会打开 CAN。
  `get_joint_angles()` 已由 SDK 转换为 rad；高速 `get_motor_states()` 同样提供
  rad/rad/s 和关节侧 N·m，不能再次乘减速比或角度换算。
* 两个 backend 应有独立的接收线程和时间戳快照。策略 50 Hz，PD/命令发送建议 500 Hz；策略线程不能直接阻塞在 CAN 接收或 ROS 回调中。

`runtime.feedback_timeout_ms`（默认 20 ms）是两条总线共享的反馈新鲜度门限；
`runtime.command_hz`、`runtime.state_reader_hz` 和
`runtime.policy_watchdog_timeout_ms` 分别控制发送、读取和策略 watchdog 频率。
ROS 策略节点与独立硬件运行时都会从同一份 YAML 读取这些值，命令行只有在显式
传入时才覆盖它们。

当前配置中 `joint_calibration.*.bus_index/can_id/direction/position_offset_rad`
仍为 `null`。因此 `loco_mani_preflight.py --for-send` 预期会拒绝启动；这不是
软件故障，而是防止在不知道真实电机映射时误发运动命令的安全门。完成现场只读
采样后，必须人工把 22 个关节的实测值写回 YAML，再重新执行 preflight。

安全布尔字段（`dry_run`、`enable_on_start`、`startup.enabled` 和
`base_height_check_enabled`）使用严格解析：`false`/`0`/`off` 会保持关闭，
`true`/`1`/`on` 才会开启，其他拼写会直接拒绝配置。不要依赖 Python 的
`bool("false")` 语义，因为它会把字符串误判为真。

## 上电顺序（默认 dry-run）

1. 配置并检查 `can0`、`can1` 的比特率、终端电阻和 USB2CAN 驱动；确认接口名没有被 udev 改写。
2. 启动接收，只读关节、IMU、Piper 固件/状态；检查 16+6 顺序、单位（rad/rad/s/Nm）、方向和零位。
3. 运行 `scripts/check_policy_abi.py`；以 shadow mode 计算动作，只记录不发送。
4. 使用机械支撑，单独验证 Piper 低增益 MIT，再验证 D1 静态站立；每次只使能一个 backend。
5. 通过外部急停和软件 gate 后才允许发送命令。`enable_on_start` 必须保持 `false`，`dry_run` 必须先改为 `false` 才能进入实机测试。

## 必须实现的安全条件

* 接收时间戳超过 20 ms、策略 watchdog 超过 40 ms、发送失败或状态错误时，立即停止策略并发送安全命令；急停后禁止自动重新使能。
* 动作、位置、速度、力矩和力矩变化率均限幅；倾角超过 35 deg 时进入故障态。
  基座高度保护只有在 `safety.base_height_check_enabled=true` 且运行时提供经过验证的高度源时才启用；当前默认配置关闭该项，避免把没有世界高度测量的 D1 反馈误当成有效高度。
* Piper 关节命令应使用训练的 `p_des/v_des/kp/kd/t_ff`，而不是把腿部 CAN 报文格式复用到 `can1`。
* 所有关节方向/offset/限位必须在 YAML 中显式配置并逐关节验证，不能依赖 XML 顺序或 SDK 默认值。

配置中的 `joint_calibration` 已为 22 个关节预留 `bus_index`、`can_id`、`direction`
和 `position_offset_rad`。这些字段当前是 `null`，表示尚未完成实物标定；只有
全部填充后才允许将 `dry_run` 改为 `false`。运行时按
`q_bus = direction * q_policy + position_offset_rad` 转换位置，速度和力矩按同一
方向转换，绝不默认采用 XML 顺序。`validate_joint_calibration(...,
require_complete=True)` 与 `JointCalibration.complete` 可作为启动 gate。

`SafetySupervisor` 在发送前还会对逐关节位置、速度、力矩和力矩变化率进行限幅；
但它不会替代驱动器内部限位，现场仍须读取并核对 Piper 的 MIT 量程、D1 电机
保护阈值以及通信超时后的硬件行为。

完整的仿真差异、验收标准和进入 CAN 的门槛见 [sim2sim_audit.md](sim2sim_audit.md)。

在 Jetson 上接入 D1 库前，可先执行只读 ABI 检查（不会加载库或打开 CAN）：

```bash
python3 scripts/check_d1_vendor_abi.py \
  /path/to/libtita_robot.so --expected-arch aarch64
```

检查报告中的 `send_motors_can`/`send_leg_motors_can` 决定实际发送路径；无论
哪一种符号，都仍需使用与该库同一编译器/`libstdc++` ABI 的 C++/pybind 适配器
验证 `api_motor_out_t` 和 `std::vector` 布局。
