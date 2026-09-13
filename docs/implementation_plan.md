# D1 + Piper-L 全身控制 sim2real 实施方案

本文是从现有 `ddt_controller`、D1 驱动、`pyAgxArm` 和 IsaacLab 训练工程
落地到 Jetson Orin NX 的实施合同。当前仓库已完成不自动使能电机的双 CAN 代码
入口、flat-C D1 ABI 适配器和只读检查工具；真实发送仍必须经过现场标定和人工门控。

## 目标运行时

```text
500 Hz 实时循环
  ├─ D1Backend (can0，16 个轮足关节 + IMU)
  ├─ PiperBackend (can1，6 个关节，pyAgxArm)
  ├─ JointCalibration / StateSnapshot
  ├─ ObservationAdapter (82 维，3 帧历史)
  ├─ ONNXPolicy (50 Hz，20 ms)
  ├─ ActionAdapter (腿部 PD/轮速 + Piper MIT)
  └─ SafetySupervisor / E-stop / watchdog
```

策略线程不得直接阻塞在 CAN 接收、ROS 回调或 Python SDK 调用上。推荐每个
总线使用独立接收线程，以带时间戳的原子快照供实时循环读取；发送线程只消费
已经过限幅和安全检查的命令。

## 分阶段路线

### 0. 资产和策略冻结

冻结以下 ABI，并把它们作为启动时检查项：

* ONNX 输入 `obs[1,246]`、输出 `actions[1,22]`，float32；
* 关节顺序 `FL/FR/RL/RR` 的 `hip, thigh, calf, wheel`，后接 Piper
  `joint1..joint6`；
* 单帧顺序：`base_ang_vel(3), projected_gravity(3), joint_pos(22),
  joint_vel(22), last_action(22), velocity_command(3), ee_pose(7)`；
* IsaacLab `dt=0.005`、`decimation=4`、策略频率 50 Hz、历史长度 3；
* 动作缩放 `[0.25,0.25,0.25,5.0] * 4 + [0.25] * 6`。

策略文件、训练配置、USD 和导出日志应一起归档并记录 SHA256。更换策略时
必须重新运行 `scripts/check_policy_abi.py`，不能仅凭文件名判断兼容性。

### 1. IsaacLab → MuJoCo 动力学对齐

仓库提供 `scripts/export_isaaclab_trace.py`，可在训练机的 IsaacLab 环境中直接
记录策略输入、动作、应用力矩、22 个关节状态、基座姿态和末端状态。它输出的
CSV 与 MuJoCo 的 `--policy-trace-path` 列格式兼容，避免重新实现
`ObservationManager` 而引入隐藏误差。例如：

```bash
./isaaclab.sh -p /home/hh/loco_mani_ws/src/ddt_controller/scripts/export_isaaclab_trace.py \
  --checkpoint /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/model_8000.pt \
  --output /tmp/isaac_policy_trace.csv --duration 10 \
  --initial-state-output /tmp/isaac_reset.json \
  --actions-output /tmp/isaac_actions.npy
```

`--initial-state-output` 保存 reset 后、首个 policy action 前的浮动基座和
22 个关节 `q/dq`；`--actions-output` 保存同一 trace 的 raw action。MuJoCo
回放时同时指定 `--action-replay /tmp/isaac_actions.npy
--initial-state /tmp/isaac_reset.json`，避免把 reset 随机化误当作动力学误差。
导出器的 `tau` 字段在 `env.step(action)` 后采样，确保它属于该行 action；
`q/dq` 仍定义为该 action 的 pre-step 状态。

在 IsaacLab 与 MuJoCo 使用相同的初始状态、命令和随机种子后，用
`scripts/compare_policy_traces.py` 比较 `frame/history/action/q/dq/tau`；先只
比较 ABI，再逐步调整碰撞和执行器参数。

按 [sim2sim_audit.md](sim2sim_audit.md) 的顺序处理：碰撞几何、轮子接触高度、
Piper 安装位姿和末端 frame、摩擦/solver 参数、DCMotor/DelayedPD 摩擦及延迟。
随后完成以下测试：

1. IsaacLab reset 第一帧和 MuJoCo 第一帧逐元素比较；
2. 关闭重力进行单关节 PD 阶跃，比较位置、速度、力矩和饱和；
3. 开启重力但关闭策略，比较静态接触、法向力和基座姿态；
4. 回放同一段 Isaac action trace 至少 5--10 秒，比较 `q/dq/tau` 和末端位姿。

修正历史观测的 term-major 排列后，当前 MJCF + 给定 ONNX 在 30 s、零速度
指令下可保持不倒（倾角约 2°）；但末端在该简化模型中仍明显漂移，不能把
“不倒”当作 WBC 任务通过。它只证明推理链路和基本时序正确，仍不能替代
IsaacLab→MuJoCo 的动力学回放验收，因而尚不能进入 CAN 控制。

### 2. 双 CAN 后端（先只读）

新增独立 `D1Backend` 和 `PiperBackend`，不要扩展旧的 `hardware_bridge`：

* D1：独立 `CanfdApi(16, 0, "can0")` 或 Jetson 上实际安装的
  `tita_robot` ABI；仓库的 `hardware/d1_vendor_adapter` 使用准确头文件和库
  导出 flat C ABI，并已针对当前库的 `send_motors_can` 编译验证；仍需确认
  Jetson 实际库版本以及 16 个电机 ID/顺序；
* Piper：独立 `pyAgxArm` 实例，`robot="piper_l"`、`comm="can"`、
  `interface="socketcan"`、`channel="can1"`；先 `connect()` 和
  `get_firmware()`，再调用 `resolve_firmware_profile()` 选择
  `default/v183/v188/v189`；
* 两条总线的接收、错误码、帧时间戳和断线状态必须独立记录；禁止把 Piper
  MIT 报文交给 D1 驱动，或反过来。

`joint_calibration` 中的 `bus_index`、`can_id`、`direction` 和
`position_offset_rad` 必须全部填写后才允许连接真实总线。运行时严格使用
`q_bus = direction * q_policy + offset` 的显式变换；未完成标定时后端只能
保持 `dry_run`，不能用数组/XML 顺序代替实测映射。

仓库中的 `d1_piper_hardware.py` 现已提供 `ObservationAdapter`、
`ActionAdapter`、`HardwarePolicyRuntime` 和 `merge_snapshots`。它们只处理
策略语义，不创建 ROS 节点或打开设备；可先用伪造状态运行 shadow 测试，再由
读取/发送适配器由 `d1_tita_adapter.py` 注入。当前发现的 `libtita_robot.so` 导出
`send_motors_can(records)`（一次发送 16 个电机），部分旧版头文件/插件则
提供 `send_leg_motors_can(records, leg_index)`（四次发送、每次 4 个电机）。
`D1Backend` 优先调用逐腿接口，否则回退到整机接口；两种路径都要求通过显式
`api_motor_out_t` 工厂构造记录，代码不会猜测 C++ 结构体 ABI。

只读阶段必须在 `vcan` 或机械支撑上验证：单位为 rad/rad/s/Nm、反馈顺序、
方向、零位、IMU 四元数顺序（D1 接口为 x/y/z/w，策略适配器使用 w/x/y/z）。

### 3. 标定和动作适配

在 YAML 中逐关节显式保存：CAN ID、策略索引、方向、位置 offset、位置/速度/
力矩限值、最大力矩变化率和故障策略。不得依赖 XML 顺序或 SDK 默认偏置。

腿部动作转换为训练中的位置 PD（轮子为速度目标）；Piper 动作转换为
`move_mit(joint_index, p_des, v_des, kp, kd, t_ff)`。Piper SDK 是逐关节接口，
因此需要固定发送周期、顺序和超时策略，并测量一次完整 6 轴发送的抖动与 CAN
占用率。固件版本会改变 MIT 的力矩编码和运动模式值，必须在实机上确认。

### 4. 实时循环和安全监督

建议 500 Hz 低层循环、50 Hz 策略更新：策略更新时刷新历史缓冲，否则保持上一
个动作；启动历史用当前观测复制 3 次，`last_action` 从零开始。实时循环使用
单调时钟和固定周期，记录最大执行时间、CAN 往返延迟和丢帧。

以下任一条件触发故障态：状态快照超过 20 ms、策略 watchdog 超过 40 ms、发送
失败、驱动错误、急停、倾角超过 35°、基座高度低于 0.25 m、越过关节/速度/力矩
或力矩变化率限制。故障态发送安全零力矩/阻尼命令并禁止自动重新使能。

默认配置必须保持 `enable_on_start=false`、`dry_run=true`；所有使能操作只允许
在外部急停已释放、机械支撑就位且人工 gate 打开后发生。

### 5. 上电验证顺序

1. 检查 `can0/can1` 名称、比特率、终端电阻和 USB2CAN 枚举；
2. 两个 backend 只读，确认 16+6 顺序、单位、方向、零位和 IMU；
3. shadow mode：计算并记录动作，但不发送；
4. 机械支撑下只使能 Piper，低增益 MIT 单轴→六轴；
5. 只使能 D1，验证静态站立和急停；
6. 低增益全身站立，随后低速 x/y/yaw；
7. 最后才开启末端跟踪和完整 WBC 策略。

任一步骤出现持续饱和、姿态漂移、CAN 延迟异常或方向不确定，应回到只读/仿真
阶段，不得通过放宽安全阈值“调稳”。

## 必须由现场确认的参数

* D1 主控和 16 个电机的确切驱动库版本、CAN-FD 比特率、ID 和关节顺序；
* Piper-L 硬件型号、固件字符串、CAN 比特率、6 个电机 ID 和 MIT 版本；
* 机械臂支架相对 D1 `base_link` 的实测 xyz/rpy、末端 frame 定义和负载；
* 每个关节的编码器方向、零位、软/硬限位和最大连续/峰值力矩；
* Jetson 上 ONNX Runtime、ROS 2 Humble、pyAgxArm 的 Python/ABI 版本；
* 外部急停、供电继电器、D1/Piper 驱动在通信超时后的实际行为。

在这些信息确认前，本仓库可以安全地做 ABI 检查、MuJoCo 标定、只读 CAN 和
shadow mode，但不应执行自动使能或发送真实全身动作。
