# loco_mani_bringup

MuJoCo sim2sim 与双 CAN sim2real 代码入口均已提供；默认启动仍不会连接真实 CAN。
运行前请确认
`mujoco`、`numpy` 和 ONNX Runtime 安装在同一个 Python 环境中。

直接运行（不依赖 ROS2 安装）:

```bash
cd /home/hh/loco_mani_ws/src/ddt_controller
/home/hh/anaconda3/envs/robridge/bin/python \
  loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py \
  --policy-path /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx \
  --headless --duration 30 --sim-dt 0.005 --control-decimation 4 \
  --initial-height 0.4567 --piper-delay-steps 4 \
  --command-velocity 0,0,0 --ee-pose 0.425,0,0.5,1,0,0,0
```

若已构建本仓库的 ROS2 工作空间，可运行:

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  headless:=true duration:=30.0 \
  python_executable:=/home/hh/anaconda3/envs/robridge/bin/python
```

启动前可执行只读检查（不会打开 CAN 或使能电机）：

```bash
/home/hh/anaconda3/envs/robridge/bin/python \
  /home/hh/loco_mani_ws/install/loco_mani_rl_controller/lib/loco_mani_rl_controller/validate_loco_mani_assets.py \
  --xml /home/hh/loco_mani_ws/install/loco_mani_description/share/loco_mani_description/mujoco/scene.xml \
  --policy /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx
```

也可以运行独立的离线 preflight（仍不会导入或打开 CAN）：

```bash
python3 install/loco_mani_rl_controller/lib/loco_mani_rl_controller/loco_mani_preflight.py \
  --check-policy
```

`--for-send` 只检查完整的 22 关节标定和 live safety gate，不会连接硬件；它会在
默认 `safety.dry_run: true` 时明确拒绝，避免把配置检查误认为实机许可。

还可以运行本地契约测试（只加载 MuJoCo，不连接硬件）：

```bash
/home/hh/anaconda3/envs/robridge/bin/python \
  /home/hh/loco_mani_ws/src/ddt_controller/scripts/test_loco_mani_contract.py
```

可通过 `policy_path`、`xml_path`、`command_velocity` 和 `ee_pose` 覆盖参数。
更换训练策略或调整控制器时，推荐直接修改
`loco_mani_rl_controller/config/d1_piper_l.yaml`：其中集中配置了策略 ABI、观测项/历史、
动作缩放、默认关节角、PD 增益、扭矩与速度限制、轮关节、DC 电机曲线、初始命令和仿真参数。
`sim2sim.launch.py` 默认加载该 YAML；命令行显式参数仍具有更高优先级。
若需要运行时用键盘调整 policy command，另开终端启动：

```bash
source /opt/ros/humble/setup.bash
source /home/hh/loco_mani_ws/install/setup.bash
ros2 launch loco_mani_bringup keyboard.launch.py
```

然后让仿真进程读取同一个命令文件：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  command_file:=/tmp/loco_mani_command.json \
  headless:=false real_time:=true duration:=0 \
  python_executable:=/home/hh/anaconda3/envs/robridge/bin/python
```

`duration:=0` 表示交互模式，仿真会一直运行到关闭 MuJoCo 窗口或按 `Ctrl-C`；若设置
为正数，达到该仿真时长后进程会正常退出。

键盘控制器会同时发布 `/command/cmd_twist` 和 `/command/cmd_pose`，并原子更新
`/tmp/loco_mani_command.json`。按键：`w/s` 调整前后速度，`a/d` 调整横向速度，
`q/e` 调整偏航速度；`j/l`、`u/o`、`i/k` 调整末端 x/y/z；`t/g`、`f/h`、`y/n`
分别调整末端 roll/pitch/yaw（单位为弧度，默认每次 0.05 rad，范围 ±0.6 rad）。
空格或 `r` 立即停止底盘，`0` 恢复末端默认位置和零姿态，`x` 退出。松开按键不会
自动减速，停止请按空格/r。姿态四元数写入命令文件时使用 `[qw,qx,qy,qz]` 顺序。
命令文件超过 0.5 s 没有心跳时，仿真读取器会把底盘速度置零。
双 CAN 真机接口的安全边界位于
`loco_mani_rl_controller/scripts/d1_piper_hardware.py`；本阶段不会由 launch
文件自动连接 `can0/can1` 或使能电机。先运行其 dry-run 配置检查，再按
`docs/implementation_plan.md` 的硬件 gate 顺序推进。

## 真机运行时（当前仅提供代码入口）

`hardware.launch.py` 启动的是独立双总线运行时，而不是旧的 D1
`ros2_control_node`。默认仍是 dry-run，不会打开任何 CAN：

如果 `pyAgxArm` 尚未安装到 Jetson 的 Python 环境，先把 SDK 源码加入该进程的
模块搜索路径（这一步不会连接 CAN）：

```bash
export PYTHONPATH=/home/hh/pyAgxArm:${PYTHONPATH:-}
```

```bash
ros2 launch loco_mani_bringup hardware.launch.py \
  duration:=5.0 \
  policy_path:=/home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx
```

若要在已有真机上做只读/shadow 测试，必须由现场人员确认两条总线和 CAN 比特率后，
显式传入 `shadow:=true hardware_gate:=true`，并设置 D1 适配器库路径。这会连接
`can0/can1`、读取状态并运行策略，但不会使能 Piper，也不会发送关节命令：

```bash
export LOCO_MANI_D1_ADAPTER_LIB=/path/to/libloco_mani_d1_vendor_adapter.so
ros2 launch loco_mani_bringup hardware.launch.py \
  shadow:=true hardware_gate:=true duration:=10.0
```

仓库已提供 `hardware/d1_vendor_adapter` 和 `d1_tita_adapter.py`；必须在目标
Jetson 上使用匹配的 ARM64 D1 头文件/库重新编译，不能把工作站 x86-64 动态库复制
过去。真实关节的 `bus_index/can_id/direction/position_offset_rad` 仍未填写，因此
不要使用 `send:=true`。正式发送还需要 `enable_piper:=true`、YAML 中的
`startup.enabled:=true`，并且必须先完成文档中的机械支撑、急停和单系统测试。
D1/Piper 真实发送还要求显式把 `safety.dry_run` 改为 `false`；默认配置保持
`safety.dry_run: true`，所以仅增加 launch 参数不会意外进入实机发送路径。
D1 主控是否已经处于直控模式不能靠猜测；现场若确认需要由软件切换，必须额外显式
传入 `d1_force_direct:=true`。该选项会发送 `SET_READY_NEXT=FORCE_DIRECT` RPC，默认
关闭，且只在 `send:=true` 时允许；shadow/只读模式永远不会切换 D1 主控状态。

若尚未完成标定，只读采样可使用：

```bash
export LOCO_MANI_D1_ADAPTER_LIB=/path/to/libloco_mani_d1_vendor_adapter.so
python3 install/loco_mani_rl_controller/lib/loco_mani_rl_controller/loco_mani_hardware_inspect.py \
  --config src/ddt_controller/loco_mani_rl_controller/config/d1_piper_l.yaml \
  --bus both --samples 20 --period 0.1 --hardware-gate \
  --output /tmp/d1_piper_feedback.json
```

该工具不会调用电机使能、MIT 运动、D1 发送或自动急停；`--hardware-gate` 只表示
人工允许打开 CAN，不能省略现场急停和机械支撑检查。

ROS 遥控输入可通过原有 `teleop_command` 节点加命令桥接转入运行时：

```bash
ros2 launch loco_mani_bringup command_bridge.launch.py
```

同一个 `/tmp/loco_mani_command.json` 只能由键盘节点或 command bridge 其中一个
进程写入，不能同时启动两个 writer。
使用 `policy_trace_path:=/tmp/mujoco_policy_trace.csv` 可输出策略频率下的
`frame/history/action/tau/q/dq`，然后用 `scripts/compare_policy_traces.py`
与 IsaacLab 导出的同列 trace 做逐元素比较。该 trace 是 sim2sim 对齐的主要
验收数据，不应只依据“机器人没有跌倒”判断策略迁移成功。
如果要复现 IsaacLab 的动作轨迹，可使用固定 raw action 回放（每行 22 个动作，
不需要再加载策略）：

```bash
python3 loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py \
  --action-replay /path/to/actions.npy --replay-rate-hz 50 \
  --command-replay /path/to/commands.npz \
  --initial-state /path/to/isaac_reset.json \
  --headless --duration 10
```

`actions.npy`/`actions.npz`、JSON 和 CSV 均支持；数组形状应为
`[N,22]`，CSV 也可以包含第一列时间戳。诊断 CSV 中名为 `action_*` 的 22 列
会被自动识别。回放轨迹短于仿真时长时，最后一行动作会保持不变，并在报告中
标记 `action_source=replay`。

在 IsaacLab 导出端增加 `--initial-state-output /tmp/isaac_reset.json
--actions-output /tmp/isaac_actions.npy --commands-output /tmp/isaac_commands.npz`，
再把这些文件同时传给 MuJoCo，
可以把 reset 随机化和首帧速度也对齐；否则只回放动作不能代表动力学等价。

可用 `observation_noise:=true` 复现训练时的观测扰动，并用
`observation_dump:=/tmp/reset.json` 导出复位帧、term-major 历史和策略元数据。
`initial_height` 可设为训练值 `0.45`，也可设为约 `0.4567` 以消除当前 MJCF
轮碰撞几何的初始穿透；`piper_delay_steps=4` 对应训练中的 Piper 最大 4 步延迟。
`passive_damping_scale` 默认 0、`passive_frictionloss_scale` 默认 1，使 XML 中
的执行器库仑摩擦生效而不重复叠加阻尼；可设为 0 做无摩擦消融。
控制器会输出 base 高度、倾角、接触数、扭矩饱和比例和末端位姿；`--stop-on-fall`
可用于自动停止异常 rollout，阈值可用 `--fall-height` 和 `--fall-tilt-deg`
覆盖（默认 0.25 m、35°）。初始根高、关节顺序、动作缩放和观测顺序均来自
训练日志中的 `params/env.yaml`，真机前仍需用实物标定值替换 Piper 安装位姿、
关节方向、零位和 CAN ID。当前版本已验证 ONNX/TorchScript 加载、22 DoF ABI、
term-major 历史观测和 30 s 零速 rollout 无 NaN，并提供接触、根高、倾角、扭矩
饱和诊断；这只能证明加载和推理链路打通，不能证明 sim2sim 或真机稳定。

训练命令的语义必须保留：`ee_pose` 的 `x/y` 是相对 `piper_base_link` 的坐标，
`z` 是世界高度；部署节点必须使用相同坐标系和 `[qw,qx,qy,qz]` 四元数约定。
