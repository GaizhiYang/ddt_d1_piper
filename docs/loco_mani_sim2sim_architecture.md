# D1 + Piper-L loco-manipulation sim2sim 代码框架与信息流

本文用于理解本仓库中 D1 四轮足机器人 + Piper-L 机械臂全身控制（WBC）策略
从加载到 MuJoCo 仿真执行的完整过程。重点是当前推荐的 ROS 2 入口：

```text
ros2 launch loco_mani_bringup sim2sim.launch.py
```

仓库还保留了一个不经过 ROS 2 的直接 Python 仿真入口。两种入口共享策略配置、
观测适配和动作语义，但运行时拓扑不同，不能混为同一个控制器。

---

## 1. 顶层架构

### 1.1 ROS 2 sim2sim（当前 launch 使用的路径）

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ros2 launch loco_mani_bringup sim2sim.launch.py                     │
└─────────────────────────────────────────────────────────────────────┘
                 │
                 ├── /mujoco_sim_ros2_node                           │
                 │       ├─ 加载 scene.xml                           │
                 │       ├─ 运行 MuJoCo 物理和 viewer                 │
                 │       ├─ MujocoRos2ControlPlugin                   │
                 │       └─ 内嵌 /controller_manager                   │
                 │              ├─ MuJoCo hardware interface           │
                 │              └─ 控制器 update/read/write            │
                 │
                 ├── /robot_state_publisher                           │
                 │       └─ URDF + /joint_states → /tf                 │
                 │
                 ├── 状态广播器                                        │
                 │       ├─ /joint_state_broadcaster                   │
                 │       │      hardware state → /joint_states         │
                 │       └─ /imu_sensor_broadcaster                     │
                 │              hardware IMU → /imu_sensor_broadcaster/imu│
                 │
                 ├── 五个 ForwardCommandController                     │
                 │       ├─ position  ← 22 维位置命令                   │
                 │       ├─ velocity  ← 22 维速度命令                   │
                 │       ├─ effort    ← 22 维前馈力矩                   │
                 │       ├─ kp        ← 22 维比例增益                   │
                 │       └─ kd        ← 22 维微分增益                   │
                 │
                 └── /d1_piper_loco_mani_policy                        │
                         ├─ 加载 ONNX                                 │
                         ├─ 订阅关节/IMU状态                           │
                         ├─ 订阅底盘和末端 command                     │
                         ├─ 构造 82 维观测 + 246 维历史                 │
                         ├─ 50 Hz 推理                                 │
                         └─ 500 Hz 发布五组关节命令                     │

 键盘（可选）：/loco_mani_keyboard
       ├─ /command/cmd_twist → policy
       └─ /command/cmd_pose  → policy
```

关键点：`/mujoco_sim_ros2_node` 和 `/controller_manager` 通常属于同一个
MuJoCo 进程；`/controller_manager` 不是另外启动的独立 `ros2_control_node`。

### 1.2 不经过 ROS 2 的直接 Python 仿真

```text
d1_piper_l_mujoco.py
    ├─ mujoco.MjModel.from_xml_path(scene.xml)
    ├─ Policy/OnnxPolicy 加载 policy.onnx
    ├─ 每个 MuJoCo 步：读取 q/dq/IMU → 构造观测
    ├─ 每 decimation 个物理步：执行一次策略
    ├─ 每个物理步：动作延迟、PD、扭矩限幅
    └─ mujoco.mj_step(model, data)
```

该路径不创建 ROS 节点、不发布 ROS 话题，也不使用 `controller_manager`。它适合
做最小化策略/动力学回放、动作 trace 对比和不依赖 ROS 的模型检查。

---

## 2. 目录与职责

```text
ddt_controller/
├── loco_mani_bringup/
│   └── launch/
│       ├── sim2sim.launch.py       # ROS 2 仿真总入口
│       ├── keyboard.launch.py      # 键盘 command 节点
│       └── hardware.launch.py       # 真机/干运行入口（默认不发送）
│
├── loco_mani_description/
│   ├── xacro/robot.xacro            # ROS robot_description 入口
│   ├── xacro/ros2control.xacro      # 独立 ros2_control 定义
│   ├── urdf/robot.urdf              # 组合后的运动学/惯性描述
│   └── mujoco/
│       ├── scene.xml                # 地面、视觉和 robot.xml 的入口
│       └── robot.xml                # MuJoCo 机体、关节、传感器、执行器
│
├── loco_mani_rl_controller/
│   ├── config/d1_piper_l.yaml       # 策略和控制 ABI 配置
│   ├── config/ros2_controllers.yaml # ros2_control 控制器参数
│   └── scripts/
│       ├── d1_piper_l_ros2_node.py  # ROS 2 策略节点
│       ├── d1_piper_l_mujoco.py     # 非 ROS 直接仿真控制器
│       ├── d1_piper_hardware.py     # 共享观测/动作/CAN适配层
│       ├── loco_mani_config.py      # YAML 解析和默认配置
│       ├── loco_mani_runtime.py     # 双 CAN 真机运行时骨架
│       ├── command_file.py           # JSON command 兼容桥
│       └── loco_mani_keyboard.py    # 键盘 command 发布器
│
├── simulation/mujoco_bridge/
│   ├── mujoco_sim_ros2/              # MuJoCo viewer/物理主程序
│   └── mujoco_ros2_control/          # MuJoCo hardware + ros2_control 插件
│
└── scripts/
    ├── check_policy_abi.py           # 检查 ONNX 输入/输出形状
    ├── validate_loco_mani_assets.py   # 检查模型、关节和执行器 ABI
    ├── export_isaaclab_trace.py      # 导出 IsaacLab 对齐 trace
    └── compare_policy_traces.py      # 比较 IsaacLab/MuJoCo trace
```

### 2.1 文件之间的依赖关系

```text
d1_piper_l.yaml
       ↓ load_resolved_config()
loco_mani_config.py
       ├─ PolicyConfig（频率、历史、增益、限值、命令）
       ├─ ObservationAdapter（硬件/ROS观测）
       └─ ActionAdapter（原始动作 → 关节命令）

robot.xacro ─ includes ─ ros2control.xacro
       ├─ robot_description 给 robot_state_publisher
       └─ ros2_control hardware/joint/sensor ABI 给 MuJoCo 插件

scene.xml ─ includes ─ robot.xml
       ├─ MuJoCo 几何、质量、关节、执行器
       └─ trunk_imu、接触和传感器数据
```

---

## 3. 启动时序（ROS 2 sim2sim）

`sim2sim.launch.py` 不是一次性无序启动所有节点，而是通过进程事件串联控制器
加载。默认 `start_paused:=true`，用于避免控制器尚未激活时 MuJoCo 已经运行并跌倒。

```text
1. 启动 /mujoco_sim_ros2_node
   └─ 加载 scene.xml，创建 MuJoCo 模型和 plugin

2. plugin 初始化 /controller_manager
   ├─ 获取 robot_description
   ├─ 解析 ros2_control 硬件和 22 个关节接口
   ├─ 创建 MujocoSystem
   └─ 创建 ControllerManager executor 线程

3. 启动 /robot_state_publisher

4. MuJoCo 进程启动后，spawner 加载并激活：
   joint_state_broadcaster
       ↓
   imu_sensor_broadcaster
       ↓
   loco_mani_position_controller
       ↓
   loco_mani_velocity_controller
       ↓
   loco_mani_effort_controller
       ↓
   loco_mani_kp_controller
       ↓
   loco_mani_kd_controller

5. kd spawner 成功退出后，启动 /d1_piper_loco_mani_policy

6. policy 节点启动后，launch 将 start_paused 设置为 false

7. MuJoCo 开始推进仿真，策略和 ros2_control 进入闭环
```

`spawner_*` 是启动阶段的临时辅助进程，控制器激活后自动退出，所以它们通常不会
出现在最终的 `ros2 node list` 中。启动期间任一关键进程失败，launch 会关闭整套图。

`duration:=0` 表示不按仿真时间自动结束，直到关闭 viewer 或按 `Ctrl-C`；正数表示
达到指定 MuJoCo 仿真秒数后退出。

---

## 4. 策略加载过程

### 4.1 配置文件解析

ROS 2 策略节点收到 `--config` 后调用：

```python
config = load_resolved_config(args.config)
```

默认文件是：

```text
loco_mani_rl_controller/config/d1_piper_l.yaml
```

配置解析器把 YAML 解析为 `PolicyConfig`，并验证固定的 D1 + Piper-L 策略 ABI：

| 项目 | 当前约定 |
|---|---|
| 活动关节 | D1 16 个 + Piper 6 个，共 22 个 |
| 关节顺序 | FL、FR、RL、RR，每条腿 hip/thigh/calf/foot，随后 joint1..6 |
| 单帧观测 | 82 维 |
| 历史长度 | 3 |
| 历史展开 | term-major，246 维 |
| 策略频率 | 50 Hz |
| 物理步长 | 0.005 s |
| 控制降采样 | 4，即 50 Hz 策略对应 200 Hz 物理控制更新基准 |
| ONNX 输出 | 22 维 raw action |

这里要区分三个“频率”：YAML 中的策略频率是 50 Hz；`d1_piper_l_ros2_node.py`
默认以 500 Hz 的墙钟定时器重复发布最近一次命令；MuJoCo 当前 `sim_dt=0.005`
意味着物理线程每秒推进 200 个仿真步。`ros2_controllers.yaml` 中
`controller_manager.update_rate` 虽配置为 500，但嵌入式插件的
`read/update/write` 是由 MuJoCo physics hook 驱动的，实际调用频率还受
`sim_dt` 和插件调度限制。因此调试延迟时应同时查看策略发布频率、物理步频率和
ControllerManager 日志，不能只看 YAML 的一个数值。

如果 YAML 提供 `policy.sha256`，`OnnxPolicy` 会先计算文件 SHA256，校验不匹配时
拒绝运行，避免误加载另一份 checkpoint。

### 4.2 ONNX 输入输出

`OnnxPolicy` 使用 ONNX Runtime 和 `CPUExecutionProvider`。当前导出的策略通常为：

```text
input:  obs[1, 246]  float32
output: actions[1, 22] float32
```

代码也兼容单输入 82 维或旧式双输入（当前帧 + 历史）的导出，但最终动作必须是
22 个有限浮点数。输入和输出的宽度检查在启动/首次推理阶段完成。

### 4.3 Python 解释器注意事项

ROS 2 Humble 的 `rclpy` 是 Python 3.10 ABI。launch 会检查用户传入的
`python_executable`；如果传入 Python 3.9 环境，可能会自动回退到运行 launch 的
Python 3.10 解释器。回退后的解释器仍必须能导入：

```text
rclpy、numpy、yaml、onnxruntime
```

这项检查只解决 Python 扩展 ABI 问题，不会自动安装缺少的依赖。

---

## 5. 状态信息流：MuJoCo 到策略

### 5.1 MuJoCo 硬件接口读取

MuJoCo 插件中的 `MujocoSystem::read()` 每次 ControllerManager 更新时读取：

```text
MuJoCo mjData
  ├─ qpos       → 关节 position
  ├─ qvel       → 关节 velocity
  ├─ qfrc_applied → 关节 effort/torque
  └─ sensordata
       ├─ trunk_quat
       ├─ trunk_gyro
       └─ trunk_accel
```

MuJoCo 模型中的 `trunk_imu` site 对应 ros2_control 中的 `trunk_imu` sensor。
IMU 四元数在 MuJoCo/硬件接口内部转换为 ROS 消息字段 `orientation.x/y/z/w`。

### 5.2 状态广播器发布

`joint_state_broadcaster` 直接从 ControllerManager 的 hardware state interface
获取 22 个关节状态，发布：

```text
/joint_states                  sensor_msgs/msg/JointState
/dynamic_joint_states          control_msgs/msg/DynamicJointState
```

`imu_sensor_broadcaster` 直接读取 `trunk_imu` 状态接口，发布：

```text
/imu_sensor_broadcaster/imu     sensor_msgs/msg/Imu
```

这两个 broadcaster 通常没有业务层普通订阅者；它们的输入是 ControllerManager
在 update 周期中提供的硬件接口。

### 5.3 robot_state_publisher 的旁路

`robot_state_publisher` 订阅 `/joint_states`，用 `robot_description` 中的 URDF
计算 link 之间的 TF：

```text
/joint_states + robot_description
              ↓
       /robot_state_publisher
              ↓
       /tf、/tf_static
```

它只负责 TF，不参与策略推理、关节命令或 MuJoCo 动力学。

### 5.4 策略节点的状态缓存

`d1_piper_l_ros2_node.py` 的 `D1PiperRos2Policy` 使用回调把数据写入线程锁保护的
`StateSnapshot`：

```text
/joint_states
  └─ 按 joint name 放入 q、dq、tau

/imu_sensor_broadcaster/imu
  ├─ ROS xyzw → 策略使用的 wxyz
  ├─ angular_velocity → gyro
  └─ linear_acceleration → accel
```

只有当 22 个关节全部出现且收到 IMU 后，`snapshot.valid` 才为真。状态缺失或
超过 `command_timeout`（默认 0.2 s）时，策略节点进入零命令模式。

---

## 6. Command 信息流：键盘/外部命令到策略

策略有两类 command 输入：底盘速度和机械臂末端目标位姿。

### 6.1 ROS 话题接口

| 输入 | 消息类型 | 含义 |
|---|---|---|
| `/command/cmd_twist` | `geometry_msgs/msg/Twist` | `linear.x`、`linear.y`、`angular.z` 对应 vx、vy、yaw rate |
| `/command/cmd_pose` | `geometry_msgs/msg/PoseStamped` | position xyz + 四元数 |

策略节点把 ROS Pose 的 `x,y,z,w` 字段重新排列为策略约定的：

```text
[x, y, z, qw, qx, qy, qz]
```

键盘节点 `/loco_mani_keyboard` 发布上述两个话题，同时原子写入：

```text
/tmp/loco_mani_command.json
```

ROS 话题在策略节点中优先级更高；JSON 文件是跨 Python 环境和非 ROS 控制器的
兼容桥。当 ROS command 长时间没有更新时，策略节点才回退读取文件。

### 6.2 Command 坐标约定

这些约定必须与训练环境保持一致：

```text
ee_pose = [x, y, z, qw, qx, qy, qz]
```

当前配置中，x/y 按 `piper_base_link` 参考，z 使用训练约定的世界高度，四元数为
标量在前的 `[qw,qx,qy,qz]`。如果真机或新的训练任务改变参考坐标系，必须同步修改
训练、YAML、键盘节点和硬件适配器，不能只改显示值。

---

## 7. 82 维观测和 246 维历史

`ObservationAdapter` 和 MuJoCo 直接控制器使用相同的观测语义。单帧按以下顺序
拼接：

```text
frame[82] = [
  base_ang_vel(3),
  projected_gravity(3),
  joint_pos(22),
  joint_vel(22),
  last_action(22),
  command_velocity(3),
  ee_pose(7)
]
```

其中：

1. `base_ang_vel` 使用机体坐标系角速度，默认乘以 0.2 观测缩放；
2. `projected_gravity` 由基座四元数将世界重力投影到机体坐标系；
3. `joint_pos` 是当前关节角减去 `default_joint_angles`，四个轮子的该项强制为 0；
4. `joint_vel` 是 22 个关节速度；
5. `last_action` 是上一时刻策略 raw action，不是已经缩放后的目标角；
6. 最后三项是 command，直接进入观测。

历史不是简单的“3 个完整 frame 首尾相接”，而是按观测 term 分开缓存：

```text
history[246] = [
  base_ang_vel 的 3 帧,
  projected_gravity 的 3 帧,
  joint_pos 的 3 帧,
  joint_vel 的 3 帧,
  last_action 的 3 帧,
  command_velocity 的 3 帧,
  ee_pose 的 3 帧
]
```

第一次写入历史时，每个 term 用当前样本复制 3 次，`last_action` 从零开始。这一
细节是 ONNX 能否得到与 IsaacLab 相同输入的关键。

---

## 8. 策略动作到关节命令

### 8.1 原始动作的含义

ONNX 输出 `raw_action[22]` 先经过动作限幅，然后使用 YAML 中的动作缩放和默认角：

```text
target_q = default_q + raw_action * action_scale
```

对 16 个腿部关节和 6 个 Piper 关节，动作顺序始终与策略 joint order 一致。

四个轮关节是特殊项：

```text
wheel position target = 当前测量角度
wheel velocity target = raw_action[wheel] * action_scale[wheel]
wheel kp = 0
```

这样轮子执行速度控制，不会因为位置误差产生意外刹车。

### 8.2 ROS 2 五路命令

ROS 策略节点把同一个 `JointCommand` 拆成五个 22 维数组，分别发布：

```text
/loco_mani_position_controller/commands  Float64MultiArray
/loco_mani_velocity_controller/commands  Float64MultiArray
/loco_mani_effort_controller/commands    Float64MultiArray
/loco_mani_kp_controller/commands         Float64MultiArray
/loco_mani_kd_controller/commands         Float64MultiArray
```

当前 WBC 动作主要由位置/速度目标 + kp/kd 构成，`effort` 通常为零前馈力矩。
策略节点大约每 50 Hz 重新推理，并以约 500 Hz 重复发布最近一次命令，以满足
ros2_control 和低层执行器的更新频率。

### 8.3 ros2_control 控制器的作用

五个 `ForwardCommandController` 不计算控制律，只完成：

```text
ROS Float64MultiArray
        ↓
ControllerManager command interface
        ↓
MujocoSystem 的 position/velocity/effort/kp/kd command
```

每个数组必须包含 22 个元素，顺序必须与 `ros2_controllers.yaml` 中的 joint 列表
和策略配置一致。

---

## 9. MuJoCo 中的最终力矩和物理推进

### 9.1 ControllerManager 的周期

MuJoCo 插件在仿真线程中执行：

```text
pre_update()
  └─ 发布 /clock

update_with_step()
  ├─ mj_step1(model, data)
  ├─ ControllerManager.read(sim_time, period)
  ├─ ControllerManager.update(sim_time, period)
  ├─ ControllerManager.write(sim_time, period)
  └─ mj_step2(model, data)
```

ControllerManager 的 `read/update/write` 让 broadcaster 和 ForwardCommandController
与 MuJoCo hardware interface 保持 ros2_control 语义。

### 9.2 MujocoSystem::write 的控制律

对每个关节，MuJoCo hardware interface 将命令合成为：

```text
tau = effort
    + kp * (position_target - position)
    + kd * (velocity_target - velocity)
```

然后根据 ros2_control 关节/命令接口的上下限进行限幅，并写入：

```text
mj_data->qfrc_applied[dof] = tau
```

随后 `mj_step2()` 使用该力矩完成约束、接触和动力学积分，更新下一时刻的
`qpos/qvel/sensordata`。这就闭合了：

```text
状态 → 观测 → ONNX → 动作 → ros2_control → tau → MuJoCo → 新状态
```

注意：ROS 2 这条路径的 `MujocoSystem::write()` 不会读取
`d1_piper_l.yaml` 中的 DC 电机速度曲线；它执行的是通用的 effort + PD 合成和
ros2_control 接口限幅。DCMotor 速度-力矩曲线、Piper 延迟队列以及更完整的动作
限幅逻辑，位于不经过 ROS 2 的 `d1_piper_l_mujoco.py` (`D1PiperRollout`) 中。
如果需要让 ROS 2 sim2sim 与直接 Python 回放逐项一致，应把这些执行器细节显式
实现到 MuJoCo ros2_control hardware interface，或在策略命令进入控制器前增加
等价的适配层，不能仅凭两条路径都能站立就认为动力学完全相同。

### 9.3 时间来源

MuJoCo 插件发布 `/clock`，ControllerManager 被强制设置为 `use_sim_time=true`。
策略节点的状态 watchdog 使用墙钟/单调时钟，因而即使 MuJoCo 暂停，状态和命令
超时保护仍能工作。

---

## 10. 两个策略控制器的区别

| 项目 | `d1_piper_l_ros2_node.py` | `d1_piper_l_mujoco.py` |
|---|---|---|
| 启动方式 | `sim2sim.launch.py` | 直接 `python ...d1_piper_l_mujoco.py` |
| ROS 节点 | 有 | 无 |
| ONNX | `OnnxPolicy`/`HardwarePolicyRuntime` | 本文件中的 `Policy` + `D1PiperRollout` |
| 状态来源 | `/joint_states`、`/imu.../imu` | 直接读 `mjData` |
| 命令来源 | ROS topic，必要时 JSON | 参数、command replay 或 JSON |
| 命令输出 | 5 个 ForwardCommandController 话题 | 直接写 `mj_data->qfrc_applied` |
| ros2_control | 使用 | 不使用 |
| 主要用途 | 验证 ROS 图、控制器连接和在线 command | 策略/动力学回放、trace 对齐和最小调试 |

两条路径应尽量使用同一个 YAML 和同一套策略 ABI。若两者结果不同，应优先比较
观测 frame/history、raw action、q/dq 和 tau trace，而不是只观察 viewer 中是否站稳。

---

## 11. 真机代码目前处于什么位置

`loco_mani_runtime.py` 和 `d1_piper_hardware.py` 已把真机所需的抽象拆出来，但
当前 `sim2sim.launch.py` 不会打开 `can0` 或 `can1`。真机目标结构是：

```text
can0（D1，16关节+IMU） ─┐
                       ├─ StateSnapshot → policy → JointCommand
can1（Piper，6关节） ──┘                         │
                                                 ├─ D1 send
                                                 └─ Piper MIT send
```

真机运行时还需要完成 CAN ID、关节顺序、方向、零位、固件、单位、限位和急停策略的
现场确认。默认 `dry_run`/`send=false`/`hardware_gate=false` 是故意保留的安全门，
不能因为 sim2sim 已经能运行就直接打开真实电机发送。

---

## 12. 推荐的理解和调试顺序

### 第一步：确认模型和策略 ABI

```bash
cd /home/hh/loco_mani_ws/src/ddt_controller
PY=/home/hh/anaconda3/envs/robridge/bin/python
POLICY=/home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx

$PY scripts/check_policy_abi.py "$POLICY"
$PY scripts/validate_loco_mani_assets.py \
  --xml loco_mani_description/mujoco/scene.xml \
  --policy "$POLICY"
```

### 第二步：只观察 ROS 图，不改变 command

```bash
ros2 node list
ros2 control list_controllers
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic echo /imu_sensor_broadcaster/imu --once
```

### 第三步：观察策略输入和输出

使用 `diagnostics_path`、`policy_trace_path` 保存策略频率和物理频率数据，再比较：

```text
frame/history → action → tau → q/dq
```

### 第四步：最后才加入键盘 command

```bash
ros2 launch loco_mani_bringup keyboard.launch.py
```

先按空格确认底盘速度为零，再小幅增加 vx、vy、yaw 和末端位姿。松开按键不会
自动减速，停止底盘应按空格或 `r`。

---

## 13. 一句话总结

本项目的 ROS 2 sim2sim 闭环可以概括为：

```text
scene.xml / robot.xml
  → MuJoCo mjData
  → MujocoSystem hardware interface
  → joint_state_broadcaster + imu_sensor_broadcaster
  → d1_piper_loco_mani_policy
  → 82-D frame + 246-D history
  → ONNX policy
  → 22-D action
  → position/velocity/effort/kp/kd 五路 ForwardCommandController
  → MujocoSystem PD/限幅
  → mj_step2()
  → 下一时刻 MuJoCo 状态
```

只要关节顺序、四元数顺序、坐标系、历史布局、动作缩放和控制周期保持一致，
策略加载和仿真控制链路就是可追踪、可测试、可逐项对齐的。
