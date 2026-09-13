# DDT 机器人 sim2sim/sim2real
本仓库是一个基于 ROS 2 的多包工作空间，包含机器人控制器、硬件桥接、仿真桥接（`Mujoco` / `Gazebo` / `Webots`）、交互控制以及机器人模型描述。控制器支持基于有限状态机的策略，并可加载 ONNX 强化学习模型进行推理。
同时提供 [Docker 镜像与启动说明](./docker/README.md)，将依赖与配置固化为可复现环境，便于快速部署、复现实验与跨设备一致运行（sim2sim）。

## D1 + Piper-L loco-manipulation 状态

`loco_mani_description` 、`loco_mani_rl_controller` 和 `loco_mani_bringup` 是针对 D1 + Piper-L WBC 策略的第一阶段包。已验证给定 ONNX 的 `246 -> 22` ABI、观测历史、动作缩放和 MuJoCo 推理循环。可行性结论与验收门槛见 [docs/feasibility_assessment.md](docs/feasibility_assessment.md)，训练配置与当前 MJCF 的差异见 [docs/sim2sim_audit.md](docs/sim2sim_audit.md)。

推荐的双 CAN、实时循环、标定和上电验收顺序见 [docs/implementation_plan.md](docs/implementation_plan.md)。

**重要：当前版本是接口/标定验证骨架，不是可直接上电的真机控制器。** 当前简化 MJCF 中 Piper 碰撞、PhysX/MuJoCo 接触参数和 DCMotor 摩擦尚未完成对齐；现有 `hardware_bridge` 仅支持 D1 单总线，不能用于 `16 + 6` 关节的 `can0/can1` 双总线控制。

### ROS 2 管理架构

`loco_mani_bringup/sim2sim.launch.py` 采用与原始
`controller/rl_controller/launch/sim_mujoco.launch.py` 相同的 ROS 2
`launch_ros.actions.Node` + `controller_manager/spawner` 生命周期：MuJoCo 进程
内嵌 `mujoco_ros2_control` 的 `controller_manager`，launch 依次加载状态广播器、
IMU 广播器和五个 `ForwardCommandController`；最后启动独立的
`d1_piper_loco_mani_policy` ROS 2 节点。策略节点通过 ROS 话题接收状态并发布
五组命令数组，因此可以用 `ros2 node list`、`ros2 topic echo` 和
`ros2 control list_controllers` 调试，且 launch 会在任一关键进程失败时关闭整套
图。该方案不会自动打开 `can0`/`can1`。

常驻节点及控制器为：

```text
/mujoco_sim_ros2_node                 MuJoCo 仿真（内嵌 /controller_manager）
/controller_manager                   ros2_control 管理器
/robot_state_publisher                URDF TF
/joint_state_broadcaster             /joint_states
/imu_sensor_broadcaster              /imu_sensor_broadcaster/imu
/loco_mani_position_controller       22 维 position 命令
/loco_mani_velocity_controller       22 维 velocity 命令
/loco_mani_effort_controller         22 维 effort 命令
/loco_mani_kp_controller              22 维 kp 命令
/loco_mani_kd_controller              22 维 kd 命令
/d1_piper_loco_mani_policy            ONNX 观测/推理/命令节点
```

`spawner_*` 进程只是启动阶段的临时辅助进程，加载并激活控制器后自动退出。
启动后可检查：

```bash
ros2 node list
ros2 control list_controllers
ros2 topic list
ros2 topic echo /joint_states --once
```

ROS 2 Humble 的 `rclpy` 是 Python 3.10 C 扩展。launch 会检查
`python_executable` 的小版本；如果传入 Python 3.9（例如旧的 `robridge` 环境），
会自动改用运行 launch 的 Python 3.10，避免 `_rclpy_pybind11` ABI 错误。该解释器
必须同时能导入 `numpy`、`yaml` 和 `onnxruntime`；在本机建议先验证：

```bash
/usr/bin/python3 -c 'import rclpy,numpy,yaml,onnxruntime; print("ROS policy runtime OK")'
```

已提供的策略文件：

```text
/home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx
```

## D1 + Piper-L sim2sim 快速验证

下面的命令只启动 MuJoCo 仿真，不会打开 `can0`/`can1`，也不会使能或发送
任何真实电机命令。策略使用 Jetson 上同样可以采用的 ONNX Runtime；工作站上
这里统一使用已经安装好 `mujoco`、`numpy` 和 `onnxruntime` 的
`robridge` Python 环境。

### 1. 编译 ROS 2 功能包

```bash
cd /home/hh/loco_mani_ws
source /opt/ros/humble/setup.bash

colcon build \
  --packages-select \
    loco_mani_description \
    loco_mani_rl_controller \
    loco_mani_bringup \
    loco_mani_d1_vendor_adapter \
  --symlink-install

source /home/hh/loco_mani_ws/install/setup.bash
```

如果只想运行 Python 版 MuJoCo 控制器，不需要 ROS 2；但仍建议先完成上面的
构建，以检查描述文件和 launch 文件是否完整。

### 2. 启动并加载 D1 + Piper-L 仿真模型

推荐通过 ROS 2 launch 启动（默认 30 s、无头模式、零速度指令）：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  headless:=true \
  duration:=30.0 \
  python_executable:=/usr/bin/python3
```

启动时会加载以下模型和策略：

```text
模型：/home/hh/loco_mani_ws/install/loco_mani_description/share/
      loco_mani_description/mujoco/scene.xml
策略：/home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/
      2026-08-10_16-51-00/exported/policy.onnx
```

需要 GUI 时，将 `headless` 设为 `false`（必须在有图形显示的环境运行）：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  headless:=false duration:=30.0 real_time:=true \
  python_executable:=/usr/bin/python3
```

### 2a. 仅查看机器人模型（不加载控制器）

如果只想检查 URDF/MJCF 的机器人结构、关节、网格和安装位姿，不要使用上面的
`sim2sim.launch.py`，因为它会加载 ONNX 策略并运行控制循环。请使用模型查看器：

```bash
cd /home/hh/loco_mani_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

/home/hh/anaconda3/envs/robridge/bin/python \
  install/loco_mani_description/lib/loco_mani_description/view_mujoco_model.py
```

该命令只解析并显示 `scene.xml`，不会加载 ONNX、不会运行策略控制器、不会执行
动力学步进，也不会打开 `can0`/`can1`。关闭窗口即可退出。若希望自动打开 60 秒：

```bash
/home/hh/anaconda3/envs/robridge/bin/python \
  install/loco_mani_description/lib/loco_mani_description/view_mujoco_model.py \
  --duration 60
```

也可以直接指定源代码中的模型：

```bash
/home/hh/anaconda3/envs/robridge/bin/python \
  loco_mani_description/scripts/view_mujoco_model.py \
  --xml-path loco_mani_description/mujoco/scene.xml
```

默认仿真按最快速度运行，因此 `duration:=10.0` 通常会在几秒内结束并自动关闭
窗口；这属于正常行为。若希望按接近真实时间运行，可增加 `real_time:=true`：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  headless:=false real_time:=true duration:=10.0 \
  python_executable:=/home/hh/anaconda3/envs/robridge/bin/python
```

仿真结束时程序会输出 `finished t=...` 后自动退出。若旧版本在 GUI 退出瞬间出现
`exit code -11`、`GLXBadDrawable` 或 `malloc(): ... corrupted`，请重新构建本仓库；
当前 MuJoCo 脚本已加入 viewer 线程的显式关闭和短暂等待，避免 GLFW 清理竞态。

### 3. 不经过 ROS 2 的直接仿真命令

```bash
cd /home/hh/loco_mani_ws/src/ddt_controller
/home/hh/anaconda3/envs/robridge/bin/python \
  loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py \
  --xml-path loco_mani_description/mujoco/scene.xml \
  --policy-path /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx \
  --headless --duration 30 \
  --sim-dt 0.005 --control-decimation 4 \
  --command-velocity 0,0,0 \
  --ee-pose 0.425,0,0.5,1,0,0,0
```

`ee_pose` 的格式为 `x,y,z,qw,qx,qy,qz`：其中 `x/y` 是相对
`piper_base_link` 的坐标，`z` 是世界坐标高度，四元数顺序是 `[qw,qx,qy,qz]`。
适配新训练策略时，优先编辑
`loco_mani_rl_controller/config/d1_piper_l.yaml`。该文件现在集中管理类似原
`controller/rl_controller/config/d1/controllers.yaml` 的重要参数：策略输入输出 ABI、
观测项及历史布局/噪声、默认关节角、动作缩放、关节 `kp/kd`、扭矩/速度限制、轮关节、
DC 电机速度曲线、初始命令、仿真步长/降采样和安全相关运行参数。显式命令行参数仍会覆盖
YAML 值。
例如，测试底盘前进和末端目标可以运行：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  command_velocity:=0.2,0,0 \
  ee_pose:=0.45,0,0.50,1,0,0,0 \
  duration:=10.0 headless:=true \
  python_executable:=/home/hh/anaconda3/envs/robridge/bin/python
```

### 3a. 键盘实时调整 policy command

当前 `sim2sim.launch.py` 的固定 `command_velocity`/`ee_pose` 参数不能在运行中由
键盘修改。仓库新增 `loco_mani_keyboard` 节点：它发布标准 ROS 话题，同时通过
原子 JSON 文件与独立 Python 控制器通信。

终端 A 启动键盘：

```bash
source /opt/ros/humble/setup.bash
source /home/hh/loco_mani_ws/install/setup.bash
ros2 launch loco_mani_bringup keyboard.launch.py
```

终端 B 启动仿真并读取该文件：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  command_file:=/tmp/loco_mani_command.json \
  headless:=false real_time:=true duration:=0 \
  python_executable:=/home/hh/anaconda3/envs/robridge/bin/python
```

键盘联动建议使用 `duration:=0` 交互模式；仿真将持续运行，直到关闭 MuJoCo 窗口或按
`Ctrl-C`。设置正数 `duration` 时，达到指定仿真秒数后会自动正常退出。

按键：`w/s` 前后、`a/d` 横向、`q/e` 偏航；`j/l`、`u/o`、`i/k` 调整末端 x/y/z；
`t/g`、`f/h`、`y/n` 分别调整末端 roll/pitch/yaw（每次默认 0.05 rad，范围 ±0.6 rad）。
空格或 `r` 停止底盘，`0` 恢复末端默认位置和零姿态，`x` 退出。命令文件超过 0.5 s 没有
更新时，控制器会自动将底盘速度置零。该节点默认只发布命令，不打开 CAN；真机
仍需保持现有 `send:=false` 和 `hardware_gate:=false` 安全门。

末端姿态命令在 ROS 话题和 JSON 文件中均使用 `[qw,qx,qy,qz]` 四元数顺序；键盘节点
内部以 XYZ 固定轴 RPY 累积，并在每次更新后归一化。可通过 `keyboard.launch.py` 的
`orientation_step`（默认 `0.05`）和 `orientation_limit`（默认 `0.6`）调整步长与限幅。

无论命令来自键盘、ROS 话题还是 JSON 文件，策略入口都会再次执行统一限幅：底盘速度
使用 `commands.velocity_limits`，末端位置使用 `commands.ee_pose.position_limits`，
姿态使用 `orientation_limits_rpy`。因此外部节点不能绕过训练范围直接把异常目标送入
策略或真机执行器。

### 4. 启动前的只读检查

这些命令只检查模型、策略 ABI 和软件契约，不访问任何 CAN 设备：

```bash
cd /home/hh/loco_mani_ws/src/ddt_controller
PY=/home/hh/anaconda3/envs/robridge/bin/python
POLICY=/home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx

$PY scripts/check_policy_abi.py "$POLICY"
$PY scripts/validate_loco_mani_assets.py \
  --xml loco_mani_description/mujoco/scene.xml \
  --policy "$POLICY"
$PY scripts/test_loco_mani_contract.py
```

正常结果应分别显示 `input=obs[1, 246] output=actions[1, 22]`、资源检查通过，
以及全部契约测试 `OK`。

### 5. 输出诊断和复现动作

为了检查高度、倾角、接触数、扭矩饱和率和末端位姿，可保存诊断 CSV；策略频率
的 `frame/history/action/q/dq/tau` 则保存到 trace CSV：

```bash
ros2 launch loco_mani_bringup sim2sim.launch.py \
  headless:=true duration:=10.0 \
  diagnostics_path:=/tmp/d1_piper_mujoco.csv \
  policy_trace_path:=/tmp/d1_piper_policy_trace.csv \
  python_executable:=/home/hh/anaconda3/envs/robridge/bin/python
```

若要把 IsaacLab 导出的固定动作在 MuJoCo 中回放，使用：

```bash
/home/hh/anaconda3/envs/robridge/bin/python \
  loco_mani_rl_controller/scripts/d1_piper_l_mujoco.py \
  --action-replay /tmp/isaac_actions.npy \
  --command-replay /tmp/isaac_commands.npz \
  --initial-state /tmp/isaac_reset.json \
  --headless --duration 10
```

当前仿真通过“不跌倒”只能说明推理链路可运行；在进入任何真机 CAN 测试前，还需
完成 IsaacLab trace 对齐、碰撞/摩擦标定、机械臂安装位姿确认和文档中列出的安全
验收步骤。真机 launch 的 `send` 默认是 `false`，请不要为了仿真测试打开它。

## 主要功能
- 强化学习控制器（支持 ONNX 推理），基于有限状态机组织控制逻辑
- `ros2_control` 硬件桥接，连接真实机器人驱动库
- `Mujoco` / `Gazebo` / `Webots` 三种仿真环境桥接与示例世界
- 键盘控制与遥控（ELRS）交互模块
- 多机器人模型与描述（`tita`、`d1`(四轮足)、`d1h`(双轮足)）

## 目录结构
- `controller/rl_controller`：基于`ros2_control`框架强化学习控制器
- `hardware`：`hardware_bridge`为`ros2_control` 硬件桥接节点与启动文件，`tita_robot`真实机器人底层驱动接口
- `interaction`：`keyboard_controller`键盘交互节点，`teleop_command`遥控手柄交互节点
- `simulation`：`Mujoco`、`Gazebo`、`Webots` 仿真桥接
- `ros_utils`: 主要为`ros`话题名称相关
- `urdfs`：机器人模型描述文件（`URDF`/`XACRO`/`Mujoco`等）

## 注意事项
在使用hardware之前，请检查当前d1-ros2 软件版本,使用`dpkg -l d1-ros2`指令检查，如果当前软件版本为4月01号的，则使用`compress_v1`分支代码，切记勿使用`main`分支，否则失控，同时使用前注意安全。以下`compress_v1`分支使用方法：
```bash
git clone https://github.com/DDTRobot/ddt_ros2_control/tree/compress_v1
cd ddt_ros2_control
#在编译hardware_bridge前，source /opt/d1-ros2/setup.bash
source /opt/d1_ros2/setup.bash 
#确保前后机无其他ros2节点在运行，然后启动硬件运控服务
colcon build --symlink-install --packages-up-to rl_controller hardware_bridge
sudo systemctl stop d1_bringup.service
ros2 launch rl_controller hw.launch.py robot:=d1

```
## 环境与依赖
- 安装onnx推理引擎
``` bash 
# 根据你的系统架构选择x64或者aarch64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.10.0/onnxruntime-linux-x64-1.10.0.tgz
tar xvf onnxruntime-linux-x64-1.10.0.tgz
sudo cp -a onnxruntime-linux-x64-1.10.0/include/* /usr/include
sudo cp -a onnxruntime-linux-x64-1.10.0/lib/* /usr/lib
```
- Ubuntu 22.04, ros2 humble, gazebo classic, webots R2025a
安装好ros2 humble后，安装以下依赖：
```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
```
- 根据需要安装仿真环境所需的依赖
1. `webots`
```bash
sudo apt install ros-humble-webots-ros2 ros-humble-webots-ros2-control
```
2. `gazebo`
```bash
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-ros2-control
```
3. `mujoco`
见下方构建部分

## 构建
以下命令展示了所有可行的编译命令，根据你的实际需要选择必要的组件进行编译。
```bash
# 创建并进入工作空间
mkdir -p ~/ddt_ros2_ws && cd ~/ddt_ros2_ws
# 将本仓库放置于 ~/ddt_ros2_ws/
mv ddt_ros2 src
# 如需使用mujoco，执行下方
git clone -b 3.3.0 https://github.com/google-deepmind/mujoco.git
# 构建
cd ~/ddt_ros2_ws
# 编译rl_controller
colcon build --symlink-install --packages-up-to rl_controller 
# 编译仿真环境
colcon build --symlink-install --packages-up-to webots_bridge # 可替换为gazebo_bridge， mujoco_bridge
# 编译机器人模型描述
colcon build --symlink-install --packages-up-to d1_description d1h_description
# 编译硬件桥接
colcon build --symlink-install --packages-up-to hardware_bridge
# 载入环境
source install/setup.bash
```

## 仿真运行
- Webots 仿真（地形可选 `empty_world`、`stairs`、`uneven`）：
```bash
ros2 launch rl_controller sim_webots.launch.py robot:=d1 terrain:=empty_world
```
- Gazebo 仿真：
```bash
ros2 launch rl_controller sim_gazebo.launch.py robot:=d1h # d1暂时加载不出来
```
- Mujoco 仿真：
在仿真`d1`时，需要手动将`d1h_description`中的`meshes`复制到`d1_description`中，并且修改`d1_description/CMakeLists.txt`中，将meshes注释取消。然后重新编译d1_description
```bashros2 launch rl_controller hw.launch.py robot:=d1

ros2 launch rl_controller sim_mujoco.launch.py robot:=d1
```

## 硬件运行
硬件桥接依赖真实机器人驱动库（见 `hardware/tita_robot/lib/*/libtita_robot.so`）。需要将代码拷贝在机器上，需要配置硬件编译需要的环境，例如`colcon`后，再进行编译。
```bash
sudo apt install python3-colcon-common-extensions
```
编译运行在硬件上运行运动控制必要的包
```bash
colcon build --symlink-install --packages-up-to rl_controller hardware_bridge
```


- 启动控制器（硬件环境）：
在启动有硬件环境的机器上，需要手动关闭已经启动的运控服务：
```bash
sudo systemctl stop d1_bringup.service 
```

**如果运行在TITA上，需要注意：**

TITA上电后运控板默认进入Ready Mode，需要运行此[start.bash](./start.bash)，让运控板进入 Direct mode


确保无其他ros2节点在运行，然后启动硬件运控服务：
```bash
ros2 launch rl_controller hw.launch.py robot:=d1
```

## 交互控制
需编译下述功能包来启用键盘控制与遥控（ELRS）交互模块
```bash
colcon build --symlink-install --packages-up-to teleop_command keyboard_controller 
source install/setup.bash
```

- 键盘控制：
```bash
ros2 run keyboard_controller keyboard_controller_node
```

- 遥控（ELRS）：
```bash
ros2 launch teleop_command teleop_command.launch.py
```

## 控制器配置与模型
- 控制器参数与模型位于：
  - `controller/rl_controller/config/<robot>/controllers.yaml`
  - ONNX 模型示例：
    - `controller/rl_controller/config/tita/stand.onnx`
    - `controller/rl_controller/config/d1/flat.onnx`、`stairs.onnx`
- 更新控制策略时，修改对应 `controllers.yaml` 与 ONNX 文件路径
- 状态机接口与实现：`controller/rl_controller/include/rl_controller/fsm/*` 与 `src/fsm/*`


## 诊断与常见问题
- Webots未找到：  的安装路径要在环境变量中，例如：
```bash
export WEBOTS_HOME=/usr/lib/webots
```
- Mujoco 未找到：确认 `mujoco` 已安装并设置 `MUJOCO_DIR`
- 控制器未加载：检查 `controller_manager` 日志与 `controllers.yaml` 配置。  
- 模型描述加载失败：确认 `robot:=<name>` 与对应 `*_description` 包存在且可用
- 如果在TITA上遇到如下编译问题，可以尝试把代码中黄框部分去掉
![bug1](/docker/bug1.png)

```
#include "rl_controller/rl_controller_parameters.hpp"
改成#include "rl_controller_parameters.hpp"
```

## 许可证
各子包的许可证可能不同，请参考对应 `package.xml` 中的 `license` 字段。
