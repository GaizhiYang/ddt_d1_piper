# D1 + Piper-L loco-manipulation 可行性评估

## 结论

项目在工程上可行，推荐采用“IsaacLab 训练 ABI 冻结 → MuJoCo sim2sim 对齐 →
双 CAN 只读/shadow → 机械支撑下分系统测试 → 全身低速测试”的路线。
当前仓库已经完成第一阶段的可运行骨架，但还没有达到可以发送真实全身动作的
条件。`can0`/`can1` 不会在启动时自动打开、使能或发送电机命令。

本次评估使用的策略文件为
`/home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx`。
它已通过只读 ABI 检查（`float32 [1,246] -> [1,22]`），SHA256 为
`c142c881aac43d90c2d5970dff392a77c7c3610196a933ba72ea2459d7c88010`。

## 已有证据

| 项目 | 结果 |
| --- | --- |
| ONNX 输入/输出 | `float32 obs[1,246] -> actions[1,22]` |
| 策略文件 SHA256 | `c142c881aac43d90c2d5970dff392a77c7c3610196a933ba72ea2459d7c88010` |
| MuJoCo 模型 | `nq=32, nv=31, nu=22` |
| 关节顺序 | 四条腿各 `hip/thigh/calf/foot`，然后 `Piper joint1..6` |
| 观测 | 82 维单帧、3 帧 term-major 历史，共 246 维 |
| 控制时序 | `dt=0.005 s`、decimation=4、策略 50 Hz |
| 静态检查 | `xacro`、`check_urdf`、资源检查、12 项契约测试通过 |
| 策略 rollout | 30 s 无 NaN、未触发跌倒；这不等于末端跟踪通过 |

可重复检查：

```bash
source /opt/ros/humble/setup.bash
cd /home/hh/loco_mani_ws
colcon build --packages-select loco_mani_description loco_mani_rl_controller loco_mani_bringup --symlink-install
source install/setup.bash

/home/hh/anaconda3/envs/robridge/bin/python \
  src/ddt_controller/scripts/check_policy_abi.py \
  /home/hh/loco-mani/logs/rsl_rl/d1_piper_wbc/2026-08-10_16-51-00/exported/policy.onnx

/home/hh/anaconda3/envs/robridge/bin/python \
  src/ddt_controller/scripts/test_loco_mani_contract.py
```

## 当前阻塞项

1. 训练 reset 高度为 0.45 m，而当前简化 MuJoCo 轮碰撞在该高度约有 6.6 mm
   穿透；0.4567 m 只是诊断补偿值，不能替换 IsaacLab 的训练参数。
2. PhysX 与 MuJoCo 的 solver、接触、摩擦和反弹模型不等价，需要用 IsaacLab
   trace 做回放标定。
3. Piper 的实际支架安装位姿、负载、末端坐标系和碰撞几何尚未实测确认。
4. 尚无 IsaacLab 的 `q/dq/tau/base/EE/observation/action` 记录，因此还不能
   给出动力学等价或 WBC 任务成功的结论。
5. Piper 固件版本会改变 MIT 报文（尤其 v188/v189）；D1 的 CAN-FD 驱动 ABI、
   16 个电机 ID、方向和零位也必须在现场确认。

补充检查发现：当前随 DDT 工程提供的 `libtita_robot.so`（amd64/arm64）实际
导出 `CanfdApi::send_motors_can(std::vector<api_motor_out_t>)`，而不导出旧文档
中出现的 `send_leg_motors_can`。仓库后端已兼容两种符号；在 Jetson 上仍须用
同版本头文件、编译器和 `libstdc++` 构建绑定，不能仅凭 `nm` 检查就认为 C++
结构体/`std::vector` ABI 已验证。

## 推荐实现

运行时拆成以下相互独立的层：

```text
D1Backend(can0)       PiperBackend(can1, pyAgxArm)
       │                         │
       └── 时间戳状态快照（只读） ──┘
                    │
         ObservationAdapter (82/246)
                    │
             ONNXPolicy (50 Hz)
                    │
 ActionAdapter (D1 位置/轮速 + Piper MIT)
                    │
       SafetySupervisor + E-stop + watchdog
```

低层建议 500 Hz，策略 50 Hz。CAN 接收线程不能阻塞策略线程；发送前必须经过
逐关节方向/offset/位置/速度/力矩/力矩变化率限幅。硬件配置保持
`enable_on_start: false`、`dry_run: true`，并要求人工 gate 和外部急停通过后才
允许进入实机测试。

## 进入真机的验收门槛

在双 CAN 发送动作前，必须完成：

1. IsaacLab 首帧与 MuJoCo 首帧逐元素对齐；
2. 无重力单关节 PD、带重力零策略静态接触测试；
3. 5--10 s Isaac action trace 在 MuJoCo 中回放，比较 `q/dq/tau/EE`；
4. `vcan` 或机械支撑上的 D1/Piper 只读与 shadow 测试；
5. 机械支撑下 Piper 单轴→六轴、D1 单独站立、全身低速运动；
6. 通过倾角、基座高度、状态超时、策略 watchdog、发送失败和急停测试。

具体参数和上电顺序见 [implementation_plan.md](implementation_plan.md)、
[sim2sim_audit.md](sim2sim_audit.md) 和 [hardware_deployment.md](hardware_deployment.md)。
