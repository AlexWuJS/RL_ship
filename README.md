# USVPlanner — RL_ship

**基于深度强化学习的无人水面艇（USV）局部避障规划器**

本项目基于论文 *"Local Collision Avoidance for Unmanned Surface Vehicles based on an End-to-End Planner with a LiDAR Beam Map"*，该论文已发表于 **IEEE Transactions on Intelligent Transportation Systems**。  
[![IEEE](https://img.shields.io/badge/IEEE_TITS-10959017-blue)](https://ieeexplore.ieee.org/document/10959017)

---

## 项目概述

USVPlanner 是一个端到端的无人艇局部避障规划器，核心创新是 **"beam map"（波束图）**——以 LiDAR 射线投射结果作为观测模态，直接将传感器数据映射为导航动作，免去繁琐的地图维护和复杂的特征提取。

项目在原版 USVPlanner（方形区域 + ORCA 随机船）的基础上扩展了 **真实航道地图** 和 **AIS 轨迹驱动的动态障碍船**，使 USV 能在真实内河航道场景中进行训练和避障。

### 核心特性

- **4 种 RL 算法**：TD3、DDPG、LSTM-TD3、PPO
- **两种训练模式**：
  - `ORCA 模式` (默认)：方形水域 + 随机 ORCA 船
  - `Waterway 模式`：真实航道地图 + AIS 轨迹船
- **LiDAR 波束图**：通过 Cython 加速的射线投射引擎，实时模拟 1800 束激光雷达
- **COLREGS 合规奖励**：连续时间短距离约束算法，无需先验知识即可实现合规航行
- **真实数据驱动**：集成 38 条内河 AIS 轨迹数据，使用三次样条插值实现平滑运动

---

## 项目结构

```
RL_ship/
├── main_TD3.py                          # TD3 训练入口（无 LSTM）
├── main_DDPG.py                         # DDPG 训练入口
├── main_LMTD3.py                        # LSTM-TD3 训练入口（带记忆）
├── marine_simulation.py                 # 基础仿真环境 CrowdSim（ORCA 模式）
├── marine_simulation_waterway.py        # 航道仿真环境 WaterwayCrowdSim
├── waterway_map.py                      # 航道静态地图模块
├── trajectory_ship.py                   # AIS 轨迹驱动动态船模块
├── info.py                              # 终止信号定义（ReachGoal / Collision / Grounding 等）
├── requirements.txt                     # Python 依赖
│
├── algos/                               # RL 算法实现
│   ├── TD3.py                           #   TD3（双延迟深度确定性策略梯度）
│   ├── DDPG.py                          #   DDPG（深度确定性策略梯度）
│   ├── LSTD3.py                         #   LSTM-TD3（带 LSTM 记忆的 TD3）
│   └── PPO.py                           #   PPO（近端策略优化）
│
├── utils/                               # 工具模块
│   ├── usv.py                           #   USV 运动学模型
│   ├── ship.py                          #   障碍船基类
│   ├── state.py                         #   状态表示（FullState / ObservableState）
│   ├── memory.py                        #   标准经验回放缓冲区
│   └── memory_LM.py                     #   LSTM 经验回放缓冲区（含隐藏状态）
│
├── policy/                              # 经典避障策略
│   ├── policy_factory.py                #   策略工厂
│   ├── orca.py                          #   ORCA（最优互惠避碰）
│   └── DWM.py                           #   DWM（动态窗格法）
│
├── C_library/                           # LiDAR 射线投射 Cython 加速库
│   ├── motion_plan_lib.pyx              #   Cython 源码
│   ├── setup.py                         #   编译脚本
│   ├── motion_plan_lib.cp312-win_amd64.pyd    # Windows Python 3.12 预编译
│   └── motion_plan_lib.cpython-38-x86_64-linux-gnu.so  # Linux Python 3.8 预编译
│
├── data/                                # 数据文件
│   ├── processed/maps/                  #   航道栅格地图 + 元信息
│   │   ├── navigation_map.npy           #     占用网格（0=可航行, 1=障碍物）
│   │   ├── navigation_map_meta.yaml     #     地图元数据
│   │   └── affine_params.yaml           #     仿射变换参数
│   └── processed/trajectories/          #   AIS 轨迹数据
│       └── ais_scenario.json            #     38 条预处理轨迹
│
├── plot/                                # 可视化脚本
│   ├── plot_env.py                      #   环境与轨迹可视化
│   ├── plot_different_policies.py       #   多策略训练曲线对比
│   └── background.png                   #   可视化背景图
│
├── figure/                              # 论文插图
│   ├── beam-map.png                     #   LiDAR 波束图可视化
│   ├── comparison.png                   #   算法轨迹对比
│   └── generalization.png               #   泛化能力验证
│
├── scripts/                             # 独立工具
│   └── visualize_correction.py          #   轨迹修正可视化
│
├── logdir/                              # 训练日志输出（git ignored）
│
└── changelog.md                         # 项目变更记录
```

---

## 核心模块详解

### 1. 仿真环境

| 类 | 文件 | 说明 |
|---|------|------|
| `CrowdSim` | `marine_simulation.py` | 基础环境。方形水域（±600），障碍船由 ORCA 策略驱动随机生成 |
| `WaterwayCrowdSim` | `marine_simulation_waterway.py` | 航道环境。继承 CrowdSim，用航道地图替换方形边界，用 AIS 轨迹船替换 ORCA 随机船 |

**观测空间（Observation）：**
- `lidar_state`：`(1800,)` — 1800 束 LiDAR 归一化测距值 `[0, 1]`
- `position_state`：`(2,)` — 目标相对于 USV 的极坐标 `[r/300, θ/π]`

**动作空间（Action）：**
- `(2,)` 连续值 `[-1, 1]`
  - `action[0]` — 加速度（线速度）
  - `action[1]` — 转向（角速度），乘以 `π/9` 转换为航向变化

**奖励函数：**
- **WR（不适惩罚）**：最近障碍物进入 `discomfort_dist` 范围时线性惩罚
- **GR（目标进度）**：`goal_distance_factor × (last_dist - current_dist)`
- **AR（航向保持）**：连续同向转向时给予微小正向奖励
- **CR（COLREGS 合规）**：根据障碍物方位引导 USV 左转/右转
- **WR_waterway（航道岸距）**（Waterway 模式独有）：USV 靠近河岸时的连续负奖励

**终止条件：**
| 信号 | 含义 |
|------|------|
| `ReachGoal` | USV 到达目标位置 |
| `Collision` | 与障碍船发生碰撞 |
| `Outside` | USV 离开可航区域（ORCA 模式越出方形边界） |
| `Grounding` | USV 搁浅（Waterway 模式触碰河岸） |
| `Timeout` | 超过最大步数限制 |

---

### 2. RL 算法 (`algos/`)

| 算法 | 文件 | 特点 |
|------|------|------|
| **TD3** | `algos/TD3.py` | 双延迟深度确定性策略梯度。双 Q 网络 + 目标策略平滑 + 延迟更新。**默认推荐** |
| **DDPG** | `algos/DDPG.py` | 深度确定性策略梯度。单 Q 网络，可选 LSTM 循环层 |
| **LSTM-TD3** | `algos/LSTD3.py` | 带 LSTM 记忆的 TD3。Actor/Critic 均使用 LSTM 层处理时序信息，每步显式传递隐藏状态 |
| **PPO** | `algos/PPO.py` | 近端策略优化。**On-policy** 算法，使用裁剪替代目标和 KL 惩罚 |

所有算法共享 LiDAR 压缩网络架构：
- 输入层：`1800` → 压缩层：`50`（LiDAR 特征）→ 与位置状态拼接 → 全连接层 → `tanh` 输出

---

### 3. 航道静态地图 (`waterway_map.py`)

**WaterwayMap** 管理真实航道栅格地图：

- **占用网格**：`180×90`，`0`=水面可航行，`1`=障碍物/陆地
- **轮廓提取**：使用 OpenCV `findContours` + `approxPolyDP`，提取约 143 条等高线段
- **距离变换**：`cv2.distanceTransform` 预计算每个像素到最近障碍物的距离（像素单位）
- **坐标变换**：
  - 像素 → 仿真：`x = 1.2059 × col - 108.5286`，`y = 1.2567 × row - 56.5506`
  - 仿真 → 像素：逆变换
- **轨迹修正**：`correct_trajectory_sim()` 对 AIS 轨迹进行全局平移，解决地图配准偏差。平移中位数 ~9.4 像素（~1.16m），修正后残留障碍物点从 64.8% 降至 0.09%

---

### 4. AIS 轨迹船 (`trajectory_ship.py`)

| 类 | 功能 |
|---|------|
| `TrajectoryShip` | 继承 `Ship`。加载单条 AIS 轨迹，使用 `scipy.interpolate.CubicSpline` 拟合 `x(t)`/`y(t)` |
| `TrajectoryManager` | 管理 38 条轨迹的激活/回收池，最多同时活跃 `max_active` 艘船 |

**数据驱动**：38 条真实内河 AIS 轨迹（2024 年 10 月 31 日 - 11 月 2 日），经三次样条插值实现平滑运动。轨迹加载时自动应用 WaterwayMap 的全局修正以对齐地图。

---

### 5. LiDAR 射线投射 (`C_library/`)

Cython 加速的二维射线投射引擎，是感知的核心：

- `InitializeEnv(n_line, n_circle, n_scan, laser_res)` — 初始化环境（线段 + 圆障碍物）
- `set_lines()` / `set_circles()` — 设置障碍几何体
- `set_robot_pose(x, y, yaw)` — 设置 USV 位姿
- `cal_laser()` — 发射 `n_scan` 束射线，计算每条射线与最近障碍物的交点
- `get_scan(i)` / `get_scan_line(i)` — 获取测距值或交点坐标

预编译二进制支持 Windows (Python 3.12) 和 Linux (Python 3.8)。

---

### 6. USV 运动学 (`utils/usv.py`)

**Usv** 类实现简化的船舶运动学模型：

- `compute_pose(action)`：根据加速度和转向角更新位置 `(px, py)` 和航向 `theta`
- 速度限制：`[0, v_max=6]`
- 转向限制：`delta_theta = π/9 × action[1]`

### 7. 经典策略 (`policy/`)

- **ORCA** (`orca.py`)：最优互惠避碰算法，包装 `Python-RVO2` 库。在 ORCA 模式下用于驱动障碍船和生成 USV 基线策略
- **DWM** (`DWM.py`)：动态窗格法，经典速度空间局部避障算法，作为对比基线

---

## 两种模式对比

| 维度 | ORCA 模式（默认） | Waterway 模式 |
|------|-------------------|---------------|
| 环境边界 | 方形水域 ±600 | 真实航道地图轮廓 |
| 障碍船 | ORCA 策略随机生成 | AIS 真实轨迹驱动 |
| 船数调度 | 训练计划：5→6→7→8 | TrajectoryManager 自动管理 ≤5 艘 |
| 搁浅检测 | 无 | 基于距离变换的河岸碰撞检测 |
| 额外奖励 | 无 | WR_waterway 岸距惩罚 |
| 时间限制 | 300 步 | 600 步 |
| 地图数据 | 无 | `data/processed/maps/navigation_map.npy` |

---

## 运行指令

### 环境准备

```powershell
# 1. 创建 conda 环境
conda create -n usv_planner python=3.8
conda activate usv_planner

# 2. 安装依赖
pip install -r requirements.txt

# 3. 编译 Cython 加速库（如需要）
cd C_library
python setup.py build_ext --inplace
cd ..

# 4. 安装 RVO2（仅 ORCA 模式需要）
# 参考：https://github.com/rebuttal-anonymous/Python-RVO2
```

### 训练

```powershell
# ORCA 模式（默认方形水域 + 随机 ORCA 船）
python main_TD3.py

# Waterway 模式（真实航道 + AIS 轨迹船）
python main_TD3.py --waterway_mode

# Waterway 模式（带自定义参数）
python main_TD3.py --waterway_mode --waterway_max_ships 3 --grounding_margin 8.0

# 使用其他算法
python main_DDPG.py --waterway_mode
python main_LMTD3.py --waterway_mode
```

### 评估

```powershell
# 加载已训练模型进行评估
python main_TD3.py --test --load_model /models/step_xxx_success_xx
python main_TD3.py --waterway_mode --test --load_model /models/step_xxx_success_xx
```

### 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_timesteps` | 1e6 | 总训练步数 |
| `--start_timesteps` | 1e4 | 初始随机探索步数 |
| `--eval_freq` | 2e4 | 评估间隔 |
| `--batch_size` | 96 | 训练批次大小 |
| `--hidden_size` | 512 | 网络宽度 |
| `--lr` | 3e-4 | 学习率 |
| `--discount` | 0.99 | 折扣因子 |
| `--tau` | 0.005 | 目标网络更新率 |
| `--expl_noise` | 0.15 | 探索噪声 |
| `--waterway_mode` | False | 启用航道模式 |
| `--waterway_max_ships` | 5 | 最大 AIS 活跃船数 |
| `--grounding_margin` | 5.0 | 搁浅安全边界（sim-units） |
| `--seed` | 32 | 随机种子 |

---

## 数据流

```
┌──────────────────────────────────────────────────────────────┐
│                     训练入口 (main_*.py)                       │
│  参数解析 → 环境创建 → 评估配置生成 → 训练循环                  │
└──────────┬───────────────────────────────────────┬────────────┘
           │ step(action)                          │ select_action()
           │                                       │ / train()
           ▼                                       ▼
┌──────────────────────┐                ┌────────────────────────┐
│    仿真环境           │                │   RL 算法              │
│  CrowdSim /           │◄───────────────│  TD3 / DDPG /         │
│  WaterwayCrowdSim     │   (扫瞄, 位置)  │  LSTD3 / PPO          │
│                      │─────(动作)─────►│                        │
│  │                    │                │  │ Actor (策略网络)     │
│  ├─ USV 运动学        │                │  ├─ Critic (Q 网络)    │
│  ├─ AIS 轨迹插值      │                │  ├─ LiDAR 压缩网络     │
│  ├─ LiDAR 射线投射    │                │  └─ 经验回放缓冲区     │
│  └─ 奖励计算          │                └────────────────────────┘
└──────────────────────┘
```

训练循环流程：
1. 环境返回 `(lidar_state, position_state)` 观测
2. 策略输出 `(2,)` 动作 `[加速度, 转向]`
3. 环境执行动作，更新 USV 位置和障碍船状态
4. 环境计算奖励，检测终止条件
5. 存储转换到经验回放缓冲区
6. 定期从缓冲区采样训练策略网络

---

## 依赖环境

| 依赖 | 版本 |
|------|------|
| Python | 3.8 - 3.12 |
| PyTorch | ≥1.11.0 |
| NumPy | 1.22.3 |
| Cython | 3.0.2 |
| Matplotlib | 3.7.3 |
| SciPy | (三次样条插值) |
| OpenCV | (图像处理) |
| rvo2 | (仅 ORCA 模式) |
| pysocialforce | 1.1.2 |

**注**：`rvo2` 库无法通过 pip 直接安装，需从源码编译。Waterway 模式不依赖 rvo2。

---

## 引用

```bibtex
@article{yao2025local,
    title={Local Collision Avoidance for Unmanned Surface Vehicles 
           based on an End-to-End Planner with a LiDAR Beam Map},
    author={Yao, Zetian and others},
    journal={IEEE Transactions on Intelligent Transportation Systems},
    year={2025},
    publisher={IEEE}
}
```

## 致谢

- LiDAR 波束图实现受 [zw199502/LSTM_EGO](https://github.com/zw199502/LSTM_EGO) 启发
- DRL 框架基于 [AntoineTheb/RNN-RL](https://github.com/AntoineTheb/RNN-RL)
- ORCA 算法实现来自 [rebuttal-anonymous/Python-RVO2](https://github.com/rebuttal-anonymous/Python-RVO2)
- 内河 AIS 数据来自 [gy65896/DeepSORVF](https://github.com/gy65896/DeepSORVF)
