# 项目变更记录

> RL_ship — 基于 SAC 的航道动态避障仿真系统

---

## 第 1 次变动

### 文件内容

| 文件/文件夹 | 内容说明 |
|------------|---------|
| `waterway_map.py` | 航道静态地图模块。加载栅格占用地图 (`navigation_map.npy`)，提取障碍物轮廓为简化多边形线段，提供像素坐标系 ↔ 仿真坐标系的双向转换，以及用于奖励整形的距离变换计算。 |
| `data/` | 航道地图与 AIS 轨迹数据。包含经处理的栅格地图 (`processed/maps/navigation_map.npy`) 及元信息、仿射变换参数；预处理后的 38 条 AIS 轨迹场景 (`processed/trajectories/ais_scenario.json`)；以及原始 AIS 轨迹 XLS 文件 (`raw/trajectories/`)。 |
| `.gitignore` | Git 忽略规则，排除 `__pycache__/`、`*.pyc`、`logdir/`、构建产物等。 |
| `changelog.md` | 项目变更记录文档（本文件）。 |

### 修改内容

| 文件 | 修改摘要 |
|------|---------|
| （首次提交） | 基于 [USVPlanner](https://github.com/yaozt98/USVPlanner.git) 项目初始化项目结构，添加航道地图与轨迹数据，实现静态地图管理模块。 |

---

## 第 2 次变动

### 文件内容

| 文件/文件夹 | 内容说明 |
|------------|---------|
| `trajectory_ship.py` | 轨迹驱动的动态障碍物船只模块。加载 `ais_scenario.json` 中 38 条 AIS 轨迹，使用三次样条插值 (`scipy.interpolate.CubicSpline`) 实现平滑运动，自动管理最多 5 艘活跃船只的激活与回收，接口兼容 `Ship` 类，可直接替换 CrowdSim 中的随机 ORCA 船只。 |

---

## 第 3 次变动

### 问题

约 65% 的 AIS 轨迹点映射到栅格地图后落在障碍物像素（陆地）上，而非水面。根因是地图的原始 PGM 地理配准与轨迹坐标变换参数之间约 0.25m（中位数）的微小偏差。

### 修改内容

| 文件 | 修改摘要 |
|------|---------|
| `waterway_map.py` | 新增轨迹级整体修正方法：`correct_trajectory_sim()` 主入口；`correct_trajectory_pixel()` 三级回退编排（平移→逐点吸附）；`_find_best_translation()` 网格搜索最优平移向量，代价函数惩罚越界点；`verify_trajectory()` 诊断统计。保留 `snap_to_water()` 作为公共 API 和最终回退。 |
| `trajectory_ship.py` | `TrajectoryShip.__init__` 用 `correct_trajectory_sim()` 整体变换替换逐点 `snap_to_water()` 循环，新增 `correction_info` 属性。`TrajectoryManager` 增加 `_log_correction_summary()` 汇总修正统计。 |
| `scripts/visualize_correction.py` | 修正效果可视化脚本，生成总览图、方法分布图、逐点吸附 vs 整体平移对比图。 |

### 方法设计

采用**轨迹级全局变换**（非逐点吸附），优先保持轨迹的几何形状：
1. **平移**（34/38 轨迹采用）：网格搜索最优 (dc, dr)，完美保留形状
2. **逐点吸附**（0/38 轨迹采用）：仅在平移残留 >10% 时回退

### 验证结果

- 修正后 **9/9867** 轨迹点残留在障碍物上（0.09%），修正前 6391 个（64.8%）
- 形状保真度：**中位数 0.000，最大值 0.000**（平移完美保留点间距离）
- 方法分布：平移 34，无需修正 4，回退吸附 0
- 平移距离：中位数 9.4 像素（~1.16m），最大值 14.1 像素（~1.73m）
