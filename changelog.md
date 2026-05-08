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
| `waterway_map.py` | 新增 `snap_to_water(x, y)` 方法，将偏离轨迹点吸附到最近水面像素。新增 `_compute_snap_offsets()` 预计算每个障碍物像素到最近水面的 (dcol, dr) 偏移量，使用 `scipy.spatial.cKDTree` 实现 O(1) 查表修正。 |
| `trajectory_ship.py` | `TrajectoryShip.__init__` 新增可选参数 `waterway_map`，传入后自动对所有轨迹点调用 `snap_to_water`，用修正后的 (x, y) 重建三次样条。`TrajectoryManager` 同步透传 `waterway_map` 参数。 |

### 验证结果

- 修正后 **0/9867** 轨迹点落在障碍物上（修正前 6391 个，64.8%）
- 轨迹修正偏移量：中位数 1.79 像素（~0.22m），最大值 8.56 像素（~1.05m）
- 全局障碍物像素偏移量：中位数 4.47 像素，最大值 23.85 像素
