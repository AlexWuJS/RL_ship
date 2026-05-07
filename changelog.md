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
