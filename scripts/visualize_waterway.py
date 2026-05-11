"""
Waterway training/evaluation visualization script.

Two modes:
  - REPLAY:  visualize saved .npz evaluation episodes
  - LIVE:    run a trained model on random episodes and record trajectories

Usage:
  python scripts/visualize_waterway.py --replay ./logdir/TD330/marine_simulation/seed_32/evaluation_episodes/eval00.npz
  python scripts/visualize_waterway.py --replay ./logdir/TD330/marine_simulation/seed_32/evaluation_episodes/
  python scripts/visualize_waterway.py --model ./logdir/TD330/marine_simulation/seed_32/models/step_900000_success_95 --episodes 5
  python scripts/visualize_waterway.py --model <path> --lidar  # show LiDAR beams
"""
import argparse
import math
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


# ── boat shape helpers (from plot_env.py) ──────────────────────────

def _draw_rotated_boat(center, rotation_angle):
    bias = np.array([
        [0, 13], [2, 12.6], [4, 11.25], [6, 8.6], [8, 0],
        [8, -4.6], [7.9, -8.9], [7.5, -15.4], [7, -21], [6, -23.5],
        [5, -25.9], [4, -26.7], [2, -27], [0, -27.5],
        [-2, -27], [-4, -26.7], [-5, -25.9], [-6, -23.5], [-7, -21],
        [-7.5, -15.4], [-7.9, -8.9], [-8, -4.6], [-8, 0],
        [-6, 8.6], [-4, 11.25], [-2, 12.6]
    ])
    polygon = []
    for p in bias:
        x_rot = (p[0] * math.cos(rotation_angle) - p[1] * math.sin(rotation_angle)) + center[0]
        y_rot = (p[0] * math.sin(rotation_angle) + p[1] * math.cos(rotation_angle)) + center[1]
        polygon.append((x_rot, y_rot))
    return polygon


def _zoom_contour(contour, margin):
    import pyclipper
    pco = pyclipper.PyclipperOffset()
    pco.MiterLimit = 15
    pco.AddPath(contour, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)
    solution = pco.Execute(margin)
    return np.array(solution).reshape(-1, 2).astype(float)


# ── waterway map helpers ───────────────────────────────────────────

def _load_waterway_map():
    from waterway_map import WaterwayMap
    return WaterwayMap(contour_simplification=1.0)


def _draw_waterway_background(ax, wm):
    """Draw the waterway occupancy grid and contour lines."""
    H, W = wm.shape
    x0, y0 = wm.pixel_to_sim(0, 0)
    x1, y1 = wm.pixel_to_sim(W, H)

    # occupancy grid (water, land)
    rgba = np.zeros((H, W, 4), dtype=np.float64)
    rgba[wm.grid == 0] = [0.15, 0.30, 0.55, 0.6]   # dark blue water
    rgba[wm.grid == 1] = [0.45, 0.35, 0.25, 0.5]     # brown land
    ax.imshow(rgba, origin='upper', interpolation='bilinear',
              extent=[x0, x1, y1, y0], zorder=0)

    # contour lines
    for (x1, y1), (x2, y2) in wm.contour_lines:
        ax.plot([x1, x2], [y1, y2], color='#8B7355', linewidth=0.8, alpha=0.7, zorder=1)


# ── episode replay ─────────────────────────────────────────────────

def _load_episode(npz_path):
    """Load a single .npz episode and return structured arrays."""
    data = np.load(npz_path, allow_pickle=True)
    out = {}
    out['usv'] = np.array(data['usv'])           # (T, 4)  [px, py, v, theta]
    out['goal'] = np.array(data['goal'])          # (T, 2)  [gx, gy]
    out['ships'] = np.array(data['ships'])        # (T, N, 4) [px, py, radius, vx]
    out['reward'] = np.array(data['reward'])      # (T, 1)
    out['subreward'] = np.array(data['subreward'])  # (T, 4)
    out['laser'] = np.array(data['laser'])        # (T, 1800, 4)
    return out


def visualize_episode(ax, episode, wm, show_lidar=False, step_stride=10):
    """Draw one episode on the given axes."""
    usv = episode['usv']
    ships = episode['ships']
    laser = episode['laser']
    T = usv.shape[0]
    ship_count = ships.shape[1] if ships.ndim == 3 else 0
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4']

    # USV trajectory
    ax.plot(usv[:, 0], usv[:, 1], color='#00ffff', linewidth=2.0, zorder=5, label='USV')
    # start and end markers
    ax.scatter(usv[0, 0], usv[0, 1], color='lime', s=120, marker='o', edgecolors='black',
               zorder=6, label='Start')
    ax.scatter(usv[-1, 0], usv[-1, 1], color='red', s=120, marker='*', edgecolors='black',
               zorder=6, label='End')

    # USV boat shapes every step_stride frames
    for t in range(0, T, step_stride):
        if t == 0:
            heading = math.atan2(usv[min(1, T-1), 1] - usv[0, 1],
                                 usv[min(1, T-1), 0] - usv[0, 0]) - math.pi / 2
        else:
            heading = math.atan2(usv[t, 1] - usv[t-1, 1],
                                 usv[t, 0] - usv[t-1, 0]) - math.pi / 2
        boat = _draw_rotated_boat((usv[t, 0], usv[t, 1]), heading)
        boat = _zoom_contour(boat, 2)
        patch = plt.Polygon(boat, closed=True, edgecolor='#00bfff', facecolor='#00ffff',
                           alpha=0.4 + 0.4 * t / T, zorder=5)
        ax.add_patch(patch)

    # Ship trajectories and final positions
    if ship_count > 0:
        for j in range(ship_count):
            # trajectory (only valid positions)
            sx = ships[:, j, 0]
            sy = ships[:, j, 1]
            # skip ships that never moved
            if sx[-1] == sx[0] and sy[-1] == sy[0]:
                continue
            ax.plot(sx, sy, color=colors[j % len(colors)], linewidth=1.0, alpha=0.6, zorder=3)

            # final position circle
            sr = ships[-1, j, 2]
            if sr > 0:
                circ = plt.Circle((sx[-1], sy[-1]), sr, fill=False, edgecolor=colors[j % len(colors)],
                                  linewidth=1.5, alpha=0.8, zorder=4)
                ax.add_patch(circ)

    # LiDAR
    if show_lidar and laser.ndim == 3:
        # pick ~30 representative frames
        lidar_frames = np.linspace(0, T - 1, min(30, T), dtype=int)
        alpha_map = np.linspace(0.15, 0.5, len(lidar_frames))
        for fi, t in enumerate(lidar_frames):
            frame = laser[t]
            for bi in range(0, 1800, 40):
                ox, oy, hx, hy = frame[bi]
                dx, dy = hx - ox, hy - oy
                if math.hypot(dx, dy) < 99.0:
                    ax.plot([ox, hx], [oy, hy], color='red', linewidth=0.3,
                            alpha=alpha_map[fi], zorder=2)

    # Goal marker
    goal = episode['goal'][-1]  # last recorded goal
    ax.scatter(goal[0], goal[1], color='gold', s=150, marker='D', edgecolors='black',
               zorder=6, label='Goal')

    # Info
    outcome = 'Success' if 'ReachGoal' in str(getattr(episode, 'outcome', '')) else (
        'Collision' if usv.shape[0] < 600 else 'Timeout'
    )
    r_total = episode['reward'].sum()
    ax.set_title(f'Steps: {T}  |  Total reward: {r_total:.1f}', fontsize=12)


def replay_mode(npz_paths, wm, show_lidar=False, save_dir=None):
    """Replay one or more saved .npz episodes."""
    n = len(npz_paths)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 7))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, path in enumerate(npz_paths):
        ep = _load_episode(path)
        ax = axes[i]
        _draw_waterway_background(ax, wm)
        visualize_episode(ax, ep, wm, show_lidar=show_lidar)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        name = os.path.basename(path).replace('.npz', '')
        ax.set_title(f'{name}  | Steps={ep["usv"].shape[0]}  Reward={ep["reward"].sum():.1f}',
                     fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # global legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00ffff', lw=2, label='USV trajectory'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10,
               label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10,
               label='End'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gold', markersize=10,
               label='Goal'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, 'waterway_replay.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'Saved: {fname}')
    else:
        plt.show()


# ── demo mode (random actions, no model needed) ────────────────────

def demo_mode(wm, episodes=3, show_lidar=False, save_dir=None):
    """Run episodes with random actions (no trained model required) for quick preview."""
    args = _build_args()
    from marine_simulation_waterway import WaterwayCrowdSim
    env = WaterwayCrowdSim(args, e_mode=True)

    logs = []
    for i in range(episodes):
        seed = 2000 + i
        env.total_timesteps = seed
        lidar, pos = env.reset(phase='val')
        done = False
        while not done:
            action = np.random.uniform(-1, 1, 2).astype(np.float32)
            lidar, pos, reward, done, info = env.step(action)
        ep_reward = sum(np.asarray(r).sum() for r in env.log_env["reward"])
        print(f'Demo episode {i+1}: steps={len(env.log_env["usv"])}, '
              f'reward={ep_reward:.1f}, info={info}')
        logs.append(dict(env.log_env))

    _plot_episodes(logs, wm, show_lidar=show_lidar, title_prefix='Demo (Random Actions)',
                   save_dir=save_dir, suffix='demo')


def _plot_episodes(episodes_logs, wm, show_lidar=False, title_prefix='',
                   save_dir=None, suffix=''):
    """Shared plotting helper: draw a grid of episodes."""
    n = len(episodes_logs)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 7))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, log in enumerate(episodes_logs):
        ax = axes[i]
        _draw_waterway_background(ax, wm)
        ep = {
            'usv': np.stack(log['usv']),
            'ships': np.stack(log['ships']),
            'laser': np.stack(log['laser']),
            'reward': np.stack(log['reward']),
            'goal': np.stack(log['goal']),
        }
        visualize_episode(ax, ep, wm, show_lidar=show_lidar)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        r_total = ep['reward'].sum()
        ax.set_title(f'{title_prefix} #{i+1}  |  Steps={ep["usv"].shape[0]}  Reward={r_total:.1f}',
                     fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#00ffff', lw=2, label='USV'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=10, label='End'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gold', markersize=10, label='Goal'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f'waterway_{suffix}.png')
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        print(f'Saved: {fname}')
    else:
        plt.show()


# ── live model rollout ─────────────────────────────────────────────

def _build_args():
    """Build a minimal args namespace for WaterwayCrowdSim."""
    p = argparse.Namespace()
    p.waterway_mode = True
    p.waterway_max_ships = 5
    p.grounding_margin = 5.0
    p.laser_angle_resolute = 0.003490659
    p.laser_min_range = 2.5
    p.laser_max_range = 100.0
    p.square_width = 300.0
    p.discomfort_distance = 30
    p.only_dynamic = True
    p.classical = False
    p.action_dim = 2
    p.lidar_dim = 1800
    p.lidar_feature_dim = 50
    p.goal_position_dim = 2
    p.env = 'marine_simulation'
    p.seed = 32
    return p


def _load_policy(model_path, device='cpu'):
    """Load a TD3 policy from a saved checkpoint directory."""
    import torch
    from algos import TD3

    lidar_state_dim = 1800
    position_state_dim = 2
    lidar_feature_dim = 50
    action_dim = 2
    max_action = 1.0
    hidden_dim = 512

    kwargs = dict(
        lidar_state_dim=lidar_state_dim, position_state_dim=position_state_dim,
        lidar_feature_dim=lidar_feature_dim, action_dim=action_dim,
        max_action=max_action, hidden_dim=hidden_dim,
        discount=0.99, tau=0.005, device=device
    )
    policy = TD3.TD3(**kwargs)
    # model_path is a directory containing actor_*, critic_*, etc.
    # the policy.load expects a prefix
    policy.load(model_path)
    policy.eval_mode()
    return policy


def run_episode(policy, env, seed=None):
    """Run one episode and return log data."""
    if seed is not None:
        env.total_timesteps = seed
    lidar, pos = env.reset(phase='val')
    done = False
    while not done:
        with torch.no_grad():
            action = policy.select_action(np.array(lidar), np.array(pos))
        lidar, pos, reward, done, info = env.step(action)
    return dict(env.log_env)


def live_mode(model_path, wm, episodes=5, show_lidar=False, save_dir=None):
    """Run live episodes with a trained model and visualize them."""
    import torch

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    print(f'Loading model from: {model_path}')
    policy = _load_policy(model_path, device=device)
    args = _build_args()
    from marine_simulation_waterway import WaterwayCrowdSim
    env = WaterwayCrowdSim(args, e_mode=True)

    logs = []
    for i in range(episodes):
        seed = 1000 + i
        print(f'Running episode {i+1}/{episodes} (seed={seed})...')
        log = run_episode(policy, env, seed=seed)
        logs.append(log)
        usv = log['usv']
        outcome = 'ReachGoal' if usv.shape[0] < 600 else 'Timeout'
        print(f'  Steps={usv.shape[0]}, Total reward={sum(log["reward"]):.1f}, Outcome={outcome}')

    _plot_episodes(logs, wm, show_lidar=show_lidar, title_prefix='Live Rollout',
                   save_dir=save_dir, suffix='live')


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waterway training visualization')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--replay', type=str, default=None,
                       help='Path to .npz file or directory of .npz evaluation episodes')
    group.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint prefix (for live rollout)')
    group.add_argument('--demo', action='store_true', default=False,
                       help='Run demo episodes with random actions (no model required)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of live episodes to run (--model mode only)')
    parser.add_argument('--lidar', action='store_true', default=False,
                       help='Show LiDAR beams')
    parser.add_argument('--save', type=str, default=None,
                       help='Save figure to this directory instead of showing')
    o = parser.parse_args()

    print('Loading waterway map...')
    wm = _load_waterway_map()
    print(f'Waterway map loaded: {wm.shape[0]}x{wm.shape[1]}, {len(wm.contour_lines)} contour lines')

    if o.replay:
        if os.path.isdir(o.replay):
            paths = sorted(glob(os.path.join(o.replay, '*.npz')))
        else:
            paths = [o.replay]
        print(f'Replaying {len(paths)} episode(s)')
        replay_mode(paths, wm, show_lidar=o.lidar, save_dir=o.save)

    elif o.model:
        live_mode(o.model, wm, episodes=o.episodes, show_lidar=o.lidar, save_dir=o.save)

    elif o.demo:
        print(f'Running {o.episodes} demo episodes (random actions)...')
        demo_mode(wm, episodes=o.episodes, show_lidar=o.lidar, save_dir=o.save)
