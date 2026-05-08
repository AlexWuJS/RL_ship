"""Generate before/after visualizations for trajectory correction."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json, os, sys

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, _ROOT)
from waterway_map import WaterwayMap

_OUT = "C:/Users/Administrator/Desktop"

wm = WaterwayMap()
H, W = wm.shape

with open(os.path.join(_ROOT, "data", "processed", "trajectories", "ais_scenario.json")) as f:
    data = json.load(f)

# ====== FIGURE 1: Before vs After overview ======
fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# Left: Before
ax = axes[0]
ax.imshow(wm.grid, cmap='gray_r', origin='upper', interpolation='none')
water_x, water_y, obs_x, obs_y = [], [], [], []
for obs in data['obstacles']:
    for pt in obs['trajectory']:
        col, row = wm.sim_to_pixel(pt['x'], pt['y'])
        c, r = int(round(col)), int(round(row))
        if 0 <= c < W and 0 <= r < H:
            if wm.grid[r, c] == 0:
                water_x.append(col); water_y.append(row)
            else:
                obs_x.append(col); obs_y.append(row)
water_x = np.array(water_x); water_y = np.array(water_y)
obs_x = np.array(obs_x); obs_y = np.array(obs_y)

if len(water_x):
    ax.scatter(water_x, water_y, c='cyan', s=2, alpha=0.6, label=f'On Water ({len(water_x)})')
if len(obs_x):
    ax.scatter(obs_x, obs_y, c='red', s=5, alpha=0.8, label=f'On Obstacle ({len(obs_x)})')
ax.set_title(f'Before Correction\n{len(obs_x)}/{len(water_x)+len(obs_x)} pts on obstacles ({100*len(obs_x)/(len(water_x)+len(obs_x)):.1f}%)')
ax.legend(loc='upper right', fontsize=8)

# Right: After (global translation)
ax = axes[1]
ax.imshow(wm.grid, cmap='gray_r', origin='upper', interpolation='none')
all_x, all_y, res_x, res_y = [], [], [], []
for obs in data['obstacles']:
    xs = np.array([p['x'] for p in obs['trajectory']])
    ys = np.array([p['y'] for p in obs['trajectory']])
    xc, yc, d = wm.correct_trajectory_sim(xs, ys)
    cc, rc = wm.sim_to_pixel(xc, yc)
    for i in range(len(cc)):
        ci, ri = int(round(cc[i])), int(round(rc[i]))
        if 0 <= ci < W and 0 <= ri < H:
            if wm.grid[ri, ci] == 0:
                all_x.append(cc[i]); all_y.append(rc[i])
            else:
                res_x.append(cc[i]); res_y.append(rc[i])

all_x = np.array(all_x); all_y = np.array(all_y)
res_x = np.array(res_x); res_y = np.array(res_y)

n_total = len(all_x) + len(res_x)
ax.scatter(all_x, all_y, c='lime', s=2, alpha=0.6, label=f'On Water ({len(all_x)})')
if len(res_x):
    ax.scatter(res_x, res_y, c='orange', s=12, alpha=1.0, marker='x', label=f'Residual ({len(res_x)})')
ax.set_title(f'After Global Translation\n{len(res_x)}/{n_total} pts residual ({100*len(res_x)/n_total:.2f}%)')
ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(f'{_OUT}/trajectory_correction_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: trajectory_correction_overview.png')

# ====== FIGURE 2: Per-trajectory method & arrows ======
fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(wm.grid, cmap='gray_r', origin='upper', interpolation='none', alpha=0.4)

for obs in data['obstacles']:
    xs = np.array([p['x'] for p in obs['trajectory']])
    ys = np.array([p['y'] for p in obs['trajectory']])
    xc, yc, d = wm.correct_trajectory_sim(xs, ys)
    method = d['method']
    cr, rr = wm.sim_to_pixel(xs, ys)
    cc, rc = wm.sim_to_pixel(xc, yc)
    clr = {'translation': 'blue', 'fallback_snap': 'red', 'none': 'gray'}.get(method, 'purple')
    ax.plot(cr, rr, color=clr, linewidth=0.8, alpha=0.3)
    ax.plot(cc, rc, color=clr, linewidth=1.2, alpha=0.9)
    if method == 'translation' and len(cr) > 0:
        mid = len(cr) // 2
        ax.annotate('', xy=(cc[mid], rc[mid]), xytext=(cr[mid], rr[mid]),
                    arrowprops=dict(arrowstyle='->', color=clr, lw=1.5, alpha=0.6))

legend_elements = [
    Line2D([0], [0], color='blue', lw=2, alpha=0.7, label='Translation (34)'),
    Line2D([0], [0], color='gray', lw=2, alpha=0.7, label='Already OK (4)'),
    Line2D([0], [0], color='red', lw=2, alpha=0.7, label='Fallback Snap (0)'),
    Line2D([0], [0], color='blue', lw=1, alpha=0.3, label='Original (faint)'),
    Line2D([0], [0], color='blue', lw=2, alpha=0.9, label='Corrected (solid)'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
ax.set_title('Per-Trajectory Global Correction\nArrows = translation vector at midpoint')

plt.tight_layout()
plt.savefig(f'{_OUT}/trajectory_correction_methods.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: trajectory_correction_methods.png')

# ====== FIGURE 3: Zoom detail - old snap vs new translation ======
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
target_id = '413792745'
for obs in data['obstacles']:
    if obs['id'] == target_id:
        xs = np.array([p['x'] for p in obs['trajectory']])
        ys = np.array([p['y'] for p in obs['trajectory']])
        xc, yc, d = wm.correct_trajectory_sim(xs, ys)
        cr, rr = wm.sim_to_pixel(xs, ys)
        cc, rc = wm.sim_to_pixel(xc, yc)

        pad = 10
        x_min = max(0, int(min(cr.min(), cc.min()) - pad))
        x_max = min(W, int(max(cr.max(), cc.max()) + pad))
        y_min = max(0, int(min(rr.min(), rc.min()) - pad))
        y_max = min(H, int(max(rr.max(), rc.max()) + pad))

        for idx, (ax, title) in enumerate(zip(axes, [
            'Old: Per-Point Snap (jagged, loses shape)',
            f'New: Global Translation ({d["dc"]},{d["dr"]}) px (preserves shape)'
        ])):
            ax.imshow(wm.grid[y_min:y_max, x_min:x_max], cmap='gray_r', origin='upper',
                     extent=[x_min, x_max, y_max, y_min], interpolation='none')
            if idx == 0:
                snap_c, snap_r = [], []
                for i in range(len(cr)):
                    sx, sy = wm.snap_to_water(float(xs[i]), float(ys[i]))
                    sc, sr = wm.sim_to_pixel(sx, sy)
                    snap_c.append(sc); snap_r.append(sr)
                ax.plot(cr, rr, 'r-', lw=1.5, alpha=0.7, label='Original')
                ax.plot(snap_c, snap_r, 'orange', lw=2, alpha=0.9, label='Per-pt snap (distorted)')
                ax.scatter(cr, rr, c='red', s=10, alpha=0.5)
                ax.scatter(snap_c, snap_r, c='orange', s=10, alpha=0.5)
            else:
                ax.plot(cr, rr, 'r-', lw=1.5, alpha=0.5, label='Original')
                ax.plot(cc, rc, 'b-', lw=2, alpha=0.9, label='Translated')
                ax.scatter(cr, rr, c='red', s=10, alpha=0.3)
                ax.scatter(cc, rc, c='blue', s=10, alpha=0.7)
            ax.set_title(title)
            ax.legend(fontsize=8)
        break

plt.tight_layout()
plt.savefig(f'{_OUT}/trajectory_correction_detail.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: trajectory_correction_detail.png')
print('Done!')
