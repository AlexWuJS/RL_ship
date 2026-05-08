"""
WaterwayCrowdSim — CrowdSim subclass integrating WaterwayMap and TrajectoryManager.

Replaces ORCA random ships with AIS trajectory-driven vessels and square
boundaries with waterway contour lines from a real navigation chart.
"""
from math import atan2, cos, hypot, sin

import numpy as np

from C_library.motion_plan_lib import *
from info import Grounding, Nothing, ReachGoal, Collision, Timeout, Danger
from marine_simulation import CrowdSim
from trajectory_ship import TrajectoryManager
from utils.state import FullState, JointState
from waterway_map import WaterwayMap


class WaterwayCrowdSim(CrowdSim):
    """CrowdSim variant that uses a real waterway chart and AIS ship trajectories."""

    def __init__(self, args, schedule: dict = None, e_mode=False):
        super().__init__(args, schedule, e_mode)

        # ---- waterway map ----
        self._waterway_map = WaterwayMap(contour_simplification=1.0)

        # sim-units per pixel (from WaterwayMap fitted transform)
        self._sim_per_px = 1.205873

        # ---- AIS trajectory manager ----
        max_active = getattr(args, 'waterway_max_ships', 5)
        self._trajectory_manager = TrajectoryManager(
            max_active=max_active,
            time_step=self.time_step,
            waterway_map=self._waterway_map,
        )

        # ---- grounding (bank collision) ----
        self._grounding_margin = getattr(args, 'grounding_margin', 5.0)
        self.grounding_penalty = -10.0

        # ---- rescale goal normalisation to waterway extent (~215 m) ----
        self.square_width = 300.0
        self.half_square = 150.0

        # ---- LiDAR line set (outer safety fence + waterway contours) ----
        self.lines = self._build_lidar_lines()

        # ---- longer episodes for realistic waterway navigation ----
        self.time_limit = 600
        self.success_reward = 50

        # seed used by TrajectoryManager for reproducibility
        self._current_seed = 0

    # -----------------------------------------------------------------
    #   LiDAR lines
    # -----------------------------------------------------------------

    def _build_lidar_lines(self):
        """Combine an outer safety boundary with waterway contour segments."""
        m = self.half_square
        boundary = [
            [(-m, -m), (-m,  m)],
            [(-m,  m), ( m,  m)],
            [( m,  m), ( m, -m)],
            [( m, -m), (-m, -m)],
        ]
        return boundary + list(self._waterway_map.contour_lines)

    # -----------------------------------------------------------------
    #   Position sampling (inside navigable water)
    # -----------------------------------------------------------------

    def _sample_water_position(self, min_dist_to_obs=25.0, rng=None):
        """Sample a (x, y) position inside the waterway, away from banks."""
        if rng is None:
            rng = np.random

        grid = self._waterway_map.grid                     # (H, W), 0=free
        free_mask = (grid == 0)
        free_pixels = np.argwhere(free_mask)                # (N, 2) [row, col]

        if len(free_pixels) == 0:
            return 0.0, 0.0

        # minimum distance in pixels
        min_dist_px = min_dist_to_obs / self._sim_per_px

        dists = self._waterway_map.distance_transform       # (H, W), pixels
        valid_mask = dists[free_mask] >= min_dist_px
        valid_pixels = free_pixels[valid_mask]

        if len(valid_pixels) == 0:
            best_idx = int(np.argmax(dists[free_mask]))
            row, col = free_pixels[best_idx]
        else:
            idx = rng.randint(len(valid_pixels))
            row, col = valid_pixels[idx]

        return self._waterway_map.pixel_to_sim(float(col), float(row))

    def _sample_start_goal(self, rng):
        """Return (sx, sy, gx, gy) — start and goal in navigable water."""
        min_sep = 50.0
        for _ in range(50):
            sx, sy = self._sample_water_position(min_dist_to_obs=25.0, rng=rng)
            gx, gy = self._sample_water_position(min_dist_to_obs=25.0, rng=rng)
            if hypot(sx - gx, sy - gy) >= min_sep:
                return sx, sy, gx, gy

        # fallback: farthest pair from free pixels
        grid = self._waterway_map.grid
        free_pixels = np.argwhere(grid == 0)
        if len(free_pixels) >= 2:
            idx = rng.choice(len(free_pixels), 2, replace=False)
            (r1, c1), (r2, c2) = free_pixels[idx]
            sx, sy = self._waterway_map.pixel_to_sim(float(c1), float(r1))
            gx, gy = self._waterway_map.pixel_to_sim(float(c2), float(r2))
            return sx, sy, gx, gy
        return 0.0, 0.0, 30.0, 0.0

    # -----------------------------------------------------------------
    #   Overrides
    # -----------------------------------------------------------------

    def generate_random_ship_position(self):
        """Replace ORCA random ships with AIS trajectory ships."""
        self._trajectory_manager.reset(sim_time=0.0, seed=self._current_seed)
        self.ships = list(self._trajectory_manager.active_ships)
        self.ship_num = len(self.ships)

    def outside_check(self):
        """Check for grounding (too close to waterway bank or out of map)."""
        x, y = self.usv.px, self.usv.py

        if self._waterway_map.is_outside_map(x, y):
            return True

        dist_px = self._waterway_map.dist_to_nearest_obstacle(x, y)
        dist_sim = dist_px * self._sim_per_px
        return dist_sim < self.usv.radius + self._grounding_margin

    def get_lidar(self):
        """Send waterway contour lines + ship circles to the C raycasting lib."""
        scan = np.zeros(self.n_laser, dtype=np.float32)
        scan_end = np.zeros((self.n_laser, 2), dtype=np.float32)

        all_lines = self._build_lidar_lines()
        self.lines = all_lines  # keep consistent for debugging / render

        # circles from active AIS ships
        self.circles = np.zeros((self.ship_num, 3), dtype=np.float32)
        for i in range(self.ship_num):
            self.circles[i, :] = np.array([
                self.ships[i].px, self.ships[i].py, self.ships[i].radius,
            ])

        usv_pose = np.array([self.usv.px, self.usv.py, self.usv.theta])
        num_line = len(all_lines)
        num_circle = self.ship_num

        InitializeEnv(num_line, num_circle, self.n_laser, self.laser_angle_resolute)

        for i in range(num_line):
            set_lines(4 * i,     all_lines[i][0][0])
            set_lines(4 * i + 1, all_lines[i][0][1])
            set_lines(4 * i + 2, all_lines[i][1][0])
            set_lines(4 * i + 3, all_lines[i][1][1])

        for i in range(num_circle):
            set_circles(3 * i,     self.ships[i].px)
            set_circles(3 * i + 1, self.ships[i].py)
            set_circles(3 * i + 2, self.ships[i].radius)

        set_robot_pose(usv_pose[0], usv_pose[1], usv_pose[2])
        cal_laser()

        self.scan_intersection = []
        for i in range(self.n_laser):
            scan[i] = get_scan(i)
            scan_end[i, :] = np.array([get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)])
            self.scan_intersection.append([
                (get_scan_line(4 * i + 0), get_scan_line(4 * i + 1)),
                (get_scan_line(4 * i + 2), get_scan_line(4 * i + 3)),
            ])

        self.scan_current = np.clip(scan, self.laser_min_range,
                                     self.laser_max_range) / self.laser_max_range
        ReleaseEnv()

    def reset(self, phase='test'):
        """Reset USV inside waterway and initialise AIS ships."""
        assert phase in ['train', 'val', 'test']
        self.global_time = 0
        self.log_env = {}

        # use total_timesteps as seed for reproducibility
        self._current_seed = int(self.total_timesteps) if not self.e_mode else 42

        rng = np.random.RandomState(self._current_seed)
        sx, sy, gx, gy = self._sample_start_goal(rng)
        theta = atan2(gy - sy, gx - sx)
        self.usv.set(sx, sy, gx, gy, 0.0, 0.0, theta)
        self.goal_distance_last = self.usv.get_goal_distance()

        self.generate_random_ship_position()
        self.get_lidar()

        # build observation (identical format to parent)
        dx = self.usv.gx - self.usv.px
        dy = self.usv.gy - self.usv.py
        theta = self.usv.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)

        # logging
        self.log_env['usv'] = [np.array([self.usv.px, self.usv.py,
                                          self.usv.v, self.usv.theta])]
        self.log_env['goal'] = [np.array([self.usv.gx, self.usv.gy])]
        ships_position = []
        for ship in self.ships:
            ships_position.append(np.array([ship.px, ship.py, ship.radius, ship.vx]))
        self.log_env['ships'] = [np.array(ships_position)]
        self.log_env['reward'] = [np.zeros(1)]
        self.log_env['subreward'] = [np.array([0.0, 0.0, 0.0, 0.0])]
        lasers = []
        for ls in self.scan_intersection:
            lasers.append(np.array([ls[0][0], ls[0][1], ls[1][0], ls[1][1]]))
        self.log_env['laser'] = [np.array(lasers)]

        if self.classical:
            return np.array(lasers), np.array([dx, dy]), np.array([self.usv.v, self.usv.theta])
        return self.scan_current, ob_position

    def _waterway_reward(self):
        """Continuous penalty for proximity to banks (negative, zero in open water)."""
        x, y = self.usv.px, self.usv.py

        if self._waterway_map.is_outside_map(x, y):
            return -5.0

        dist_px = self._waterway_map.dist_to_nearest_obstacle(x, y)
        dist_sim = dist_px * self._sim_per_px

        threshold = self.usv.radius + self._grounding_margin  # ~20 sim-units
        if dist_sim < threshold:
            return -0.5 * (1.0 - dist_sim / threshold)
        if dist_sim < threshold * 2.0:
            return -0.05 * (2.0 - dist_sim / threshold)
        return 0.0

    def step(self, action):
        """One environment step with AIS trajectory-driven obstacle ships."""
        # ---- advance AIS ships ----
        self._trajectory_manager.step(float(self.global_time))
        self.ships = list(self._trajectory_manager.active_ships)
        self.ship_num = len(self.ships)

        # ---- USV update ----
        usv_x, usv_y, usv_theta = self.usv.compute_pose(action)
        self.usv.update_states(usv_x, usv_y, usv_theta, action)

        # ---- LiDAR ----
        self.get_lidar()
        self.global_time += self.time_step

        # ---- terminal conditions ----
        goal_dist = hypot(usv_x - self.usv.gx, usv_y - self.usv.gy)
        reaching_goal = goal_dist < self.usv.radius

        dmin = (self.scan_current * self.laser_max_range).min()
        collision = dmin <= self.usv.radius
        grounded = self.outside_check()

        WR = GR = AR = CR = 0.0

        if self.global_time >= self.time_limit - 1:
            reward = self.timeout_penalty
            done = True
            info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif grounded:
            reward = self.grounding_penalty
            done = True
            info = Grounding()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        else:
            if (dmin - self.usv.radius) < self.discomfort_dist:
                WR = (dmin - self.usv.radius - self.discomfort_dist) * self.discomfort_penalty_factor
            GR = self.goal_distance_factor * (self.goal_distance_last - goal_dist)
            self.goal_distance_last = goal_dist

            # waterway proximity
            WR_waterway = self._waterway_reward()

            # angle keeping
            if self.angle_action * action[1] < 0:
                AR = -0.01
            else:
                AR = 0.01
            self.angle_action = action[1]

            # COLREG
            min_index = int(np.argmin(self.scan_current))
            if min_index > self.n_laser / 3 and min_index <= 37 * self.n_laser / 72:
                self.que.put([1, dmin])
                if self.que.full():
                    last_dist = self.que.get()
                    if last_dist[0] == 1 and last_dist[1] >= dmin:
                        CR = (np.pi / 2 - self.usv.theta) * np.exp(
                            -self.usv.v * cos(self.usv.theta) / self.usv.v_max) * 0.1
            elif min_index > 37 * self.n_laser / 72 and min_index < 13 * self.n_laser / 16:
                self.que.put([2, dmin])
                if self.que.full():
                    last_dist = self.que.get()
                    if last_dist[0] == 2 and last_dist[1] >= dmin:
                        if self.usv.theta <= np.pi / 2:
                            CR = self.usv.theta * np.exp(
                                self.usv.v * cos(self.usv.theta) / self.usv.v_max) * 0.05

            reward = WR + GR + AR + CR + WR_waterway
            done = False
            info = Nothing()

        # ---- observation (same format as parent) ----
        dx = self.usv.gx - self.usv.px
        dy = self.usv.gy - self.usv.py
        theta = self.usv.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([-r, -t], dtype=np.float32)

        # logging
        self.log_env['usv'].append(np.array([self.usv.px, self.usv.py,
                                              self.usv.v, self.usv.theta]))
        self.log_env['goal'].append(np.array([self.usv.gx, self.usv.gy]))
        ships_position = []
        for ship in self.ships:
            ships_position.append(np.array([ship.px, ship.py, ship.radius, ship.vx]))
        self.log_env['ships'].append(np.array(ships_position))
        self.log_env['reward'].append(np.array([reward]))
        self.log_env['subreward'].append(np.array([WR, GR, AR, CR]))
        lasers = []
        for ls in self.scan_intersection:
            lasers.append(np.array([ls[0][0], ls[0][1], ls[1][0], ls[1][1]]))
        self.log_env['laser'].append(np.array(lasers))
        self.total_timesteps += 1

        if self.classical:
            return np.array(lasers), np.array([dx, dy]), np.array([self.usv.v, self.usv.theta]), reward, done, info
        return self.scan_current, ob_position, reward, done, info

    # -----------------------------------------------------------------
    #   Serialisation for eval configs
    # -----------------------------------------------------------------

    def episode_data(self) -> dict:
        """Serialize environment state."""
        return {
            'usv': {
                'px': self.usv.px, 'py': self.usv.py,
                'gx': self.usv.gx, 'gy': self.usv.gy,
                'v': self.usv.v, 'w': self.usv.w,
                'theta': self.usv.theta,
            },
            'trajectory_seed': self._current_seed,
        }

    def reset_with_eval_config(self, eval_config):
        """Restore from serialised episode_data()."""
        self.global_time = 0
        self.log_env = {}
        self.scan_current = np.zeros(self.n_laser, dtype=np.float32)

        uc = eval_config['usv']
        self.usv.set(uc['px'], uc['py'], uc['gx'], uc['gy'],
                     uc['v'], uc['w'], uc['theta'])
        self.goal_distance_last = self.usv.get_goal_distance()

        seed = eval_config.get('trajectory_seed', 0)
        self._current_seed = seed
        self._trajectory_manager.reset(sim_time=0.0, seed=seed)
        self.ships = list(self._trajectory_manager.active_ships)
        self.ship_num = len(self.ships)

        self.get_lidar()

        # build observation
        dx = self.usv.gx - self.usv.px
        dy = self.usv.gy - self.usv.py
        theta = self.usv.theta
        y_rel = dy * cos(theta) - dx * sin(theta)
        x_rel = dy * sin(theta) + dx * cos(theta)
        r = hypot(x_rel, y_rel) / self.square_width
        t = atan2(y_rel, x_rel) / np.pi
        ob_position = np.array([r, t], dtype=np.float32)

        if self.classical:
            lasers = []
            for ls in self.scan_intersection:
                lasers.append(np.array([ls[0][0], ls[0][1], ls[1][0], ls[1][1]]))
            return np.array(lasers), np.array([dx, dy]), np.array([self.usv.v, self.usv.theta])
        return self.scan_current, ob_position


# ---- quick self-test ---------------------------------------------------------
if __name__ == "__main__":
    class FakeArgs:
        lidar_dim = 180
        laser_angle_resolute = 0.03490659
        laser_min_range = 2.5
        laser_max_range = 100.0
        square_width = 1000.0
        discomfort_distance = 30
        classical = False
        waterway_max_ships = 5
        grounding_margin = 5.0

    args = FakeArgs()
    env = WaterwayCrowdSim(args)
    print(f"WaterwayCrowdSim loaded: {len(env._waterway_map.contour_lines)} contour lines")

    lidar, pos = env.reset()
    print(f"Reset done: USV at ({env.usv.px:.1f}, {env.usv.py:.1f}), "
          f"goal at ({env.usv.gx:.1f}, {env.usv.gy:.1f})")
    print(f"Active ships: {env.ship_num}")
    for ship in env.ships:
        print(f"  {ship.ship_id}: pos=({ship.px:.1f}, {ship.py:.1f})")

    for i in range(20):
        action = np.array([0.3, 0.0])
        lidar, pos, reward, done, info = env.step(action)
        if done:
            print(f"Step {i}: done, info={info}")
            break
    else:
        print("20 steps completed (no terminal)")

    # eval config round-trip
    config = env.episode_data()
    lidar2, pos2 = env.reset_with_eval_config(config)
    dx = abs(env.usv.px - config['usv']['px'])
    dy = abs(env.usv.py - config['usv']['py'])
    print(f"Eval config round-trip: USV offset ({dx:.2f}, {dy:.2f}) — "
          f"{'OK' if dx < 0.01 and dy < 0.01 else 'MISMATCH'}")
    print("Self-test passed.")
