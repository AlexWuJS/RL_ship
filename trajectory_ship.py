"""
Trajectory-driven dynamic obstacle vessel module.

Loads AIS vessel trajectories from ais_scenario.json, fits cubic splines
for smooth interpolation, and manages a set of active obstacle ships
compatible with the CrowdSim environment's Ship interface.
"""
import json
import os
import random
from math import atan2, hypot
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

from utils.state import FullState, ObservableState
from utils.ship import Ship

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_TRAJECTORY_PATH = os.path.join(_DATA_DIR, "processed", "trajectories", "ais_scenario.json")


class TrajectoryShip(Ship):
    """A vessel that follows a pre-recorded AIS trajectory via cubic spline."""

    def __init__(self, trajectory_data: dict, time_step: float = 1.0):
        super().__init__(time_step=time_step)

        self.ship_id = trajectory_data["id"]
        self.radius = trajectory_data["radius"]
        self.start_time = trajectory_data["start_time"]
        self.end_time = trajectory_data["end_time"]

        # Extract waypoints
        pts = trajectory_data["trajectory"]
        self._t = np.array([p["t"] for p in pts], dtype=np.float64)
        self._x = np.array([p["x"] for p in pts], dtype=np.float64)
        self._y = np.array([p["y"] for p in pts], dtype=np.float64)
        self._original_len = len(pts)

        # ---- cubic spline interpolation ----
        _bc = "natural"          # natural boundary (zero second derivative)
        self._spline_x = CubicSpline(self._t, self._x, bc_type=_bc)
        self._spline_y = CubicSpline(self._t, self._y, bc_type=_bc)
        # First derivatives → velocity
        self._spline_vx = self._spline_x.derivative(1)
        self._spline_vy = self._spline_y.derivative(1)

        # Active state
        self._active = False
        self._current_t = 0.0

        # Policy is None (no ORCA) — movement is spline-driven
        self.policy = None

    # ---- public query helpers ----------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        self._active = value

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    # ---- synchronise to simulation time -----------------------------------

    def sync(self, sim_time: float):
        """Set the ship state from the spline at the given simulation time.

        ``sim_time`` is the global environment clock.  The ship's internal
        trajectory time is ``sim_time - self.start_time``.
        """
        t_local = sim_time - self.start_time
        t_local = float(np.clip(t_local, self._t[0], self._t[-1]))
        self._current_t = t_local

        px = float(self._spline_x(t_local))
        py = float(self._spline_y(t_local))
        vx = float(self._spline_vx(t_local))
        vy = float(self._spline_vy(t_local))
        theta = atan2(vy, vx)

        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.theta = theta

        # Set goal to current trajectory endpoint (for reached_destination)
        self.gx = float(self._x[-1])
        self.gy = float(self._y[-1])

    # ---- Ship interface overrides -------------------------------------------

    def act(self, ob):
        """Return the spline-computed velocity as the action."""
        return [self.vx, self.vy]

    def update_states(self, action, sfm=False):
        """Trajectory ships are updated via sync(), not external actions."""
        pass

    def reached_destination(self) -> bool:
        return self._current_t >= self._t[-1]

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius,
                         self.gx, self.gy, self.v_pref, self.theta)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        # Override to no-op — state is set via sync()
        pass

    # ---- metadata -----------------------------------------------------------

    def __repr__(self):
        return (f"TrajectoryShip({self.ship_id}, "
                f"t=[{self.start_time:.0f},{self.end_time:.0f}], "
                f"pts={self._original_len})")


class TrajectoryManager:
    """Manages a pool of TrajectoryShip instances and handles activation."""

    def __init__(self, max_active: int = 5, time_step: float = 1.0):
        self.max_active = max_active
        self.time_step = time_step

        # Load all trajectories from JSON
        with open(_TRAJECTORY_PATH, "r") as f:
            raw = json.load(f)

        self._config = raw
        self._all_ships: list[TrajectoryShip] = [
            TrajectoryShip(obs, time_step=time_step)
            for obs in raw["obstacles"]
        ]

        # Active ships
        self._active_ships: list[TrajectoryShip] = []

    # ---- properties ---------------------------------------------------------

    @property
    def active_ships(self) -> list[TrajectoryShip]:
        return self._active_ships

    @property
    def all_ships(self) -> list[TrajectoryShip]:
        return self._all_ships

    @property
    def config(self) -> dict:
        return self._config

    # ---- lifecycle ----------------------------------------------------------

    def reset(self, sim_time: float = 0.0, seed: Optional[int] = None):
        """Reset and activate a random subset of trajectories."""
        if seed is not None:
            random.seed(seed)

        for ship in self._all_ships:
            ship.active = False

        # Pick max_active ships randomly (prefer ones with shorter duration
        # for variety during training).
        candidates = [s for s in self._all_ships]
        random.shuffle(candidates)
        selected = candidates[:self.max_active]

        self._active_ships = []
        for ship in selected:
            ship.active = True
            ship.sync(sim_time)
            self._active_ships.append(ship)

    def step(self, sim_time: float):
        """Update all active ships to the given simulation time."""
        still_active = []
        for ship in self._active_ships:
            ship.sync(sim_time)
            if not ship.reached_destination():
                still_active.append(ship)

        # If ships finished, activate replacements from the pool
        vacancies = self.max_active - len(still_active)
        if vacancies > 0:
            inactive = [s for s in self._all_ships
                        if not s.active and s.start_time <= sim_time]
            random.shuffle(inactive)
            for ship in inactive[:vacancies]:
                ship.active = True
                ship.sync(sim_time)
                still_active.append(ship)

        self._active_ships = still_active

    # ---- serialisation (for eval config, mirrors CrowdSim.episode_data) -----

    def episode_data(self) -> dict:
        """Return a dict compatible with CrowdSim's eval config format."""
        data = {"ships": {"px": [], "py": [], "px_end": [],
                          "py_end": [], "v_pref": [], "radius": []}}
        for ship in self._active_ships:
            data["ships"]["px"].append(ship.px)
            data["ships"]["py"].append(ship.py)
            data["ships"]["px_end"].append(ship.gx)
            data["ships"]["py_end"].append(ship.gy)
            data["ships"]["v_pref"].append(ship.v_pref)
            data["ships"]["radius"].append(ship.radius)
        return data


# ---- quick self-test --------------------------------------------------------
if __name__ == "__main__":
    mgr = TrajectoryManager(max_active=5)
    mgr.reset(sim_time=0.0)

    print(f"TrajectoryManager loaded {len(mgr.all_ships)} ship trajectories.")
    print(f"Active ships at t=0: {len(mgr.active_ships)}")
    for s in mgr.active_ships:
        print(f"  {s.ship_id}: pos=({s.px:.1f}, {s.py:.1f}) "
              f"vel=({s.vx:.3f}, {s.vy:.3f}) theta={np.degrees(s.theta):.0f}")

    # Advance and check
    for t in [120, 600, 1200]:
        mgr.step(float(t))
        print(f"t={t}s: {len(mgr.active_ships)} active")
        for s in mgr.active_ships[:3]:
            print(f"  {s.ship_id}: pos=({s.px:.1f}, {s.py:.1f})")

    # Verify spline continuity (velocities should be smooth)
    print("\nSpline smoothness test (1st vessel, derivative continuity):")
    s = mgr.active_ships[0]
    for t in [0, 60, 120, 180]:
        s.sync(t)
        print(f"  t={t:3.0f}: x={s.px:.4f} y={s.py:.4f} vx={s.vx:.4f} vy={s.vy:.4f}")
