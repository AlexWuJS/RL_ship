"""
Waterway (channel) static map module.

Loads the grid occupancy map, extracts obstacle contours as simplified
polygons, and provides coordinate transforms between pixel and simulation
coordinate frames. Also computes a distance transform for reward shaping.
"""
import os
from math import hypot

import cv2
import numpy as np


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


class WaterwayMap:
    """Manages the static waterway channel map."""

    def __init__(self, contour_simplification: float = 1.0):
        map_path = os.path.join(_DATA_DIR, "processed", "maps", "navigation_map.npy")
        meta_path = os.path.join(_DATA_DIR, "processed", "maps", "navigation_map_meta.yaml")

        self._grid: np.ndarray = np.load(map_path)          # uint8, (H, W), 0=free 1=occupied
        self._H, self._W = self._grid.shape

        with open(meta_path, "r") as f:
            raw = f.read()
        self._meta = self._parse_yaml(raw)

        # ------------------------------------------------------------------
        # Coordinate mapping  (derived from trajectory data fitting):
        #   x_sim = 1.2059 * col - 108.5286
        #   y_sim = 1.2567 * row - 56.5506
        # with col ∈ [0, W-1], row ∈ [0, H-1].
        # ------------------------------------------------------------------
        self._col_to_x = 1.205873
        self._x_offset = -108.5286
        self._row_to_y = 1.256679
        self._y_offset = -56.5506

        # Precompute obstacle contour line segments (in sim coordinates)
        self._contour_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
        self._extract_contours(contour_simplification)

        # Precompute distance transform for reward shaping
        self._distance_transform: np.ndarray = self._compute_distance_transform()

        # Precompute snap offsets: for each pixel, the vector to the nearest water pixel
        self._snap_offset: np.ndarray = self._compute_snap_offsets()

    # ---- public helpers ---------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Occupancy grid, shape (H, W), 0=free 1=occupied."""
        return self._grid

    @property
    def shape(self):
        """Grid shape (H, W)."""
        return self._H, self._W

    @property
    def contour_lines(self):
        """Line segments of obstacle contours in simulation coordinates."""
        return self._contour_lines

    @property
    def distance_transform(self) -> np.ndarray:
        """Distance from each free pixel to the nearest obstacle (pixels)."""
        return self._distance_transform

    # ---- coordinate transforms --------------------------------------------

    def sim_to_pixel(self, x: float, y: float):
        """Convert simulation (x, y) to pixel (col, row)."""
        col = (x - self._x_offset) / self._col_to_x
        row = (y - self._y_offset) / self._row_to_y
        return col, row

    def pixel_to_sim(self, col: float, row: float):
        """Convert pixel (col, row) to simulation (x, y)."""
        x = self._col_to_x * col + self._x_offset
        y = self._row_to_y * row + self._y_offset
        return x, y

    def dist_to_nearest_obstacle(self, x: float, y: float) -> float:
        """Distance (in pixels) from (x, y) to nearest obstacle.

        Returns pixel-distance; multiply by ~0.123 m/pixel for metres.
        Returns 0 if outside the map.
        """
        col, row = self.sim_to_pixel(x, y)
        c, r = int(round(col)), int(round(row))
        if 0 <= c < self._W and 0 <= r < self._H:
            return float(self._distance_transform[r, c])
        return 0.0

    def is_outside_map(self, x: float, y: float) -> bool:
        """Whether (x, y) is outside the map bounds."""
        col, row = self.sim_to_pixel(x, y)
        return not (0 <= col < self._W and 0 <= row < self._H)

    def snap_to_water(self, x: float, y: float) -> tuple[float, float]:
        """Snap an (x, y) position to the nearest water (free) pixel.

        Returns unchanged (x, y) if already on water or outside map.
        """
        col, row = self.sim_to_pixel(x, y)
        c, r = int(round(col)), int(round(row))
        if not (0 <= c < self._W and 0 <= r < self._H):
            return x, y
        if self._grid[r, c] == 0:          # already on water
            return x, y
        dc, dr = self._snap_offset[r, c, 0], self._snap_offset[r, c, 1]
        return self.pixel_to_sim(c + dc, r + dr)

    # ---- trajectory-level correction ---------------------------------------

    def correct_trajectory_sim(self, x_array: np.ndarray, y_array: np.ndarray,
                                method: str = "auto"):
        """Apply a smooth global transform to bring an entire trajectory onto water.

        Args:
            x_array, y_array: 1D arrays of simulation coordinates.
            method: ``"auto"`` (try translation → warp → per-point snap),
                    ``"translation"``, or ``"warp"``.

        Returns:
            (x_corrected, y_corrected, diagnostics) — corrected sim-coordinate
            arrays of the same length, plus a dict with correction details.
        """
        cols, rows = self.sim_to_pixel(
            np.asarray(x_array, dtype=np.float64),
            np.asarray(y_array, dtype=np.float64),
        )
        pixel_coords = np.column_stack([cols, rows])               # (N, 2)
        corrected_px, diag = self.correct_trajectory_pixel(pixel_coords, method)
        x_corr, y_corr = self.pixel_to_sim(corrected_px[:, 0], corrected_px[:, 1])
        return x_corr, y_corr, diag

    def correct_trajectory_pixel(self, pixel_coords: np.ndarray,
                                  method: str = "auto"):
        """Orchestrate trajectory correction with fallback chain.

        Returns (corrected_coords, diagnostics).  *corrected_coords* has the
        same shape as *pixel_coords*.
        """
        diag = {"method": "none", "n_total": len(pixel_coords)}
        diag["obstacle_before"] = int(self._count_on_obstacle(pixel_coords))

        if diag["obstacle_before"] == 0:
            diag["obstacle_after"] = 0
            return np.copy(pixel_coords), diag

        # ---- Stage 1: optimal translation -----------------------------------
        if method in ("auto", "translation"):
            dc, dr, cost = self._find_best_translation(pixel_coords, max_offset=15)
            translated = pixel_coords + np.array([dc, dr], dtype=np.float64)
            obs_after = int(self._count_on_obstacle(translated))
            obs_ratio = obs_after / len(pixel_coords)
            if obs_ratio <= 0.10:                                # ≤10% residual
                diag["method"] = "translation"
                diag["dc"] = dc
                diag["dr"] = dr
                diag["obstacle_after"] = obs_after
                return self._clip_to_map(translated), diag

        # ---- Stage 2: per-point snap (last resort) --------------------------
        # Only used when translation leaves >10% on obstacles.
        snapped = np.copy(pixel_coords)
        for i in range(len(snapped)):
            c, r = snapped[i]
            ci, ri = int(round(c)), int(round(r))
            if 0 <= ci < self._W and 0 <= ri < self._H and self._grid[ri, ci] == 1:
                dc = self._snap_offset[ri, ci, 0]
                dr = self._snap_offset[ri, ci, 1]
                snapped[i, 0] = ci + dc
                snapped[i, 1] = ri + dr
        diag["method"] = "fallback_snap"
        diag["obstacle_after"] = int(self._count_on_obstacle(snapped))
        return self._clip_to_map(snapped), diag

    def _find_best_translation(self, pixel_coords: np.ndarray,
                                max_offset: int = 15):
        """Grid-search for the (dc, dr) that minimises obstacle collisions.

        Cost = ``n_obstacle + 2.0 * n_oob + 0.01 * (|dc| + |dr|)``.
        Out-of-bounds points are penalised heavily so translations that push
        the trajectory off the map are avoided.

        Returns ``(dc, dr, best_cost)``.
        """
        best_dc, best_dr = 0, 0
        best_cost = float("inf")

        c_floor = np.floor(pixel_coords[:, 0]).astype(np.int64)
        r_floor = np.floor(pixel_coords[:, 1]).astype(np.int64)

        for dc in range(-max_offset, max_offset + 1):
            for dr in range(-max_offset, max_offset + 1):
                ci = c_floor + dc
                ri = r_floor + dr
                valid = (0 <= ci) & (ci < self._W) & (0 <= ri) & (ri < self._H)
                n_oob = len(pixel_coords) - int(valid.sum())
                obs_count = int(np.sum(self._grid[ri[valid], ci[valid]]))
                cost = obs_count + 2.0 * n_oob + 0.01 * (abs(dc) + abs(dr))
                if cost < best_cost:
                    best_cost = cost
                    best_dc, best_dr = dc, dr
                    if obs_count == 0 and n_oob == 0:
                        return best_dc, best_dr, best_cost

        return best_dc, best_dr, best_cost

    def _fit_polynomial_warp(self, pixel_coords: np.ndarray, degree: int = 2):
        """Fit a low-degree polynomial warp along the trajectory arc length.

        Uses the precomputed ``_snap_offset`` as raw targets then fits
        ``dc(s)`` and ``dr(s)`` with weighted least-squares.  The arc-length
        parameter *s* ∈ [0, 1] ensures the deformation varies smoothly along
        the trajectory regardless of speed variations in the original data.

        Returns ``(corrected_coords, (dc_coeffs, dr_coeffs))``.
        """
        from numpy.polynomial.polynomial import Polynomial

        cols = pixel_coords[:, 0]
        rows = pixel_coords[:, 1]

        # Arc-length parameterisation
        ds = np.sqrt(np.diff(cols) ** 2 + np.diff(rows) ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        if s[-1] > 1e-10:
            s = s / s[-1]                         # normalise to [0, 1]

        # Raw target offsets from nearest-water map
        dc_raw = np.zeros(len(cols), dtype=np.float64)
        dr_raw = np.zeros(len(rows), dtype=np.float64)
        weights = np.ones(len(cols), dtype=np.float64)

        for i in range(len(cols)):
            ci, ri = int(round(cols[i])), int(round(rows[i]))
            if 0 <= ci < self._W and 0 <= ri < self._H:
                if self._grid[ri, ci] == 1:
                    dc_raw[i] = self._snap_offset[ri, ci, 0]
                    dr_raw[i] = self._snap_offset[ri, ci, 1]
                    dist = self._distance_transform[ri, ci]
                    weights[i] = max(1.0, float(dist))   # more weight to points further from water

        # Fit polynomials with weighted least-squares
        fit_dc = Polynomial.fit(s, dc_raw, degree, w=weights)
        fit_dr = Polynomial.fit(s, dr_raw, degree, w=weights)

        dc_smooth = fit_dc(s)
        dr_smooth = fit_dr(s)

        corrected = np.column_stack([
            cols + dc_smooth,
            rows + dr_smooth,
        ])

        return corrected, ((fit_dc.coef, fit_dr.coef))

    def verify_trajectory(self, pixel_coords: np.ndarray) -> dict:
        """Return diagnostic metrics for a trajectory in pixel coordinates."""
        result = {
            "n_total": len(pixel_coords),
            "n_obstacle": int(self._count_on_obstacle(pixel_coords)),
        }

        # Shape fidelity: relative change in consecutive distances
        if len(pixel_coords) >= 2:
            diffs = np.diff(pixel_coords, axis=0)
            dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
            result["mean_step"] = float(np.mean(dists))
            result["max_step"] = float(np.max(dists))

        return result

    # ---- internal helpers ---------------------------------------------------

    def _count_on_obstacle(self, pixel_coords: np.ndarray) -> np.integer:
        """Count how many pixel coordinates land on obstacle grid cells."""
        ci = np.round(pixel_coords[:, 0]).astype(int)
        ri = np.round(pixel_coords[:, 1]).astype(int)
        valid = (0 <= ci) & (ci < self._W) & (0 <= ri) & (ri < self._H)
        if not valid.any():
            return np.int64(0)
        return np.sum(self._grid[ri[valid], ci[valid]])

    def _clip_to_map(self, pixel_coords: np.ndarray) -> np.ndarray:
        """Clip pixel coordinates to lie within map bounds."""
        clipped = np.copy(pixel_coords)
        clipped[:, 0] = np.clip(clipped[:, 0], 0, self._W - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, self._H - 1)
        return clipped

    # ---- private ----------------------------------------------------------

    def _parse_yaml(self, raw: str):
        meta = {}
        for line in raw.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                try:
                    meta[k] = float(v) if "." in v else int(v)
                except ValueError:
                    meta[k] = v
        return meta

    def _extract_contours(self, epsilon: float):
        binary = (self._grid * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for cnt in contours:
            if len(cnt) < 3:
                continue  # skip noise
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            n = len(approx)
            if n < 2:
                continue
            # Convert each edge of the polygon → a line segment
            for i in range(n):
                r1, c1 = approx[i][0][1], approx[i][0][0]
                r2, c2 = approx[(i + 1) % n][0][1], approx[(i + 1) % n][0][0]
                # Skip zero-length edges
                if r1 == r2 and c1 == c2:
                    continue
                x1, y1 = self.pixel_to_sim(c1, r1)
                x2, y2 = self.pixel_to_sim(c2, r2)
                lines.append([(x1, y1), (x2, y2)])

        self._contour_lines = lines

    def _compute_distance_transform(self) -> np.ndarray:
        free = (1 - self._grid).astype(np.uint8)
        return cv2.distanceTransform(free, cv2.DIST_L2, 5)

    def _compute_snap_offsets(self) -> np.ndarray:
        """Precompute (dcol, dr) offset vectors to nearest water for each pixel.

        Shape (H, W, 2).  Zero vector for pixels that are already water.
        """
        from scipy.spatial import cKDTree

        offsets = np.zeros((self._H, self._W, 2), dtype=np.float32)

        water_rc = np.argwhere(self._grid == 0)                 # (N, 2) [row, col]
        tree = cKDTree(water_rc)

        obs_rc = np.argwhere(self._grid == 1)                   # (M, 2) [row, col]
        if len(obs_rc) == 0:
            return offsets

        _, idx = tree.query(obs_rc)
        nearest = water_rc[idx]                                  # (M, 2) [row, col]

        offsets[obs_rc[:, 0], obs_rc[:, 1], 0] = nearest[:, 1] - obs_rc[:, 1]  # dcol
        offsets[obs_rc[:, 0], obs_rc[:, 1], 1] = nearest[:, 0] - obs_rc[:, 0]  # drow

        return offsets


# ---- quick self-test ------------------------------------------------------
if __name__ == "__main__":
    wm = WaterwayMap()
    print(f"Map grid shape ....... {wm.shape}")
    print(f"Contour line segments   {len(wm.contour_lines)}")
    print(f"Distance transform ..   {wm.distance_transform.shape}")
    print(f"Snap offset array ...   {wm._snap_offset.shape}")
    c, r = 71.4, 4.5
    x, y = wm.pixel_to_sim(c, r)
    print(f"pixel ({c:.1f}, {r:.1f}) -> sim ({x:.2f}, {y:.2f})")
    print(f"dist at sim (0, 0) ..   {wm.dist_to_nearest_obstacle(0, 0):.2f} px")

    # Snap test
    import json, os
    with open(os.path.join(_DATA_DIR, "processed", "trajectories", "ais_scenario.json")) as f:
        data = json.load(f)
    obs = data["obstacles"][0]
    pt = obs["trajectory"][0]
    x_raw, y_raw = pt["x"], pt["y"]
    x_snap, y_snap = wm.snap_to_water(x_raw, y_raw)
    col_r, row_r = wm.sim_to_pixel(x_raw, y_raw)
    col_s, row_s = wm.sim_to_pixel(x_snap, y_snap)
    print(f"\nSnap test: ship {obs['id']}")
    print(f"  raw  ({x_raw:.2f}, {y_raw:.2f}) -> pixel ({col_r:.1f}, {row_r:.1f}) grid={wm.grid[int(row_r), int(col_r)]}")
    print(f"  snap ({x_snap:.2f}, {y_snap:.2f}) -> pixel ({col_s:.1f}, {row_s:.1f}) grid={wm.grid[int(row_s), int(col_s)]}")
    print(f"  offset: ({x_snap-x_raw:.3f}, {y_snap-y_raw:.3f}) m")
