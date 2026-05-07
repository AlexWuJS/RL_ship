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


# ---- quick self-test ------------------------------------------------------
if __name__ == "__main__":
    wm = WaterwayMap()
    print(f"Map grid shape ....... {wm.shape}")
    print(f"Contour line segments   {len(wm.contour_lines)}")
    print(f"Distance transform ..   {wm.distance_transform.shape}")
    c, r = 71.4, 4.5
    x, y = wm.pixel_to_sim(c, r)
    print(f"pixel ({c:.1f}, {r:.1f}) -> sim ({x:.2f}, {y:.2f})")
    print(f"dist at sim (0, 0) ..   {wm.dist_to_nearest_obstacle(0, 0):.2f} px")
    print(f"dist at sim (-22.4, -50.9) .. {wm.dist_to_nearest_obstacle(-22.4, -50.9):.2f} px")
