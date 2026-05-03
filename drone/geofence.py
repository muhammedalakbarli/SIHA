"""Geofencing – polygon containment check and breach actions.

A Geofence defines a 2-D polygon in geographic coordinates (lat/lon).
When the drone flies outside the polygon an action (callback or RTL) is
triggered.

Example::

    fence = Geofence(
        polygon=[(40.41, 49.86), (40.41, 49.88), (40.43, 49.88), (40.43, 49.86)],
        on_breach=lambda: client.return_to_launch(),
    )
    # In the telemetry loop:
    fence.check(telem)
"""

from __future__ import annotations

import logging
import math
from typing import Callable, List, Optional, Tuple

from drone.telemetry import Telemetry

logger = logging.getLogger(__name__)

LatLon = Tuple[float, float]


class Geofence:
    """Polygon geofence with configurable breach action.

    Args:
        polygon:    Ordered list of (lat, lon) vertices defining the boundary.
                    The polygon is automatically closed (last → first edge).
        on_breach:  Callable invoked once when the drone first exits the polygon.
                    Re-armed after the drone re-enters.
        min_alt_m:  Optional minimum altitude (AGL).  Below this the breach
                    action is also triggered.
        max_alt_m:  Optional maximum altitude (AGL).  Above this the breach
                    action is triggered.

    Raises:
        ValueError: If fewer than 3 vertices are supplied.
    """

    def __init__(
        self,
        polygon: List[LatLon],
        on_breach: Optional[Callable[[], None]] = None,
        min_alt_m: Optional[float] = None,
        max_alt_m: Optional[float] = None,
    ) -> None:
        if len(polygon) < 3:
            raise ValueError("Geofence requires at least 3 vertices")
        self._polygon   = polygon
        self._on_breach = on_breach
        self._min_alt   = min_alt_m
        self._max_alt   = max_alt_m
        self._breached  = False   # edge-trigger: fire once per exit

    # ── Public ───────────────────────────────────────────────────────────

    @property
    def polygon(self) -> List[LatLon]:
        return list(self._polygon)

    def is_inside(self, lat: float, lon: float) -> bool:
        """Return True if (lat, lon) lies inside the polygon."""
        return _point_in_polygon(lat, lon, self._polygon)

    def check(self, telem: Telemetry) -> bool:
        """Evaluate the geofence against current telemetry.

        Calls ``on_breach`` on the first tick after leaving the polygon or
        violating the altitude limits.  Re-arms when the drone returns inside.

        Args:
            telem: Live telemetry object.

        Returns:
            True if currently breached, False if safe.
        """
        inside = self.is_inside(telem.lat, telem.lon)

        alt_ok = True
        if self._min_alt is not None and telem.altitude_rel < self._min_alt:
            alt_ok = False
        if self._max_alt is not None and telem.altitude_rel > self._max_alt:
            alt_ok = False

        currently_breached = not inside or not alt_ok

        if currently_breached and not self._breached:
            logger.warning(
                "GEOFENCE BREACH: lat=%.5f lon=%.5f alt=%.1fm",
                telem.lat, telem.lon, telem.altitude_rel,
            )
            if self._on_breach:
                try:
                    self._on_breach()
                except Exception:
                    logger.exception("Geofence breach callback failed")

        self._breached = currently_breached
        return currently_breached

    def distance_to_boundary(self, lat: float, lon: float) -> float:
        """Return approximate distance in metres to the nearest boundary edge."""
        min_d = math.inf
        n = len(self._polygon)
        for i in range(n):
            a = self._polygon[i]
            b = self._polygon[(i + 1) % n]
            d = _dist_point_to_segment(lat, lon, a, b)
            if d < min_d:
                min_d = d
        return min_d


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _point_in_polygon(lat: float, lon: float, polygon: List[LatLon]) -> bool:
    """Ray-casting algorithm for point-in-polygon test."""
    n      = len(polygon)
    inside = False
    j      = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def _latlon_to_metres(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in metres between two lat/lon points."""
    R   = 6_371_000.0
    φ1  = math.radians(lat1)
    φ2  = math.radians(lat2)
    dφ  = math.radians(lat2 - lat1)
    dλ  = math.radians(lon2 - lon1)
    a   = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _dist_point_to_segment(
    lat: float, lon: float, a: LatLon, b: LatLon
) -> float:
    """Approximate distance from point (lat, lon) to segment a–b in metres."""
    # Project onto segment, clamp to [0, 1]
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return _latlon_to_metres(lat, lon, ax, ay)
    t = ((lat - ax) * dx + (lon - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px = ax + t * dx
    py = ay + t * dy
    return _latlon_to_metres(lat, lon, px, py)
