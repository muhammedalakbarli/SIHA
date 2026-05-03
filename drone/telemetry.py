"""Live drone telemetry data model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List


# GPS fix type constants (MAVLink GPS_FIX_TYPE)
GPS_FIX_NONE = 0
GPS_FIX_NO_FIX = 1
GPS_FIX_2D = 2
GPS_FIX_3D = 3
GPS_FIX_DGPS = 4
GPS_FIX_RTK_FLOAT = 5
GPS_FIX_RTK_FIXED = 6

GPS_FIX_LABELS = {
    GPS_FIX_NONE:      "NO GPS",
    GPS_FIX_NO_FIX:    "NO FIX",
    GPS_FIX_2D:        "2D FIX",
    GPS_FIX_3D:        "3D FIX",
    GPS_FIX_DGPS:      "DGPS",
    GPS_FIX_RTK_FLOAT: "RTK FLT",
    GPS_FIX_RTK_FIXED: "RTK FIX",
}


@dataclass
class Telemetry:
    """All live telemetry fields updated by :class:`~drone.mavlink_client.MAVLinkClient`.

    Default values are representative of a grounded drone at Baku, Azerbaijan.
    In demo mode (no MAVLink connection) these are used as-is, allowing the
    FPV HUD to be tested without real hardware.
    """

    # ── Position ──────────────────────────────────────────────────────────
    lat: float = 40.4093
    lon: float = 49.8671
    altitude_rel: float = 0.0    # metres above home (AGL)
    altitude_abs: float = 0.0    # metres above sea level (MSL)

    # ── Velocity ──────────────────────────────────────────────────────────
    groundspeed: float = 0.0     # m/s
    airspeed: float = 0.0        # m/s
    vertical_speed: float = 0.0  # m/s  (positive = climb)

    # ── Attitude (degrees) ────────────────────────────────────────────────
    roll: float = 0.0            # degrees, right = positive
    pitch: float = 0.0           # degrees, nose-up = positive
    yaw: float = 0.0             # degrees (0–360, clockwise)
    heading: int = 0             # magnetic heading (0–359)

    # ── Power ─────────────────────────────────────────────────────────────
    battery_voltage: float = 0.0    # volts
    battery_remaining: int = 100    # percent (0–100)
    throttle: int = 0               # percent (0–100)

    # ── GPS ───────────────────────────────────────────────────────────────
    gps_fix: int = GPS_FIX_NONE
    gps_satellites: int = 0

    # ── RC ────────────────────────────────────────────────────────────────
    rc_channels: List[int] = field(default_factory=lambda: [1500] * 8)
    rssi: int = 0                   # receiver signal strength (0–255)

    # ── Flight status ─────────────────────────────────────────────────────
    flight_mode: str = "STABILIZE"
    armed: bool = False
    connected: bool = False
    last_heartbeat: float = field(default_factory=time.time)

    # ── Convenience ───────────────────────────────────────────────────────

    @property
    def gps_fix_label(self) -> str:
        return GPS_FIX_LABELS.get(self.gps_fix, "UNK")

    @property
    def groundspeed_kmh(self) -> float:
        return self.groundspeed * 3.6

    @property
    def heartbeat_age(self) -> float:
        """Seconds since last heartbeat received."""
        return time.time() - self.last_heartbeat
