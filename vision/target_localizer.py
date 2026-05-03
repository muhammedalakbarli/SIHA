"""GPS coordinate estimation for detected targets.

Given the drone's position, attitude, camera field-of-view, and the pixel
coordinates of a detected object, calculates the approximate geographic
coordinates (lat/lon) of that object on the ground.

The algorithm uses a flat-earth approximation (accurate for distances < 5 km)
and assumes the camera is mounted on a stabilised gimbal or rigidly to the
drone body.

Example::

    loc = TargetLocalizer(fov_h_deg=62.2, fov_v_deg=48.8)
    result = loc.localize(
        drone_lat=40.4093, drone_lon=49.8671,
        drone_alt_m=80.0,
        drone_yaw_deg=45.0, drone_pitch_deg=-2.0, drone_roll_deg=1.5,
        target_px=400, target_py=350,
        frame_w=1280, frame_h=720,
        gimbal_pitch_deg=-75.0,
    )
    if result:
        print(f"Target at {result.lat:.5f}, {result.lon:.5f}, dist={result.slant_m:.0f}m")
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class LocalizationResult:
    """Result of a single target localisation."""

    lat:     float    # estimated latitude
    lon:     float    # estimated longitude
    slant_m: float    # slant range from drone to target in metres
    az_deg:  float    # bearing from drone to target (0 = North, clockwise)


class TargetLocalizer:
    """Estimates the GPS position of a detected target.

    Args:
        fov_h_deg: Camera horizontal field of view in degrees.
                   Common values: webcam ≈ 62°, GoPro ≈ 118°, zoom lens ≈ 30°.
        fov_v_deg: Camera vertical field of view in degrees.
                   If omitted, inferred from fov_h and the frame aspect ratio.
    """

    # Earth semi-major axis (metres)
    _R = 6_371_000.0

    def __init__(
        self,
        fov_h_deg: float = 62.2,
        fov_v_deg: Optional[float] = None,
    ) -> None:
        self._fov_h = math.radians(fov_h_deg)
        self._fov_v = math.radians(fov_v_deg) if fov_v_deg else None

    # ── Public ───────────────────────────────────────────────────────────

    def localize(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_alt_m: float,
        drone_yaw_deg: float,
        drone_pitch_deg: float,
        drone_roll_deg: float,
        target_px: int,
        target_py: int,
        frame_w: int,
        frame_h: int,
        gimbal_pitch_deg: float = -90.0,
    ) -> Optional[LocalizationResult]:
        """Calculate the GPS position of a pixel in the camera frame.

        Args:
            drone_lat:        Drone latitude (degrees).
            drone_lon:        Drone longitude (degrees).
            drone_alt_m:      Drone altitude AGL in metres.
            drone_yaw_deg:    Drone magnetic heading (0–360).
            drone_pitch_deg:  Drone pitch angle (positive = nose up).
            drone_roll_deg:   Drone roll angle (positive = right wing down).
            target_px:        Target pixel x-coordinate (0 = left edge).
            target_py:        Target pixel y-coordinate (0 = top edge).
            frame_w:          Frame width in pixels.
            frame_h:          Frame height in pixels.
            gimbal_pitch_deg: Gimbal pitch relative to drone body (−90 = nadir).

        Returns:
            :class:`LocalizationResult` or ``None`` if the ray does not
            intersect the ground (e.g. camera pointing above horizon).
        """
        if drone_alt_m <= 0.1:
            return None

        # ── Camera FOV ────────────────────────────────────────────────────
        fov_h = self._fov_h
        fov_v = self._fov_v if self._fov_v else fov_h * (frame_h / frame_w)

        # Normalised pixel offsets (−1 … +1, (0,0) = frame centre)
        nx = (target_px - frame_w / 2.0) / (frame_w / 2.0)
        ny = (target_py - frame_h / 2.0) / (frame_h / 2.0)

        # Ray angles in camera frame (radians)
        az_cam = nx * (fov_h / 2.0)   # positive = right
        el_cam = ny * (fov_v / 2.0)   # positive = down in image

        # ── Combine gimbal pitch + pixel elevation ────────────────────────
        # Gimbal pitch: −90° = straight down, 0° = horizontal
        gimbal_pitch_rad = math.radians(gimbal_pitch_deg)
        # Total elevation below horizontal
        el_total = -(gimbal_pitch_rad + el_cam)   # positive = downward

        if el_total <= 0:
            return None   # ray points upward, no ground intersection

        # ── Slant range (flat earth) ──────────────────────────────────────
        slant_m = drone_alt_m / math.sin(el_total)

        # ── Horizontal offset in rotated camera frame ─────────────────────
        horiz_m = drone_alt_m / math.tan(el_total)
        right_m = horiz_m * math.tan(az_cam)

        # ── Rotate from camera/body frame to NED ─────────────────────────
        # Apply drone roll and pitch corrections (small-angle approximation)
        roll_rad  = math.radians(drone_roll_deg)
        pitch_rad = math.radians(drone_pitch_deg)

        fwd_m   = horiz_m - horiz_m * pitch_rad   # pitch correction
        right_m = right_m - horiz_m * roll_rad    # roll correction

        # Rotate by drone yaw to get NED offsets
        yaw_rad    = math.radians(drone_yaw_deg)
        north_m    = fwd_m   * math.cos(yaw_rad) - right_m * math.sin(yaw_rad)
        east_m     = fwd_m   * math.sin(yaw_rad) + right_m * math.cos(yaw_rad)

        # ── NED offsets → lat/lon ─────────────────────────────────────────
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(drone_lat))

        target_lat = drone_lat + north_m / m_per_deg_lat
        target_lon = drone_lon + east_m  / m_per_deg_lon

        # Bearing from drone to target
        dy = east_m
        dx = north_m
        bearing = (math.degrees(math.atan2(dy, dx)) + 360) % 360

        return LocalizationResult(
            lat=target_lat,
            lon=target_lon,
            slant_m=slant_m,
            az_deg=bearing,
        )

    def localize_centre(
        self,
        drone_lat: float,
        drone_lon: float,
        drone_alt_m: float,
        drone_yaw_deg: float,
        gimbal_pitch_deg: float = -90.0,
    ) -> Optional[LocalizationResult]:
        """Localise the frame centre (convenience wrapper, no frame needed)."""
        return self.localize(
            drone_lat, drone_lon, drone_alt_m,
            drone_yaw_deg, 0.0, 0.0,
            0, 0, 1, 1,
            gimbal_pitch_deg=gimbal_pitch_deg,
        )
