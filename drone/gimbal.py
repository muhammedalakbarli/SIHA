"""Gimbal / camera-mount controller.

Sends MAVLink DO_MOUNT_CONTROL commands to a drone-mounted gimbal.
Supports both angle-mode (point to specific pitch/yaw) and ROI-mode
(lock onto a GPS coordinate).

Requires an active :class:`~drone.mavlink_client.MAVLinkClient`.

Example::

    gimbal = GimbalController(client)
    gimbal.set_angles(pitch_deg=-45, yaw_deg=0)   # look 45° down, straight ahead
    gimbal.look_at(target_lat=40.412, target_lon=49.870, drone_telem=telem)
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drone.mavlink_client import MAVLinkClient
    from drone.telemetry import Telemetry

logger = logging.getLogger(__name__)

# MAVLink mount modes
_MAV_MOUNT_MODE_RETRACT      = 0
_MAV_MOUNT_MODE_NEUTRAL      = 1
_MAV_MOUNT_MODE_MAVLINK_TARGETING = 2   # angle control
_MAV_MOUNT_MODE_RC_TARGETING  = 3
_MAV_MOUNT_MODE_GPS_POINT     = 4       # ROI / GPS lock


class GimbalController:
    """Control a drone-mounted gimbal via MAVLink.

    Args:
        client:         Active MAVLink client.
        stabilised:     Whether the gimbal is stabilised (affects angle reference).
        pitch_min_deg:  Hardware pitch limit (negative = down).
        pitch_max_deg:  Hardware pitch limit (positive = up).
    """

    def __init__(
        self,
        client: "MAVLinkClient",
        stabilised: bool = True,
        pitch_min_deg: float = -90.0,
        pitch_max_deg: float = 0.0,
    ) -> None:
        self._client      = client
        self._stabilised  = stabilised
        self._pitch_min   = pitch_min_deg
        self._pitch_max   = pitch_max_deg

        # Current commanded angles
        self.pitch_deg: float = -90.0   # nadir (pointing straight down)
        self.roll_deg:  float = 0.0
        self.yaw_deg:   float = 0.0

    # ── Public ───────────────────────────────────────────────────────────

    def set_angles(
        self,
        pitch_deg: float,
        yaw_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> None:
        """Command the gimbal to specific angles.

        Args:
            pitch_deg: Pitch in degrees (negative = down, -90 = nadir).
            yaw_deg:   Yaw relative to drone heading (degrees).
            roll_deg:  Roll in degrees (usually 0).
        """
        pitch_deg = max(self._pitch_min, min(self._pitch_max, pitch_deg))
        self.pitch_deg = pitch_deg
        self.yaw_deg   = yaw_deg
        self.roll_deg  = roll_deg

        self._send_mount_control(
            pitch_deg * 100,  # centidegrees
            roll_deg  * 100,
            yaw_deg   * 100,
            mode=_MAV_MOUNT_MODE_MAVLINK_TARGETING,
        )
        logger.debug(
            "Gimbal → pitch=%.1f° yaw=%.1f° roll=%.1f°",
            pitch_deg, yaw_deg, roll_deg,
        )

    def look_at(
        self,
        target_lat: float,
        target_lon: float,
        target_alt: float = 0.0,
        drone_telem: "Telemetry | None" = None,
    ) -> None:
        """Point the gimbal at a GPS coordinate (ROI mode).

        Args:
            target_lat: Target latitude.
            target_lon: Target longitude.
            target_alt: Target altitude MSL in metres.
            drone_telem: If supplied, also updates ``pitch_deg`` / ``yaw_deg``
                         fields to reflect the commanded angles.
        """
        if drone_telem is not None:
            # Calculate angles for informational purposes
            bearing, pitch = _bearing_and_pitch(
                drone_telem.lat, drone_telem.lon, drone_telem.altitude_abs,
                target_lat, target_lon, target_alt,
            )
            self.yaw_deg   = bearing
            self.pitch_deg = pitch

        # Send GPS-point ROI command (MAVLink DO_SET_ROI)
        self._send_do_set_roi(target_lat, target_lon, target_alt)

    def nadir(self) -> None:
        """Point the gimbal straight down (nadir / 90° pitch down)."""
        self.set_angles(pitch_deg=-90.0, yaw_deg=0.0)

    def neutral(self) -> None:
        """Return gimbal to neutral / stowed position."""
        self._send_mount_control(0, 0, 0, mode=_MAV_MOUNT_MODE_NEUTRAL)

    # ── Private ──────────────────────────────────────────────────────────

    def _send_mount_control(
        self,
        pitch_cd: float,
        roll_cd: float,
        yaw_cd: float,
        mode: int,
    ) -> None:
        mav = self._client._mav
        if mav is None:
            logger.warning("Gimbal: not connected")
            return
        try:
            mav.mav.mount_control_send(
                mav.target_system,
                mav.target_component,
                int(pitch_cd),
                int(roll_cd),
                int(yaw_cd),
                0,  # save_position
            )
        except Exception:
            logger.exception("mount_control_send failed")

    def _send_do_set_roi(
        self, lat: float, lon: float, alt: float
    ) -> None:
        mav = self._client._mav
        if mav is None:
            return
        try:
            mav.mav.command_long_send(
                mav.target_system,
                mav.target_component,
                201,  # MAV_CMD_DO_SET_ROI
                0,
                0, 0, 0, 0,
                lat, lon, alt,
            )
        except Exception:
            logger.exception("DO_SET_ROI send failed")


# ── Geometry ─────────────────────────────────────────────────────────────────

def _bearing_and_pitch(
    from_lat: float, from_lon: float, from_alt: float,
    to_lat: float,   to_lon: float,   to_alt: float,
) -> tuple[float, float]:
    """Return (bearing_deg, pitch_deg) from one point to another."""
    R       = 6_371_000.0
    φ1      = math.radians(from_lat)
    φ2      = math.radians(to_lat)
    dλ      = math.radians(to_lon - from_lon)
    y       = math.sin(dλ) * math.cos(φ2)
    x       = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    bearing = (math.degrees(math.atan2(y, x)) + 360) % 360

    # Horizontal distance
    a    = math.sin((φ2 - φ1) / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    horiz = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    vert  = to_alt - from_alt

    pitch = math.degrees(math.atan2(vert, horiz)) if horiz > 0 else -90.0
    return bearing, pitch
