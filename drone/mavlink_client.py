"""MAVLink ground-control client.

Connects to a drone (ArduPilot / PX4) via serial or UDP, reads telemetry
in a background thread, and exposes command helpers (arm, disarm, set_mode,
RC override).

Connection strings (pymavlink format)
--------------------------------------
  Serial     :  /dev/ttyUSB0,57600   or   COM3,57600
  UDP input  :  udpin:0.0.0.0:14550
  UDP output :  udpout:192.168.1.10:14550
  TCP        :  tcp:192.168.1.10:5760
  SITL (sim) :  127.0.0.1:14550
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import List, Optional

from drone.telemetry import Telemetry

logger = logging.getLogger(__name__)

_HEARTBEAT_TIMEOUT = 5.0   # seconds before marking as disconnected


class MAVLinkClient:
    """Thread-based MAVLink GCS client.

    Args:
        connection_string: pymavlink connection URI (see module docstring).
        telemetry: :class:`~drone.telemetry.Telemetry` instance to update.
        baud: Baud rate (only relevant for serial connections).

    Example::

        telem  = Telemetry()
        client = MAVLinkClient("udpin:0.0.0.0:14550", telem)
        client.start()
        # ... main loop ...
        client.stop()
    """

    def __init__(
        self,
        connection_string: str,
        telemetry: Telemetry,
        baud: int = 57600,
    ) -> None:
        self._conn_str   = connection_string
        self.telemetry   = telemetry
        self._baud       = baud
        self._mav        = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock       = threading.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background telemetry thread (non-blocking)."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="mavlink-rx"
        )
        self._thread.start()
        logger.info("MAVLink client started: %s", self._conn_str)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("MAVLink client stopped")

    # ── Commands ─────────────────────────────────────────────────────────

    def arm(self) -> None:
        """Send ARM command."""
        self._send_command(400, param1=1)

    def disarm(self) -> None:
        """Send DISARM command."""
        self._send_command(400, param1=0)

    def set_mode(self, mode: str) -> None:
        """Switch to a named flight mode (e.g. 'LOITER', 'GUIDED')."""
        if self._mav:
            try:
                self._mav.set_mode(mode)
            except Exception:
                logger.exception("Failed to set mode: %s", mode)

    def send_rc_override(self, channels: List[int]) -> None:
        """Send MAVLink RC_CHANNELS_OVERRIDE for manual control.

        Args:
            channels: List of 8 RC channel values (1000–2000 µs).
                      Use 0 to release a channel back to transmitter.
        """
        if self._mav is None:
            return
        ch = list(channels) + [0] * 8
        try:
            with self._lock:
                self._mav.mav.rc_channels_override_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    *ch[:8],
                )
        except Exception:
            logger.debug("RC override send failed", exc_info=True)

    def take_off(self, altitude_m: float = 10.0) -> None:
        """Send TAKEOFF command (GUIDED mode, relative altitude)."""
        self._send_command(22, param7=altitude_m)

    def land(self) -> None:
        """Send LAND command."""
        self._send_command(21)

    def return_to_launch(self) -> None:
        """Send RTL command."""
        self._send_command(20)

    # ── Internal ─────────────────────────────────────────────────────────

    def _send_command(self, command_id: int, **kwargs) -> None:
        if self._mav is None:
            logger.warning("Not connected – cannot send command %d", command_id)
            return
        params = {f"param{i}": kwargs.get(f"param{i}", 0) for i in range(1, 8)}
        try:
            with self._lock:
                self._mav.mav.command_long_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    command_id,
                    0,
                    *(params[f"param{i}"] for i in range(1, 8)),
                )
        except Exception:
            logger.exception("command_long_send failed (cmd=%d)", command_id)

    def _run(self) -> None:
        try:
            from pymavlink import mavutil
        except ImportError:
            logger.error("pymavlink is not installed: pip install pymavlink")
            return

        logger.info("Connecting to %s …", self._conn_str)
        try:
            self._mav = mavutil.mavlink_connection(
                self._conn_str, baud=self._baud
            )
            hb = self._mav.wait_heartbeat(timeout=15)
            if hb is None:
                logger.error("No heartbeat received – check connection")
                return
            self.telemetry.connected = True
            self.telemetry.last_heartbeat = time.time()
            logger.info(
                "Connected to sysid=%d compid=%d",
                self._mav.target_system,
                self._mav.target_component,
            )
            # Request data streams
            self._mav.mav.request_data_stream_send(
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                10,  # 10 Hz
                1,   # start
            )
        except Exception:
            logger.exception("MAVLink connection failed")
            self.telemetry.connected = False
            return

        while not self._stop_event.is_set():
            # Check heartbeat timeout
            if self.telemetry.heartbeat_age > _HEARTBEAT_TIMEOUT:
                if self.telemetry.connected:
                    logger.warning("Heartbeat lost")
                    self.telemetry.connected = False

            try:
                msg = self._mav.recv_match(blocking=True, timeout=0.5)
                if msg is not None:
                    self._dispatch(msg, mavutil)
            except Exception:
                logger.debug("Error reading MAVLink message", exc_info=True)

        self.telemetry.connected = False

    def _dispatch(self, msg, mavutil) -> None:
        """Route a received MAVLink message to the appropriate handler."""
        t = msg.get_type()

        if t == "HEARTBEAT":
            self.telemetry.flight_mode    = self._mav.flightmode
            self.telemetry.armed          = bool(
                msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            self.telemetry.connected      = True
            self.telemetry.last_heartbeat = time.time()

        elif t == "ATTITUDE":
            self.telemetry.roll  = math.degrees(msg.roll)
            self.telemetry.pitch = math.degrees(msg.pitch)
            self.telemetry.yaw   = math.degrees(msg.yaw) % 360

        elif t == "GLOBAL_POSITION_INT":
            self.telemetry.lat          = msg.lat / 1e7
            self.telemetry.lon          = msg.lon / 1e7
            self.telemetry.altitude_abs = msg.alt / 1000.0
            self.telemetry.altitude_rel = msg.relative_alt / 1000.0

        elif t == "VFR_HUD":
            self.telemetry.airspeed      = msg.airspeed
            self.telemetry.groundspeed   = msg.groundspeed
            self.telemetry.heading       = msg.heading
            self.telemetry.throttle      = msg.throttle
            self.telemetry.vertical_speed = msg.climb

        elif t == "BATTERY_STATUS":
            if msg.battery_remaining >= 0:
                self.telemetry.battery_remaining = msg.battery_remaining
            if msg.voltages and msg.voltages[0] != 65535:
                self.telemetry.battery_voltage = msg.voltages[0] / 1000.0

        elif t == "GPS_RAW_INT":
            self.telemetry.gps_fix        = msg.fix_type
            self.telemetry.gps_satellites = msg.satellites_visible

        elif t == "RC_CHANNELS":
            self.telemetry.rssi = msg.rssi
            self.telemetry.rc_channels = [
                msg.chan1_raw, msg.chan2_raw, msg.chan3_raw, msg.chan4_raw,
                msg.chan5_raw, msg.chan6_raw, msg.chan7_raw, msg.chan8_raw,
            ]
