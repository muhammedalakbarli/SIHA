"""UAV / drone-style HUD (Heads-Up Display) renderer.

Applies the visual look of a real PUA (pilotsuz uçuş aparatı) feed:
  - Green phosphor tint + scanlines + vignette
  - Corner-bracket frame border
  - Centre crosshair / reticle
  - Target acquisition boxes (corner-bracket style, not solid rectangles)
  - Top bar  – callsign, timestamp, GPS
  - Bottom bar – FPS, altitude, speed, heading, battery
  - Blinking REC indicator
  - Mini compass rose (top-right)

``render()``     – standard detection HUD (simulated or injected telemetry)
``render_fpv()`` – full FPV GCS HUD with ADI + tapes (real Telemetry object)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from drone.telemetry import Telemetry

# ── Colour palette (BGR) ────────────────────────────────────────────────────
_C_PRIMARY = (0, 255, 65)      # bright phosphor green
_C_DIM     = (0, 160, 40)      # secondary / dim green
_C_ALERT   = (30, 30, 220)     # alert red   (REC, low battery)
_C_YELLOW  = (0, 210, 210)     # caution yellow
_FONT      = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Detection:
    """A single detected object returned by a detector."""

    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class HUDRenderer:
    """Render a military-drone-style HUD on top of every camera frame.

    Args:
        callsign:  Callsign shown in the top-left corner.
        lat / lon: Simulated GPS position (used when no Telemetry is supplied).
        altitude:  Simulated altitude in metres.
        speed:     Simulated ground speed in km/h.
        heading:   Simulated magnetic heading (0–359).
        battery:   Simulated battery level (0–100 %).
        recording: Whether the REC blink indicator is active.

    Usage (detection HUD)::

        hud = HUDRenderer(callsign="SIHA-01")
        out = hud.render(frame, detections=dets, mode="YOLO")

    Usage (FPV HUD with real MAVLink telemetry)::

        hud = HUDRenderer(callsign="SIHA-01")
        out = hud.render_fpv(frame, telemetry=telem, detections=dets)
    """

    def __init__(
        self,
        callsign: str = "SIHA",
        lat: float = 40.4093,
        lon: float = 49.8671,
        altitude: float = 120.5,
        speed: float = 35.2,
        heading: int = 45,
        battery: int = 87,
        recording: bool = True,
    ) -> None:
        self.callsign  = callsign
        self.lat       = lat
        self.lon       = lon
        self.altitude  = altitude
        self.speed     = speed
        self.heading   = heading
        self.battery   = battery
        self.recording = recording

        self._frame_count  = 0
        self._fps          = 0.0
        self._fps_timer    = time.time()
        self._vignette_cache: Optional[np.ndarray] = None
        self._vignette_shape: Optional[tuple] = None

    # ── Public render methods ────────────────────────────────────────────────

    def render(
        self,
        frame: np.ndarray,
        detections: Optional[List[Detection]] = None,
        mode: str = "YOLO",
        telemetry: Optional["Telemetry"] = None,
    ) -> np.ndarray:
        """Return a new frame with the detection HUD composited on top.

        Args:
            frame:      Raw BGR camera frame.
            detections: Detected objects to draw as target boxes.
            mode:       Detector mode label (e.g. ``"YOLO"``).
            telemetry:  Optional live :class:`~drone.telemetry.Telemetry`.
                        When supplied, real values are used in place of the
                        simulated defaults set in ``__init__``.

        Returns:
            New BGR frame – the input is never modified.
        """
        h, w = frame.shape[:2]

        out = self._apply_green_tint(frame)
        out = self._apply_scanlines(out)
        out = self._apply_vignette(out, h, w)

        self._draw_frame_brackets(out, h, w)
        self._draw_crosshair(out, h, w)

        if detections:
            for det in detections:
                self._draw_target(out, det)

        self._draw_top_bar(out, h, w, mode, telemetry)
        self._draw_bottom_bar(out, h, w, telemetry)
        self._draw_compass(out, h, w, telemetry)
        self._update_fps()

        return out

    def render_fpv(
        self,
        frame: np.ndarray,
        telemetry: "Telemetry",
        detections: Optional[List[Detection]] = None,
    ) -> np.ndarray:
        """Return a full FPV GCS frame: HUD + ADI + tapes + telemetry.

        This method adds the full glass-cockpit attitude instruments
        (artificial horizon, heading tape, altitude tape, speed tape)
        on top of the base HUD.

        Args:
            frame:      Raw BGR camera frame.
            telemetry:  Live :class:`~drone.telemetry.Telemetry` object.
            detections: Optional detected objects to draw as target boxes.

        Returns:
            New annotated BGR frame.
        """
        from vision.attitude import AttitudeRenderer

        h, w = frame.shape[:2]

        # ── Base frame effects ────────────────────────────────────────────
        out = self._apply_green_tint(frame)
        out = self._apply_scanlines(out)
        out = self._apply_vignette(out, h, w)

        # ── Attitude instruments ──────────────────────────────────────────
        adi = AttitudeRenderer()
        adi.draw_adi(out, roll_deg=telemetry.roll, pitch_deg=telemetry.pitch)
        adi.draw_heading_tape(out, heading_deg=telemetry.heading)
        adi.draw_altitude_tape(
            out,
            altitude_m=telemetry.altitude_rel,
            vspeed_ms=telemetry.vertical_speed,
        )
        adi.draw_speed_tape(out, speed_ms=telemetry.groundspeed)

        # ── Detection targets ─────────────────────────────────────────────
        if detections:
            for det in detections:
                self._draw_target(out, det)

        # ── Frame border + crosshair (drawn after instruments) ────────────
        self._draw_frame_brackets(out, h, w)
        self._draw_crosshair(out, h, w)

        # ── Info bars ─────────────────────────────────────────────────────
        self._draw_fpv_top_bar(out, h, w, telemetry)
        self._draw_fpv_bottom_bar(out, h, w, telemetry)
        self._draw_compass(out, h, w, telemetry)
        self._draw_status_panel(out, h, w, telemetry)

        self._update_fps()
        return out

    # ── Frame-level effects ──────────────────────────────────────────────────

    @staticmethod
    def _apply_green_tint(frame: np.ndarray) -> np.ndarray:
        out = frame.astype(np.float32)
        out[:, :, 0] = np.clip(out[:, :, 0] * 0.55, 0, 255)
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.20, 0, 255)
        out[:, :, 2] = np.clip(out[:, :, 2] * 0.55, 0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def _apply_scanlines(frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        out[::2] = (out[::2] * 0.70).astype(np.uint8)
        return out

    def _apply_vignette(self, frame: np.ndarray, h: int, w: int) -> np.ndarray:
        if self._vignette_shape != (h, w):
            cx, cy = w / 2, h / 2
            max_r  = math.sqrt(cx ** 2 + cy ** 2)
            Y, X   = np.ogrid[:h, :w]
            r      = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            mask   = 1.0 - np.clip(r / (max_r * 0.80), 0.0, 1.0) ** 2 * 0.60
            self._vignette_cache = mask.astype(np.float32)
            self._vignette_shape = (h, w)
        return (frame * self._vignette_cache[:, :, np.newaxis]).astype(np.uint8)

    # ── Structural elements ──────────────────────────────────────────────────

    @staticmethod
    def _draw_frame_brackets(img: np.ndarray, h: int, w: int) -> None:
        length = min(w, h) // 9
        c = _C_PRIMARY
        for ox, oy, sx, sy in [
            (0,     0,      1,  1),
            (w - 1, 0,     -1,  1),
            (0,     h - 1,  1, -1),
            (w - 1, h - 1, -1, -1),
        ]:
            cv2.line(img, (ox, oy), (ox + sx * length, oy), c, 2)
            cv2.line(img, (ox, oy), (ox, oy + sy * length), c, 2)

    @staticmethod
    def _draw_crosshair(img: np.ndarray, h: int, w: int) -> None:
        cx, cy = w // 2, h // 2
        arm, gap = 22, 9
        cv2.line(img, (cx - arm - gap, cy), (cx - gap, cy),       _C_PRIMARY, 1)
        cv2.line(img, (cx + gap, cy),       (cx + arm + gap, cy), _C_PRIMARY, 1)
        cv2.line(img, (cx, cy - arm - gap), (cx, cy - gap),       _C_PRIMARY, 1)
        cv2.line(img, (cx, cy + gap),       (cx, cy + arm + gap), _C_PRIMARY, 1)
        cv2.circle(img, (cx, cy), 2,            _C_PRIMARY, -1)
        cv2.circle(img, (cx, cy), arm + gap + 5, _C_DIM,    1)

    # ── Target box ───────────────────────────────────────────────────────────

    @staticmethod
    def _draw_target(img: np.ndarray, det: Detection) -> None:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        bw = x2 - x1
        bh = y2 - y1
        blen = max(8, min(bw, bh) // 4)
        c = _C_PRIMARY if det.confidence >= 0.60 else _C_DIM

        for px, py, dx, dy in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(img, (px, py), (px + dx * blen, py), c, 2)
            cv2.line(img, (px, py), (px, py + dy * blen), c, 2)

        label  = f"{det.label.upper()}  {det.confidence:.0%}"
        tw, th = cv2.getTextSize(label, _FONT, 0.42, 1)[0]
        lx = max(x1, 0)
        ly = max(y1 - 6, th + 2)
        cv2.putText(img, label, (lx, ly), _FONT, 0.42, c, 1, cv2.LINE_AA)

        dist_txt = f"{int(bh * 0.45)}m"
        dx2 = x1 + bw // 2 - cv2.getTextSize(dist_txt, _FONT, 0.35, 1)[0][0] // 2
        cv2.putText(img, dist_txt, (dx2, y2 + 12), _FONT, 0.35, _C_DIM, 1, cv2.LINE_AA)

    # ── Detection-mode HUD bars ──────────────────────────────────────────────

    def _draw_top_bar(
        self,
        img: np.ndarray,
        h: int,
        w: int,
        mode: str,
        telem: Optional["Telemetry"],
    ) -> None:
        lat     = telem.lat      if telem else self.lat
        lon     = telem.lon      if telem else self.lon
        heading = telem.heading  if telem else self.heading
        rec     = self.recording

        cv2.line(img, (0, 50), (w, 50), _C_DIM, 1)

        cv2.putText(img, self.callsign,    (12, 22), _FONT, 0.65, _C_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, f"MODE:{mode}",   (12, 40), _FONT, 0.38, _C_DIM,    1, cv2.LINE_AA)

        ts  = time.strftime("%Y-%m-%d  %H:%M:%S UTC")
        tw  = cv2.getTextSize(ts, _FONT, 0.40, 1)[0][0]
        cv2.putText(img, ts, (w // 2 - tw // 2, 18), _FONT, 0.40, _C_DIM, 1, cv2.LINE_AA)

        gps = f"LAT {lat:.4f}   LON {lon:.4f}"
        tw2 = cv2.getTextSize(gps, _FONT, 0.38, 1)[0][0]
        cv2.putText(img, gps, (w // 2 - tw2 // 2, 35), _FONT, 0.38, _C_DIM, 1, cv2.LINE_AA)

        if rec and int(time.time()) % 2 == 0:
            rx = w - 58
            cv2.circle(img, (rx, 15), 5, _C_ALERT, -1)
            cv2.putText(img, "REC", (rx + 9, 19), _FONT, 0.40, _C_ALERT, 1, cv2.LINE_AA)

    def _draw_bottom_bar(
        self,
        img: np.ndarray,
        h: int,
        w: int,
        telem: Optional["Telemetry"],
    ) -> None:
        alt  = telem.altitude_rel       if telem else self.altitude
        spd  = telem.groundspeed_kmh    if telem else self.speed
        hdg  = telem.heading            if telem else self.heading
        bat  = telem.battery_remaining  if telem else self.battery
        y    = h - 10

        cv2.line(img, (0, h - 32), (w, h - 32), _C_DIM, 1)

        cv2.putText(img, f"FPS {self._fps:4.1f}",   (12,  y), _FONT, 0.40, _C_DIM,    1, cv2.LINE_AA)
        cv2.putText(img, f"ALT {alt:.1f}m",          (90,  y), _FONT, 0.40, _C_PRIMARY, 1, cv2.LINE_AA)
        cv2.putText(img, f"SPD {spd:.1f}km/h",       (195, y), _FONT, 0.40, _C_DIM,    1, cv2.LINE_AA)
        cv2.putText(img, f"HDG {hdg:03d}",           (320, y), _FONT, 0.40, _C_DIM,    1, cv2.LINE_AA)

        bat_col = _C_ALERT if bat < 20 else (_C_YELLOW if bat < 40 else _C_PRIMARY)
        bat_txt = f"BAT {bat}%"
        tw = cv2.getTextSize(bat_txt, _FONT, 0.40, 1)[0][0]
        cv2.putText(img, bat_txt, (w - tw - 12, y), _FONT, 0.40, bat_col, 1, cv2.LINE_AA)

    # ── FPV-mode HUD bars ────────────────────────────────────────────────────

    def _draw_fpv_top_bar(
        self, img: np.ndarray, h: int, w: int, telem: "Telemetry"
    ) -> None:
        """Top bar with callsign, timestamp, GPS, ARM state, REC."""
        arm_col = _C_ALERT if not telem.armed else _C_PRIMARY
        arm_txt = "ARMED" if telem.armed else "DISARMED"
        con_col = _C_ALERT if not telem.connected else _C_DIM

        cv2.line(img, (0, 52), (w, 52), _C_DIM, 1)

        # Left: callsign + ARM
        cv2.putText(img, self.callsign, (12, 20), _FONT, 0.65, _C_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(img, arm_txt,       (12, 38), _FONT, 0.40, arm_col,    1, cv2.LINE_AA)

        # Centre: timestamp + GPS
        ts  = time.strftime("%Y-%m-%d  %H:%M:%S UTC")
        tw  = cv2.getTextSize(ts, _FONT, 0.40, 1)[0][0]
        cv2.putText(img, ts, (w // 2 - tw // 2, 16), _FONT, 0.40, _C_DIM, 1, cv2.LINE_AA)

        gps = f"LAT {telem.lat:.5f}   LON {telem.lon:.5f}"
        tw2 = cv2.getTextSize(gps, _FONT, 0.38, 1)[0][0]
        cv2.putText(img, gps, (w // 2 - tw2 // 2, 31), _FONT, 0.38, _C_DIM, 1, cv2.LINE_AA)

        # Connection dot
        cv2.putText(img, f"MODE:{telem.flight_mode}", (w // 2 - 45, 46),
                    _FONT, 0.35, con_col, 1, cv2.LINE_AA)

        # Right: GPS fix + REC
        gps_lbl = telem.gps_fix_label
        gps_col = _C_PRIMARY if telem.gps_fix >= 3 else _C_ALERT
        cv2.putText(img, f"{gps_lbl} {telem.gps_satellites}sat",
                    (w - 130, 20), _FONT, 0.38, gps_col, 1, cv2.LINE_AA)

        if self.recording and int(time.time()) % 2 == 0:
            rx = w - 58
            cv2.circle(img, (rx, 36), 5, _C_ALERT, -1)
            cv2.putText(img, "REC", (rx + 9, 40), _FONT, 0.40, _C_ALERT, 1, cv2.LINE_AA)

    def _draw_fpv_bottom_bar(
        self, img: np.ndarray, h: int, w: int, telem: "Telemetry"
    ) -> None:
        """Bottom bar with FPS, alt, speed, heading, throttle, battery."""
        y = h - 10
        cv2.line(img, (0, h - 32), (w, h - 32), _C_DIM, 1)

        cv2.putText(img, f"FPS {self._fps:4.1f}",               (12,  y), _FONT, 0.38, _C_DIM,    1, cv2.LINE_AA)
        cv2.putText(img, f"ALT {telem.altitude_rel:.1f}m",      (90,  y), _FONT, 0.38, _C_PRIMARY, 1, cv2.LINE_AA)
        cv2.putText(img, f"SPD {telem.groundspeed_kmh:.1f}km/h",(195, y), _FONT, 0.38, _C_DIM,    1, cv2.LINE_AA)
        cv2.putText(img, f"HDG {telem.heading:03d}",            (320, y), _FONT, 0.38, _C_DIM,    1, cv2.LINE_AA)
        cv2.putText(img, f"THR {telem.throttle}%",              (405, y), _FONT, 0.38, _C_DIM,    1, cv2.LINE_AA)

        # RSSI
        rssi_col = _C_ALERT if telem.rssi < 60 else _C_DIM
        cv2.putText(img, f"RSSI {telem.rssi}", (490, y), _FONT, 0.38, rssi_col, 1, cv2.LINE_AA)

        # Battery
        bat = telem.battery_remaining
        bat_col = _C_ALERT if bat < 20 else (_C_YELLOW if bat < 40 else _C_PRIMARY)
        bat_txt = f"BAT {bat}%"
        if telem.battery_voltage > 0:
            bat_txt += f" {telem.battery_voltage:.1f}V"
        tw = cv2.getTextSize(bat_txt, _FONT, 0.38, 1)[0][0]
        cv2.putText(img, bat_txt, (w - tw - 12, y), _FONT, 0.38, bat_col, 1, cv2.LINE_AA)

    def _draw_status_panel(
        self, img: np.ndarray, h: int, w: int, telem: "Telemetry"
    ) -> None:
        """Small status panel on the left, below the speed tape."""
        x, y = 10, h // 2 + self._TAPE_H_HALF + 20
        lines = [
            (f"V/S  {telem.vertical_speed:+.1f}m/s", _C_DIM),
            (f"ROLL {telem.roll:+.1f}°",              _C_DIM),
            (f"PTCH {telem.pitch:+.1f}°",             _C_DIM),
        ]
        for i, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (x, y + i * 16), _FONT, 0.32, col, 1, cv2.LINE_AA)

    _TAPE_H_HALF = 100   # half of AttitudeRenderer._TAPE_H

    # ── Compass ───────────────────────────────────────────────────────────────

    def _draw_compass(
        self,
        img: np.ndarray,
        h: int,
        w: int,
        telem: Optional["Telemetry"],
    ) -> None:
        heading = telem.heading if telem else self.heading
        cx, cy, r = w - 48, 85, 24

        cv2.circle(img, (cx, cy), r, _C_DIM, 1)
        cv2.circle(img, (cx, cy), 2, _C_DIM, -1)

        ang   = math.radians(heading - 90)
        nx    = int(cx + r * 0.78 * math.cos(ang))
        ny    = int(cy + r * 0.78 * math.sin(ang))
        cv2.line(img, (cx, cy), (nx, ny), _C_PRIMARY, 2)
        ox = int(cx - r * 0.45 * math.cos(ang))
        oy = int(cy - r * 0.45 * math.sin(ang))
        cv2.line(img, (cx, cy), (ox, oy), _C_DIM, 1)

        for label, deg in [("N", -90), ("E", 0), ("S", 90), ("W", 180)]:
            ra  = math.radians(deg)
            lx  = int(cx + (r + 9) * math.cos(ra)) - 3
            ly  = int(cy + (r + 9) * math.sin(ra)) + 4
            col = _C_PRIMARY if label == "N" else _C_DIM
            cv2.putText(img, label, (lx, ly), _FONT, 0.28, col, 1, cv2.LINE_AA)

    # ── FPS counter ───────────────────────────────────────────────────────────

    def _update_fps(self) -> None:
        self._frame_count += 1
        now     = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps         = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer   = now
