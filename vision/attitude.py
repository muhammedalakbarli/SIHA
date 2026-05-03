"""Flight attitude instrument renderers.

Implements the classic glass-cockpit / military-HUD instruments:
  * ADI  (Artificial Direction Indicator / artificial horizon)
  * Heading tape  (horizontal, top-centre)
  * Altitude tape (vertical, right side)
  * Speed tape    (vertical, left side)
  * Vertical-speed indicator (small, beside altitude tape)

All rendering is additive-blended onto the existing frame so the
camera video is always visible through the HUD.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

# ── Colour palette (BGR) – same as hud.py ───────────────────────────────────
_C_PRIMARY = (0, 255, 65)
_C_DIM     = (0, 160, 40)
_C_ALERT   = (30, 30, 220)
_C_YELLOW  = (0, 210, 210)
_FONT      = cv2.FONT_HERSHEY_SIMPLEX


class AttitudeRenderer:
    """Draws all attitude-related HUD instruments onto a BGR frame.

    All methods modify *img* in-place and return nothing.
    """

    # ADI geometry
    _PITCH_SCALE = 2.8    # pixels per degree of pitch
    _ADI_RADIUS  = 90     # clip radius in pixels

    # Tape geometry
    _TAPE_H      = 200    # height of altitude / speed tapes
    _TAPE_W      = 56     # width of altitude / speed tapes

    # ── ADI ──────────────────────────────────────────────────────────────

    def draw_adi(
        self,
        img: np.ndarray,
        roll_deg: float,
        pitch_deg: float,
    ) -> None:
        """Draw the artificial horizon at the frame centre.

        The pitch ladder is drawn on a temporary canvas, rotated by the
        roll angle, clipped to a circle, then additively blended onto *img*.
        The aircraft reference symbol and roll arc are fixed (not rotated).

        Args:
            img:       Target BGR image (modified in-place).
            roll_deg:  Current roll angle in degrees (right = positive).
            pitch_deg: Current pitch angle in degrees (nose-up = positive).
        """
        h, w  = img.shape[:2]
        cx, cy = w // 2, h // 2

        r   = self._ADI_RADIUS
        ps  = self._PITCH_SCALE
        pad = r + 65
        sz  = pad * 2
        mid = pad

        # ── Build pitch ladder on a black canvas ─────────────────────────
        ladder = np.zeros((sz, sz, 3), dtype=np.uint8)

        # Pitch up → horizon moves DOWN on screen
        py_offset = int(pitch_deg * ps)
        hy = mid + py_offset

        # Horizon line (full width)
        cv2.line(ladder, (0, hy), (sz, hy), _C_PRIMARY, 2)

        # Pitch ladder marks (±45°, every 5°)
        for deg in range(-45, 46, 5):
            if deg == 0:
                continue
            y    = hy - int(deg * ps)
            is10 = (deg % 10 == 0)
            hw   = 28 if is10 else 14
            cv2.line(ladder, (mid - hw, y), (mid + hw, y), _C_DIM, 1)
            if is10:
                lbl = str(abs(deg))
                cv2.putText(ladder, lbl,
                            (mid + hw + 3, y + 4), _FONT, 0.30,
                            _C_DIM, 1, cv2.LINE_AA)
                tw = cv2.getTextSize(lbl, _FONT, 0.30, 1)[0][0]
                cv2.putText(ladder, lbl,
                            (mid - hw - tw - 3, y + 4), _FONT, 0.30,
                            _C_DIM, 1, cv2.LINE_AA)

        # ── Rotate canvas by –roll (roll right → horizon tilts right) ────
        M       = cv2.getRotationMatrix2D((float(mid), float(mid)), -roll_deg, 1.0)
        rotated = cv2.warpAffine(ladder, M, (sz, sz))

        # ── Clip to circle ────────────────────────────────────────────────
        mask = np.zeros((sz, sz), dtype=np.uint8)
        cv2.circle(mask, (mid, mid), r, 255, -1)
        rotated = cv2.bitwise_and(rotated, rotated, mask=mask)

        # Outer ring
        cv2.circle(rotated, (mid, mid), r, _C_DIM, 1)

        # ── Roll arc + tick marks (drawn on rotated canvas but fixed arc) ─
        self._draw_roll_arc(rotated, mid, mid, r + 14, roll_deg)

        # ── Fixed aircraft reference symbol (not rotated) ─────────────────
        wing, gap = 26, 7
        cv2.line(rotated, (mid - wing - gap, mid), (mid - gap, mid), _C_PRIMARY, 2)
        cv2.line(rotated, (mid + gap,        mid), (mid + wing + gap, mid), _C_PRIMARY, 2)
        cv2.rectangle(rotated,
                      (mid - 3, mid - 3), (mid + 3, mid + 3),
                      _C_PRIMARY, -1)

        # ── Blit onto main image ──────────────────────────────────────────
        self._blit_additive(img, rotated, cx - mid, cy - mid)

    # ── Heading tape ─────────────────────────────────────────────────────

    def draw_heading_tape(self, img: np.ndarray, heading_deg: int) -> None:
        """Draw a horizontal heading tape at the top-centre of *img*.

        Args:
            img:         Target BGR image (modified in-place).
            heading_deg: Current magnetic heading (0–359).
        """
        h_img, w_img = img.shape[:2]
        tape_w, tape_h = 280, 26
        y0  = 54
        x0  = w_img // 2 - tape_w // 2
        mid = w_img // 2

        cv2.rectangle(img, (x0, y0), (x0 + tape_w, y0 + tape_h), (0, 0, 0), -1)
        cv2.rectangle(img, (x0, y0), (x0 + tape_w, y0 + tape_h), _C_DIM, 1)

        px_per_deg = tape_w / 60.0   # show ±30° range

        for d in range(-40, 41):
            hdg  = (heading_deg + d) % 360
            xpos = x0 + tape_w // 2 + int(d * px_per_deg)
            if xpos < x0 or xpos > x0 + tape_w:
                continue
            if hdg % 10 == 0:
                tick_y = y0 + tape_h - 7
                cv2.line(img, (xpos, tick_y), (xpos, y0 + tape_h - 1), _C_DIM, 1)
                lbl = f"{hdg:03d}"
                tw  = cv2.getTextSize(lbl, _FONT, 0.29, 1)[0][0]
                cv2.putText(img, lbl, (xpos - tw // 2, y0 + 13),
                            _FONT, 0.29, _C_DIM, 1, cv2.LINE_AA)
            elif hdg % 5 == 0:
                cv2.line(img, (xpos, y0 + tape_h - 4),
                              (xpos, y0 + tape_h - 1), _C_DIM, 1)

        # Centre pointer triangle (pointing up into tape)
        pts = np.array([
            [mid,     y0 + tape_h],
            [mid - 5, y0 + tape_h - 9],
            [mid + 5, y0 + tape_h - 9],
        ])
        cv2.fillPoly(img, [pts], _C_PRIMARY)

        # Current heading readout above the tape
        hdg_txt = f"{heading_deg:03d}°"
        tw = cv2.getTextSize(hdg_txt, _FONT, 0.46, 1)[0][0]
        cv2.putText(img, hdg_txt, (mid - tw // 2, y0 - 2),
                    _FONT, 0.46, _C_PRIMARY, 1, cv2.LINE_AA)

    # ── Altitude tape ────────────────────────────────────────────────────

    def draw_altitude_tape(
        self, img: np.ndarray, altitude_m: float, vspeed_ms: float = 0.0
    ) -> None:
        """Draw a vertical altitude tape on the right side of *img*.

        Args:
            img:         Target BGR image.
            altitude_m:  Current altitude in metres (AGL).
            vspeed_ms:   Vertical speed in m/s (positive = climb).
        """
        h_img, w_img = img.shape[:2]
        tw, th = self._TAPE_W, self._TAPE_H
        x0 = w_img - tw - 10
        y0 = h_img // 2 - th // 2

        cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + th), (0, 0, 0), -1)
        cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + th), _C_DIM, 1)

        px_per_m = th / 60.0   # ±30 m visible range

        for dm in range(-40, 41):
            alt  = altitude_m + dm
            ypos = y0 + th // 2 - int(dm * px_per_m)
            if ypos < y0 or ypos > y0 + th:
                continue
            if alt % 10 == 0:
                cv2.line(img, (x0, ypos), (x0 + 9, ypos), _C_DIM, 1)
                lbl = f"{int(alt)}"
                cv2.putText(img, lbl, (x0 + 11, ypos + 4),
                            _FONT, 0.28, _C_DIM, 1, cv2.LINE_AA)
            elif alt % 5 == 0:
                cv2.line(img, (x0, ypos), (x0 + 5, ypos), _C_DIM, 1)

        # Current altitude readout box
        mid_y   = y0 + th // 2
        alt_txt = f"{altitude_m:.1f}"
        bx0, bx1 = x0 - 3, x0 + tw + 3
        by0, by1 = mid_y - 11, mid_y + 11
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), _C_PRIMARY, 1)
        stw = cv2.getTextSize(alt_txt, _FONT, 0.38, 1)[0][0]
        cv2.putText(img, alt_txt, (x0 + (tw - stw) // 2, mid_y + 5),
                    _FONT, 0.38, _C_PRIMARY, 1, cv2.LINE_AA)

        # "ALT m" label
        cv2.putText(img, "ALT", (x0 + 14, y0 - 5),
                    _FONT, 0.30, _C_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "m", (x0 + tw - 14, y0 - 5),
                    _FONT, 0.28, _C_DIM, 1, cv2.LINE_AA)

        # Vertical-speed indicator (small bar to the right of tape)
        self._draw_vspeed(img, x0 + tw + 5, y0, th, vspeed_ms)

    # ── Speed tape ───────────────────────────────────────────────────────

    def draw_speed_tape(self, img: np.ndarray, speed_ms: float) -> None:
        """Draw a vertical ground-speed tape on the left side of *img*.

        Args:
            img:      Target BGR image.
            speed_ms: Current speed in m/s.
        """
        h_img = img.shape[0]
        tw, th = self._TAPE_W, self._TAPE_H
        x0 = 10
        y0 = h_img // 2 - th // 2

        speed_kmh  = speed_ms * 3.6
        px_per_kmh = th / 40.0   # ±20 km/h visible range

        cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + th), (0, 0, 0), -1)
        cv2.rectangle(img, (x0, y0), (x0 + tw, y0 + th), _C_DIM, 1)

        for dk in range(-30, 31):
            spd  = speed_kmh + dk
            ypos = y0 + th // 2 - int(dk * px_per_kmh)
            if ypos < y0 or ypos > y0 + th or spd < 0:
                continue
            if spd % 10 == 0:
                cv2.line(img,
                         (x0 + tw - 9, ypos), (x0 + tw, ypos), _C_DIM, 1)
                lbl = f"{int(spd)}"
                ltw = cv2.getTextSize(lbl, _FONT, 0.28, 1)[0][0]
                cv2.putText(img, lbl,
                            (x0 + tw - 11 - ltw, ypos + 4),
                            _FONT, 0.28, _C_DIM, 1, cv2.LINE_AA)
            elif spd % 5 == 0:
                cv2.line(img,
                         (x0 + tw - 5, ypos), (x0 + tw, ypos), _C_DIM, 1)

        # Current speed readout box
        mid_y   = y0 + th // 2
        spd_txt = f"{speed_kmh:.1f}"
        bx0, bx1 = x0 - 3, x0 + tw + 3
        by0, by1 = mid_y - 11, mid_y + 11
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), _C_PRIMARY, 1)
        stw = cv2.getTextSize(spd_txt, _FONT, 0.38, 1)[0][0]
        cv2.putText(img, spd_txt, (x0 + (tw - stw) // 2, mid_y + 5),
                    _FONT, 0.38, _C_PRIMARY, 1, cv2.LINE_AA)

        cv2.putText(img, "GND", (x0 + 10, y0 - 5),
                    _FONT, 0.30, _C_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, "km/h", (x0 + 4, y0 + th + 14),
                    _FONT, 0.28, _C_DIM, 1, cv2.LINE_AA)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _draw_roll_arc(
        canvas: np.ndarray, cx: int, cy: int, r: int, roll_deg: float
    ) -> None:
        """Draw roll-scale arc and moving pointer on *canvas*."""
        tick_marks = [-60, -45, -30, -20, -10, 0, 10, 20, 30, 45, 60]
        for deg in tick_marks:
            a        = math.radians(deg - 90)
            tlen     = 10 if deg % 30 == 0 else 5
            x0 = int(cx + r * math.cos(a))
            y0 = int(cy + r * math.sin(a))
            x1 = int(cx + (r + tlen) * math.cos(a))
            y1 = int(cy + (r + tlen) * math.sin(a))
            cv2.line(canvas, (x0, y0), (x1, y1), _C_DIM, 1)

        # Roll pointer (fixed at top, moves with roll)
        pa  = math.radians(-roll_deg - 90)
        px0 = int(cx + r * math.cos(pa))
        py0 = int(cy + r * math.sin(pa))
        px1 = int(cx + (r - 10) * math.cos(pa))
        py1 = int(cy + (r - 10) * math.sin(pa))
        cv2.line(canvas, (px0, py0), (px1, py1), _C_PRIMARY, 2)

    @staticmethod
    def _draw_vspeed(
        img: np.ndarray, x0: int, y0: int, tape_h: int, vspeed_ms: float
    ) -> None:
        """Draw a small vertical-speed indicator to the right of the alt tape."""
        vbar_w, vbar_h = 8, tape_h
        mid_y = y0 + tape_h // 2

        # Scale: ±5 m/s = ±full tape height / 2
        scale   = (tape_h / 2) / 5.0
        offset  = int(vspeed_ms * scale)
        offset  = max(-tape_h // 2 + 2, min(tape_h // 2 - 2, offset))

        # Background
        cv2.rectangle(img, (x0, y0), (x0 + vbar_w, y0 + vbar_h), (0, 0, 0), -1)
        cv2.rectangle(img, (x0, y0), (x0 + vbar_w, y0 + vbar_h), _C_DIM, 1)

        # Centre tick
        cv2.line(img, (x0, mid_y), (x0 + vbar_w, mid_y), _C_DIM, 1)

        # Fill bar (positive = up → bar goes above centre)
        if offset != 0:
            bar_y0 = min(mid_y, mid_y - offset)
            bar_y1 = max(mid_y, mid_y - offset)
            col = _C_PRIMARY if vspeed_ms >= 0 else _C_YELLOW
            cv2.rectangle(img,
                          (x0 + 1, bar_y0),
                          (x0 + vbar_w - 1, bar_y1),
                          col, -1)

        # Value label
        vs_lbl = f"{vspeed_ms:+.1f}"
        cv2.putText(img, vs_lbl, (x0 - 2, y0 + vbar_h + 14),
                    _FONT, 0.25, _C_DIM, 1, cv2.LINE_AA)

    @staticmethod
    def _blit_additive(
        dst: np.ndarray, src: np.ndarray, x0: int, y0: int
    ) -> None:
        """Additively blend *src* onto *dst* at offset (x0, y0).

        Only pixels where *src* is non-zero are blended, preserving
        the camera video underneath the HUD graphics.
        """
        sh, sw = src.shape[:2]
        dh, dw = dst.shape[:2]

        dx0 = max(0, x0);  dy0 = max(0, y0)
        dx1 = min(dw, x0 + sw); dy1 = min(dh, y0 + sh)
        sx0 = dx0 - x0;   sy0 = dy0 - y0
        sx1 = sx0 + (dx1 - dx0); sy1 = sy0 + (dy1 - dy0)

        if dx1 <= dx0 or dy1 <= dy0:
            return

        patch  = src[sy0:sy1, sx0:sx1]
        region = dst[dy0:dy1, dx0:dx1]
        mask   = np.any(patch > 0, axis=2)

        blended = np.clip(
            region.astype(np.int16) + (patch.astype(np.int16) * 0.72),
            0, 255,
        ).astype(np.uint8)
        region[mask] = blended[mask]
