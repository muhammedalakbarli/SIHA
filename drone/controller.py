"""Input controller: keyboard (and optional pygame joystick) → RC channel values.

Channel mapping — Mode 2 (standard multi-rotor):
  CH1  Roll      right stick, left (-) / right (+)
  CH2  Pitch     right stick, back (-) / forward (+)
  CH3  Throttle  left stick,  down (-) / up (+)
  CH4  Yaw       left stick,  left (-) / right (+)

Keyboard bindings:
  W / S        – Pitch forward / back
  A / D        – Roll left / right
  ↑ / ↓        – Throttle up / down
  ← / →        – Yaw left / right
  R            – Throttle to mid (hover placeholder)
  T            – Return roll/pitch to neutral instantly
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# RC pulse-width constants (µs)
RC_MID  = 1500
RC_MIN  = 1000
RC_MAX  = 2000
RC_STEP = 15   # µs per control tick
RC_RETURN_STEP = 25   # faster return-to-centre for self-centring axes


class KeyboardController:
    """Map OpenCV waitKey() codes to MAVLink RC channel override values.

    Roll and pitch are self-centring (return to 1500 µs when no key is
    pressed).  Throttle and yaw are *sticky* (hold last value).

    Args:
        step: RC step in µs per key press (default 15).
    """

    def __init__(self, step: int = RC_STEP) -> None:
        self._step = step
        # [CH1-roll, CH2-pitch, CH3-throttle, CH4-yaw, CH5..CH8]
        self._ch: List[int] = [RC_MID] * 8
        self._ch[2] = RC_MIN   # throttle starts at minimum

    # ── Public ───────────────────────────────────────────────────────────

    def process_key(self, key: int) -> bool:
        """Update channels from a waitKey() result.

        Args:
            key: Raw value returned by ``cv2.waitKey()``.

        Returns:
            True if the key was consumed, False if it was not a control key.
        """
        k = key & 0xFF

        # ── Pitch (CH2) ───────────────────────────────────────────────────
        if k == ord("w"):
            self._ch[1] = max(RC_MIN, self._ch[1] - self._step)
        elif k == ord("s"):
            self._ch[1] = min(RC_MAX, self._ch[1] + self._step)

        # ── Roll (CH1) ────────────────────────────────────────────────────
        elif k == ord("a"):
            self._ch[0] = max(RC_MIN, self._ch[0] - self._step)
        elif k == ord("d"):
            self._ch[0] = min(RC_MAX, self._ch[0] + self._step)

        # ── Throttle (CH3) – arrow up / down ─────────────────────────────
        elif key in (0x260000, 2490368) or k == 82:   # up
            self._ch[2] = min(RC_MAX, self._ch[2] + self._step)
        elif key in (0x280000, 2621440) or k == 84:   # down
            self._ch[2] = max(RC_MIN, self._ch[2] - self._step)

        # ── Yaw (CH4) – arrow left / right ───────────────────────────────
        elif key in (0x250000, 2424832) or k == 81:   # left
            self._ch[3] = max(RC_MIN, self._ch[3] - self._step)
        elif key in (0x270000, 2555904) or k == 83:   # right
            self._ch[3] = min(RC_MAX, self._ch[3] + self._step)

        # ── Shortcuts ────────────────────────────────────────────────────
        elif k == ord("r"):
            self._ch[2] = RC_MID   # throttle to 50 %
        elif k == ord("t"):
            self._ch[0] = RC_MID   # instant re-centre roll
            self._ch[1] = RC_MID   # instant re-centre pitch
        else:
            return False

        return True

    def auto_centre(self) -> None:
        """Gradually return roll and pitch to neutral (call once per frame)."""
        for i in (0, 1):
            if self._ch[i] > RC_MID:
                self._ch[i] = max(RC_MID, self._ch[i] - RC_RETURN_STEP)
            elif self._ch[i] < RC_MID:
                self._ch[i] = min(RC_MID, self._ch[i] + RC_RETURN_STEP)

    @property
    def channels(self) -> List[int]:
        """Return current 8-channel RC override list (µs)."""
        return list(self._ch)

    def channel_pct(self, index: int) -> float:
        """Return channel *index* (0-based) as a –100 … +100 % value."""
        val = self._ch[index]
        if val == RC_MID:
            return 0.0
        if val > RC_MID:
            return (val - RC_MID) / (RC_MAX - RC_MID) * 100.0
        return -(RC_MID - val) / (RC_MID - RC_MIN) * 100.0


# ── Optional joystick support (requires pygame) ───────────────────────────────

class JoystickController:
    """Read a USB joystick / gamepad via pygame and map to RC channels.

    Axis mapping (default – adjust to your controller):
        axis 0 – Roll
        axis 1 – Pitch  (inverted)
        axis 3 – Yaw
        axis 2 – Throttle (inverted, -1 = full throttle)

    Args:
        joystick_index: pygame joystick index (0 = first detected).
    """

    def __init__(self, joystick_index: int = 0) -> None:
        self._joy = None
        self._ch: List[int] = [RC_MID] * 8
        self._ch[2] = RC_MIN

        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                logger.warning("No joystick detected")
                return
            self._joy = pygame.joystick.Joystick(joystick_index)
            self._joy.init()
            logger.info("Joystick: %s", self._joy.get_name())
        except ImportError:
            logger.warning("pygame not installed – joystick unavailable")

    def update(self) -> None:
        """Poll joystick state (call once per frame)."""
        if self._joy is None:
            return
        try:
            import pygame
            pygame.event.pump()
            self._ch[0] = self._axis_to_rc(self._joy.get_axis(0))         # roll
            self._ch[1] = self._axis_to_rc(-self._joy.get_axis(1))        # pitch (inv)
            self._ch[3] = self._axis_to_rc(self._joy.get_axis(3))         # yaw
            # Throttle: axis 2 ranges -1 (up) … +1 (down) – inverted
            thr_raw = -self._joy.get_axis(2)                               # -1…+1
            self._ch[2] = int(RC_MIN + (thr_raw + 1) / 2 * (RC_MAX - RC_MIN))
        except Exception:
            logger.debug("Joystick read error", exc_info=True)

    @property
    def channels(self) -> List[int]:
        return list(self._ch)

    @staticmethod
    def _axis_to_rc(value: float) -> int:
        """Convert joystick axis value (-1 … +1) to RC µs (1000 … 2000)."""
        return int(RC_MID + value * (RC_MAX - RC_MID))
