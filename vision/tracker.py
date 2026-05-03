"""Single-target visual tracker using OpenCV tracking algorithms.

Wraps OpenCV's built-in trackers (CSRT, KCF, MIL) with a clean API.
The tracker is initialised by providing a bounding box, then updated
frame-by-frame.  Lost targets are handled gracefully.

Example::

    tracker = TargetTracker(algorithm="CSRT")
    # User clicks on a detection:
    tracker.init(frame, bbox=(x, y, w, h))
    # In the main loop:
    bbox = tracker.update(frame)
    if bbox:
        tracker.draw(frame)
    else:
        print("Target lost")
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colour for the tracker overlay
_C_LOCKED  = (0, 255, 220)   # cyan
_C_SEARCH  = (0, 180, 255)   # orange
_FONT      = cv2.FONT_HERSHEY_SIMPLEX

BBox = Tuple[int, int, int, int]   # x, y, w, h


class TargetTracker:
    """Frame-to-frame visual target tracker.

    Args:
        algorithm:    One of ``"CSRT"`` (accurate, slower),
                      ``"KCF"`` (fast, less accurate),
                      ``"MIL"`` (robust to occlusion).
        max_lost:     Number of consecutive failed updates before the
                      tracker is automatically stopped.
    """

    _CREATORS = {
        "CSRT": lambda: cv2.TrackerCSRT_create(),
        "KCF":  lambda: cv2.TrackerKCF_create(),
        "MIL":  lambda: cv2.TrackerMIL_create(),
    }

    def __init__(
        self,
        algorithm: str = "CSRT",
        max_lost: int = 15,
    ) -> None:
        if algorithm not in self._CREATORS:
            raise ValueError(
                f"Unknown tracker algorithm '{algorithm}'. "
                f"Choose from: {list(self._CREATORS)}"
            )
        self._algorithm   = algorithm
        self._max_lost    = max_lost
        self._tracker     = None
        self._bbox: Optional[BBox] = None
        self._active      = False
        self._lost_streak = 0
        self._label       = ""

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """True while the tracker has a live target."""
        return self._active

    @property
    def bbox(self) -> Optional[BBox]:
        """Last known bounding box (x, y, w, h) or None."""
        return self._bbox

    @property
    def centre(self) -> Optional[Tuple[int, int]]:
        """Centre pixel of the tracked box, or None."""
        if self._bbox is None:
            return None
        x, y, w, h = self._bbox
        return x + w // 2, y + h // 2

    # ── Public API ────────────────────────────────────────────────────────

    def init(
        self,
        frame: np.ndarray,
        bbox: BBox,
        label: str = "TARGET",
    ) -> None:
        """Initialise (or re-initialise) the tracker on a bounding box.

        Args:
            frame: BGR frame in which the target currently appears.
            bbox:  Bounding box as (x, y, width, height) in pixels.
            label: Optional label shown in the HUD overlay.
        """
        creator = self._CREATORS[self._algorithm]
        self._tracker     = creator()
        self._tracker.init(frame, bbox)
        self._bbox        = tuple(map(int, bbox))   # type: ignore[assignment]
        self._label       = label
        self._active      = True
        self._lost_streak = 0
        logger.info("Tracker initialised: algorithm=%s bbox=%s", self._algorithm, bbox)

    def init_from_detection(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        label: str = "TARGET",
    ) -> None:
        """Convenience wrapper – accepts x1/y1/x2/y2 detection coordinates."""
        self.init(frame, (x1, y1, x2 - x1, y2 - y1), label=label)

    def update(self, frame: np.ndarray) -> Optional[BBox]:
        """Update tracker with the next frame.

        Args:
            frame: New BGR camera frame.

        Returns:
            Updated (x, y, w, h) bounding box, or ``None`` if the target
            was lost.
        """
        if not self._active or self._tracker is None:
            return None

        ok, raw_bbox = self._tracker.update(frame)

        if ok:
            self._bbox        = tuple(map(int, raw_bbox))  # type: ignore[assignment]
            self._lost_streak = 0
            return self._bbox
        else:
            self._lost_streak += 1
            logger.debug("Tracker: lost frame %d/%d", self._lost_streak, self._max_lost)
            if self._lost_streak >= self._max_lost:
                logger.info("Tracker: target lost after %d frames", self._max_lost)
                self.stop()
            return None

    def stop(self) -> None:
        """Release the tracker and clear state."""
        self._active      = False
        self._tracker     = None
        self._bbox        = None
        self._lost_streak = 0

    def draw(self, img: np.ndarray) -> None:
        """Draw the tracker overlay on *img* (in-place).

        Shows a cyan acquisition box when locked, with an animated
        corner-bracket style and a "LOCKED" label.
        """
        if not self._active or self._bbox is None:
            return

        x, y, w, h = self._bbox
        x2, y2 = x + w, y + h
        blen = max(8, min(w, h) // 5)
        c = _C_LOCKED

        # Corner brackets
        for px, py, dx, dy in [
            (x,  y,   1,  1), (x2, y,  -1,  1),
            (x,  y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(img, (px, py), (px + dx * blen, py), c, 2)
            cv2.line(img, (px, py), (px, py + dy * blen), c, 2)

        # Centre crosshair
        cx = x + w // 2
        cy_box = y + h // 2
        cv2.line(img, (cx - 6, cy_box), (cx + 6, cy_box), c, 1)
        cv2.line(img, (cx, cy_box - 6), (cx, cy_box + 6), c, 1)

        # Label
        txt = f"LOCKED  {self._label}"
        cv2.putText(img, txt, (x, y - 8), _FONT, 0.40, c, 1, cv2.LINE_AA)
