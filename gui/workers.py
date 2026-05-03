"""Background worker threads for camera capture and telemetry processing.

``CameraWorker`` runs in a dedicated QThread and emits processed frames
(with HUD + optional detection overlay) ready for display.

``TelemetryPoller`` is a lightweight QTimer-driven poller that reads the
shared Telemetry object and emits update signals at a fixed rate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

if TYPE_CHECKING:
    from drone.telemetry import Telemetry
    from vision.camera import Camera
    from vision.hud import Detection, HUDRenderer
    from vision.tracker import TargetTracker

logger = logging.getLogger(__name__)

# Run object/face detection only every N frames to keep video smooth on CPU
_DETECT_EVERY_N = 3


class CameraWorker(QObject):
    """Captures frames, runs detection, applies HUD, emits processed frames.

    Signals:
        frame_ready:   Emits a processed BGR numpy array for every frame.
        detections_ready: Emits the latest list of Detection objects.
        error:         Emits an error message string.
    """

    frame_ready      = pyqtSignal(np.ndarray)
    detections_ready = pyqtSignal(list)
    error            = pyqtSignal(str)

    def __init__(
        self,
        source,
        telemetry: "Telemetry",
        hud: "HUDRenderer",
        detector=None,
        tracker: Optional["TargetTracker"] = None,
    ) -> None:
        super().__init__()
        self._source    = source
        self._telem     = telemetry
        self._hud       = hud
        self._detector  = detector
        self._tracker   = tracker
        self._running   = False
        self._frame_idx = 0
        self._last_dets: List["Detection"] = []

    @pyqtSlot()
    def run(self) -> None:
        """Main capture loop – runs until stop() is called."""
        from vision.camera import Camera

        self._running = True
        try:
            with Camera(source=self._source) as cam:
                while self._running:
                    frame = cam.read()
                    if frame is None:
                        continue

                    self._frame_idx += 1

                    # ── Detection (throttled) ─────────────────────────────
                    if self._detector and self._frame_idx % _DETECT_EVERY_N == 0:
                        try:
                            self._last_dets = self._detector.detect_raw(frame)
                            self.detections_ready.emit(list(self._last_dets))
                        except Exception:
                            logger.debug("Detection error", exc_info=True)

                    # ── Tracker ───────────────────────────────────────────
                    if self._tracker and self._tracker.is_active:
                        self._tracker.update(frame)

                    # ── HUD render ────────────────────────────────────────
                    try:
                        output = self._hud.render_fpv(
                            frame,
                            telemetry=self._telem,
                            detections=self._last_dets,
                        )
                    except Exception:
                        logger.debug("HUD render error", exc_info=True)
                        output = frame

                    # ── Tracker overlay (on top of HUD) ───────────────────
                    if self._tracker and self._tracker.is_active:
                        self._tracker.draw(output)

                    self.frame_ready.emit(output)

        except RuntimeError as exc:
            logger.error("Camera error: %s", exc)
            self.error.emit(str(exc))

    def stop(self) -> None:
        self._running = False

    def set_detector(self, detector) -> None:
        self._detector = detector

    def set_tracker(self, tracker: "TargetTracker") -> None:
        self._tracker = tracker


def make_camera_thread(worker: CameraWorker) -> QThread:
    """Create and wire a QThread for *worker*."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    return thread
