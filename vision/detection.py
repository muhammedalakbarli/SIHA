"""Haar Cascade-based face detection module."""

import logging
from typing import TYPE_CHECKING, List

import cv2
import numpy as np

if TYPE_CHECKING:
    from vision.hud import Detection

logger = logging.getLogger(__name__)


class FaceDetector:
    """Detect faces in a frame using OpenCV's Haar Cascade classifier.

    This is a lightweight alternative to YOLOv8 when only face detection
    is needed and no GPU / heavy dependencies are available.

    Args:
        scale_factor: How much the image is reduced at each scale (> 1.0).
        min_neighbors: Minimum neighbouring rectangles to retain a detection.
        color: BGR color for the bounding-box rectangles.
        thickness: Line thickness for the bounding-box rectangles.

    Raises:
        RuntimeError: If the Haar Cascade XML file cannot be loaded.
    """

    def __init__(
        self,
        scale_factor: float = 1.3,
        min_neighbors: int = 5,
        color: tuple = (255, 0, 0),
        thickness: int = 2,
    ) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._classifier = cv2.CascadeClassifier(cascade_path)

        if self._classifier.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from: {cascade_path}"
            )

        if scale_factor <= 1.0:
            raise ValueError(f"scale_factor must be > 1.0, got {scale_factor}")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._color = color
        self._thickness = thickness
        logger.info("FaceDetector initialised (scale=%.2f, min_neighbors=%d)", scale_factor, min_neighbors)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces and draw bounding boxes on *frame*.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            A copy of *frame* with bounding boxes drawn around detected faces.

        Raises:
            ValueError: If *frame* is None or empty.
        """
        if frame is None or frame.size == 0:
            raise ValueError("detect() received an empty or None frame")

        output = frame.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                self._color,
                self._thickness,
            )

        if len(faces):
            logger.debug("Detected %d face(s)", len(faces))

        return output

    def detect_raw(self, frame: np.ndarray) -> "List[Detection]":
        """Detect faces and return raw detection objects (no drawing).

        Args:
            frame: BGR image as a numpy array.

        Returns:
            List of :class:`~vision.hud.Detection` instances.

        Raises:
            ValueError: If *frame* is None or empty.
        """
        from vision.hud import Detection

        if frame is None or frame.size == 0:
            raise ValueError("detect_raw() received an empty or None frame")

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
        )
        detections: List[Detection] = []
        for (x, y, w, h) in faces:
            detections.append(Detection("face", 1.0, x, y, x + w, y + h))
        return detections
