"""Camera capture module – USB, RTSP, HTTP, and file sources."""

from __future__ import annotations

import logging
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Wraps OpenCV VideoCapture with context-manager support.

    Accepts USB camera indices, RTSP/HTTP URLs, GStreamer pipelines,
    and video file paths as the ``source`` argument.

    Usage::

        # USB webcam
        with Camera(0) as cam:
            frame = cam.read()

        # RTSP stream (typical drone video link)
        with Camera("rtsp://192.168.1.1:554/live") as cam:
            frame = cam.read()

        # Video file
        with Camera("flight.avi") as cam:
            frame = cam.read()

        # GStreamer pipeline
        with Camera("udpsrc port=5600 ! ... ! appsink") as cam:
            frame = cam.read()
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width:  int = 1280,
        height: int = 720,
        api_preference: int = cv2.CAP_ANY,
    ) -> None:
        """Open the video source.

        Args:
            source:         Integer device index, URL, file path, or
                            GStreamer pipeline string.
            width:          Desired capture width (ignored for RTSP/files).
            height:         Desired capture height (ignored for RTSP/files).
            api_preference: OpenCV backend override (e.g. ``cv2.CAP_FFMPEG``
                            for low-latency RTSP decoding).

        Raises:
            ValueError:  If an integer index < 0 is provided.
            RuntimeError: If the source cannot be opened.
        """
        if isinstance(source, int) and source < 0:
            raise ValueError(f"Camera index must be >= 0, got {source}")

        self._source = source
        self._is_stream = isinstance(source, str) and (
            source.startswith("rtsp://")
            or source.startswith("rtmp://")
            or source.startswith("http://")
            or source.startswith("https://")
        )

        # For RTSP: request FFMPEG backend for lower latency
        if self._is_stream and api_preference == cv2.CAP_ANY:
            api_preference = cv2.CAP_FFMPEG

        self._cap = cv2.VideoCapture(source, api_preference)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        # Set resolution hints (effective for USB cameras; ignored by streams)
        if not self._is_stream:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # For RTSP: minimise internal buffer to reduce latency
        if self._is_stream:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps      = self._cap.get(cv2.CAP_PROP_FPS) or 0
        logger.info(
            "Camera opened: source=%s  %dx%d @ %.1f fps",
            source, actual_w, actual_h, fps,
        )

    # ── Context-manager ───────────────────────────────────────────────────

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 0.0

    @property
    def is_stream(self) -> bool:
        """True if the source is a network stream (RTSP/HTTP)."""
        return self._is_stream

    # ── Public API ────────────────────────────────────────────────────────

    def read(self) -> Optional[np.ndarray]:
        """Capture and return a single frame.

        For network streams, grabs the latest available frame (skips
        buffered frames to minimise latency).

        Returns:
            BGR numpy array, or ``None`` if capture fails.
        """
        if self._is_stream:
            # Discard buffered frames: grab without decoding until empty
            self._cap.grab()

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Failed to read frame from source: %s", self._source)
            return None
        return frame

    def release(self) -> None:
        """Release the underlying capture resource."""
        if self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released: %s", self._source)
