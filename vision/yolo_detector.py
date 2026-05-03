"""YOLOv8-based object detection module."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from vision.hud import Detection

logger = logging.getLogger(__name__)


class YoloDetector:
    """Run YOLOv8 inference on individual frames.

    Args:
        model_path: Path to a YOLOv8 ``.pt`` weights file.
        confidence: Minimum detection confidence (0–1).
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.

    Raises:
        FileNotFoundError: If *model_path* does not exist on disk.
        ImportError: If the ``ultralytics`` package is not installed.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO  # local import keeps startup fast
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required: pip install ultralytics"
            ) from exc

        path = Path(model_path)
        if path.suffix and not path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        if not (0.0 < confidence <= 1.0):
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")

        self._model = YOLO(str(path))
        self._confidence = confidence
        self._device = device
        logger.info(
            "YoloDetector loaded: model=%s, confidence=%.2f, device=%s",
            model_path,
            confidence,
            device,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run inference and return the annotated frame.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Annotated BGR frame with bounding boxes drawn.

        Raises:
            ValueError: If *frame* is None or empty.
        """
        if frame is None or frame.size == 0:
            raise ValueError("detect() received an empty or None frame")

        results = self._model(
            frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        return results[0].plot()

    def detect_raw(self, frame: np.ndarray) -> "List[Detection]":
        """Run inference and return raw detection objects (no drawing).

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

        results = self._model(
            frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        detections: List[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf  = float(box.conf[0])
                label = r.names[int(box.cls[0])]
                detections.append(Detection(label, conf, x1, y1, x2, y2))
        return detections
