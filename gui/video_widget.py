"""Video display widget – renders OpenCV BGR frames in a QLabel."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class VideoWidget(QWidget):
    """Displays a live BGR camera feed scaled to fit the widget.

    Call :meth:`update_frame` with a numpy BGR array to refresh the display.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._label.setStyleSheet("background: #000;")
        self._label.setText("No signal")
        self._label.setStyleSheet(
            "background: #000; color: #00ff41; font-family: monospace; font-size: 14px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    @pyqtSlot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        """Render *frame* (BGR numpy array) in the widget."""
        qt_img  = _bgr_to_qimage(frame)
        pixmap  = QPixmap.fromImage(qt_img)
        scaled  = pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a BGR numpy array to a QImage (RGB888)."""
    h, w, ch = frame.shape
    rgb = frame[..., ::-1].copy()   # BGR → RGB, ensure contiguous
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
