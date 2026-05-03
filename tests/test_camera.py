"""Tests for vision/camera.py (Camera)."""

import pytest
from unittest.mock import MagicMock, patch

from vision.camera import Camera


class TestCamera:
    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="Camera index"):
            Camera(source=-1)

    @patch("vision.camera.cv2.VideoCapture")
    def test_failed_open_raises(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_cls.return_value = mock_cap

        with pytest.raises(RuntimeError, match="Failed to open"):
            Camera(source=0)

    @patch("vision.camera.cv2.VideoCapture")
    def test_read_returns_frame(self, mock_cap_cls):
        import numpy as np

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_cap_cls.return_value = mock_cap

        cam = Camera(source=0)
        frame = cam.read()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    @patch("vision.camera.cv2.VideoCapture")
    def test_read_returns_none_on_failure(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap_cls.return_value = mock_cap

        cam = Camera(source=0)
        assert cam.read() is None

    @patch("vision.camera.cv2.VideoCapture")
    def test_context_manager_releases(self, mock_cap_cls):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_cls.return_value = mock_cap

        with Camera(source=0):
            pass

        mock_cap.release.assert_called_once()
