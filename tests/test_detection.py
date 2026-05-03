"""Tests for vision/detection.py (FaceDetector)."""

import numpy as np
import pytest

from vision.detection import FaceDetector


class TestFaceDetector:
    def test_init_default(self):
        fd = FaceDetector()
        assert fd._scale_factor == 1.3
        assert fd._min_neighbors == 5

    def test_init_invalid_scale_factor(self):
        with pytest.raises(ValueError, match="scale_factor"):
            FaceDetector(scale_factor=0.9)

    def test_detect_returns_same_shape(self):
        fd = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = fd.detect(frame)
        assert result.shape == frame.shape

    def test_detect_does_not_mutate_input(self):
        fd = FaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        fd.detect(frame)
        np.testing.assert_array_equal(frame, original)

    def test_detect_raises_on_none(self):
        fd = FaceDetector()
        with pytest.raises(ValueError, match="empty or None"):
            fd.detect(None)

    def test_detect_raises_on_empty_array(self):
        fd = FaceDetector()
        with pytest.raises(ValueError, match="empty or None"):
            fd.detect(np.array([]))
