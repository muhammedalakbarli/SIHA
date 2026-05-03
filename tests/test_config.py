"""Tests for utils/config.py."""

import pytest

from utils.config import AppConfig, CameraConfig, DetectorConfig


class TestCameraConfig:
    def test_defaults(self):
        cfg = CameraConfig()
        assert cfg.index == 0
        assert cfg.width == 1280
        assert cfg.height == 720

    def test_custom_values(self):
        cfg = CameraConfig(index=1, width=640, height=480)
        assert cfg.index == 1
        assert cfg.width == 640


class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.model_path == "yolov8n.pt"
        assert cfg.confidence == 0.5
        assert cfg.device == "cpu"

    def test_custom_confidence(self):
        cfg = DetectorConfig(confidence=0.7)
        assert cfg.confidence == 0.7


class TestAppConfig:
    def test_validate_invalid_confidence(self):
        cfg = AppConfig(detector=DetectorConfig(confidence=0.0))
        with pytest.raises(ValueError, match="confidence"):
            cfg.validate()

    def test_validate_confidence_above_one(self):
        cfg = AppConfig(detector=DetectorConfig(confidence=1.5))
        with pytest.raises(ValueError, match="confidence"):
            cfg.validate()

    def test_validate_negative_camera_index(self):
        cfg = AppConfig(camera=CameraConfig(index=-1))
        with pytest.raises(ValueError, match="camera index"):
            cfg.validate()

    def test_validate_missing_model_file(self, tmp_path):
        cfg = AppConfig(detector=DetectorConfig(model_path=str(tmp_path / "missing.pt")))
        with pytest.raises(FileNotFoundError):
            cfg.validate()

    def test_from_dict(self):
        data = {
            "camera": {"index": 2},
            "detector": {"confidence": 0.6},
            "log_level": "DEBUG",
        }
        cfg = AppConfig.from_dict(data)
        assert cfg.camera.index == 2
        assert cfg.detector.confidence == 0.6
        assert cfg.log_level == "DEBUG"

    def test_valid_config_passes(self, tmp_path):
        model = tmp_path / "model.pt"
        model.touch()
        cfg = AppConfig(
            camera=CameraConfig(index=0),
            detector=DetectorConfig(confidence=0.5, model_path=str(model)),
        )
        cfg.validate()  # should not raise
