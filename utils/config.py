"""Application configuration management."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CameraConfig:
    """Camera capture settings."""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


@dataclass
class DetectorConfig:
    """Object detection settings."""

    model_path: str = "yolov8n.pt"
    confidence: float = 0.5
    device: str = "cpu"


@dataclass
class FaceDetectorConfig:
    """Face detection settings."""

    scale_factor: float = 1.3
    min_neighbors: int = 5
    color: tuple = (255, 0, 0)
    thickness: int = 2


@dataclass
class AppConfig:
    """Top-level application configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    face_detector: FaceDetectorConfig = field(default_factory=FaceDetectorConfig)
    window_title: str = "SIHA - AI Vision System"
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Create AppConfig from a plain dictionary."""
        camera = CameraConfig(**data.get("camera", {}))
        detector = DetectorConfig(**data.get("detector", {}))
        face_detector = FaceDetectorConfig(**data.get("face_detector", {}))
        return cls(
            camera=camera,
            detector=detector,
            face_detector=face_detector,
            window_title=data.get("window_title", "SIHA - AI Vision System"),
            log_level=data.get("log_level", "INFO"),
        )

    def validate(self) -> None:
        """Raise ValueError if any config value is invalid."""
        if not (0.0 < self.detector.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in (0, 1], got {self.detector.confidence}"
            )
        if self.camera.index < 0:
            raise ValueError(
                f"camera index must be >= 0, got {self.camera.index}"
            )
        model = Path(self.detector.model_path)
        if model.suffix and not model.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.detector.model_path}"
            )
