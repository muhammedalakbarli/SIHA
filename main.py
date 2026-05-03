"""SIHA - AI Vision System entry point."""

import argparse
import logging
import sys

import cv2

from utils.config import AppConfig, CameraConfig, DetectorConfig
from vision.camera import Camera
from vision.detection import FaceDetector
from vision.hud import HUDRenderer
from vision.yolo_detector import YoloDetector


def setup_logging(level: str) -> None:
    """Configure root logger with a timestamped console handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="siha",
        description="SIHA – Real-time AI Vision System",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="INDEX",
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        metavar="PATH",
        help="Path to YOLOv8 weights file (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Detection confidence threshold 0–1 (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="Torch device, e.g. cpu or cuda:0 (default: cpu)",
    )
    parser.add_argument(
        "--mode",
        choices=["yolo", "face"],
        default="yolo",
        help="Detection mode: 'yolo' or 'face' (default: yolo)",
    )
    parser.add_argument(
        "--callsign",
        default="SIHA",
        metavar="NAME",
        help="HUD callsign displayed in the top-left corner (default: SIHA)",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Disable the UAV HUD overlay (plain annotated output)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    """Construct AppConfig from parsed CLI arguments."""
    return AppConfig(
        camera=CameraConfig(index=args.camera),
        detector=DetectorConfig(
            model_path=args.model,
            confidence=args.confidence,
            device=args.device,
        ),
        log_level=args.log_level,
    )


def run(config: AppConfig, mode: str, callsign: str, use_hud: bool) -> None:
    """Main capture-and-detect loop.

    Args:
        config:    Application configuration.
        mode:      ``"yolo"`` or ``"face"``.
        callsign:  HUD callsign string.
        use_hud:   When True, render the UAV HUD overlay.
    """
    logger = logging.getLogger(__name__)

    # ── Build detector ────────────────────────────────────────────────────
    if mode == "face":
        detector = FaceDetector(
            scale_factor=config.face_detector.scale_factor,
            min_neighbors=config.face_detector.min_neighbors,
        )
        mode_label = "FACE"
        logger.info("Running in face-detection mode")
    else:
        detector = YoloDetector(
            model_path=config.detector.model_path,
            confidence=config.detector.confidence,
            device=config.detector.device,
        )
        mode_label = "YOLO"
        logger.info("Running in YOLO object-detection mode")

    hud = HUDRenderer(callsign=callsign) if use_hud else None

    # ── Capture loop ──────────────────────────────────────────────────────
    with Camera(
        index=config.camera.index,
        width=config.camera.width,
        height=config.camera.height,
    ) as cam:
        logger.info("Starting capture loop – press 'q' to quit")

        while True:
            frame = cam.read()
            if frame is None:
                logger.error("Camera returned no frame – stopping")
                break

            try:
                if hud is not None:
                    # HUD path: raw detections → drone overlay
                    detections = detector.detect_raw(frame)
                    output = hud.render(frame, detections=detections, mode=mode_label)
                else:
                    # Plain path: annotated frame from detector
                    output = detector.detect(frame)
            except Exception:
                logger.exception("Processing failed on this frame – skipping")
                output = frame

            cv2.imshow(config.window_title, output)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key received")
                break

    cv2.destroyAllWindows()


def main() -> None:
    """Application entry point."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        config = build_config(args)
        config.validate()
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    try:
        run(
            config,
            mode=args.mode,
            callsign=args.callsign,
            use_hud=not args.no_hud,
        )
    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    main()
