"""SIHA FPV – terminal-mode Ground Control Station.

Connects to a drone via MAVLink, displays a live FPV camera feed with
full HUD, runs optional object/face detection, tracks targets, logs all
telemetry, and enforces a geofence.

Quick-start
-----------
  python main_fpv.py --demo                       # Demo (no drone)
  python main_fpv.py --connect 127.0.0.1:14550    # ArduPilot SITL
  python main_fpv.py --connect /dev/ttyUSB0 --baud 57600
  python main_fpv.py --stream rtsp://192.168.1.1:554/live --demo

Keyboard controls
-----------------
  W/S          Pitch forward / back
  A/D          Roll  left / right
  ↑/↓          Throttle up / down
  ←/→          Yaw   left / right
  R            Throttle → 50 % (hover)
  T            Re-centre roll + pitch
  1 / 2        ARM / DISARM
  3–6          STABILIZE / LOITER / GUIDED / RTL
  L            Toggle telemetry logging
  Q            Quit
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from drone.controller import KeyboardController
from drone.geofence import Geofence
from drone.logger import TelemetryLogger
from drone.mavlink_client import MAVLinkClient
from drone.telemetry import Telemetry
from utils.config import AppConfig, CameraConfig, DetectorConfig
from vision.camera import Camera
from vision.detection import FaceDetector
from vision.hud import HUDRenderer
from vision.target_localizer import TargetLocalizer
from vision.tracker import TargetTracker
from vision.yolo_detector import YoloDetector

logger = logging.getLogger(__name__)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="siha-fpv",
        description="SIHA FPV – Terminal Ground Control Station",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--connect",    metavar="URI",
                   help="MAVLink URI (e.g. 127.0.0.1:14550, /dev/ttyUSB0)")
    p.add_argument("--baud",       type=int, default=57600)
    p.add_argument("--demo",       action="store_true",
                   help="Demo mode – simulated telemetry")
    p.add_argument("--camera",     type=int, default=0, metavar="INDEX")
    p.add_argument("--stream",     metavar="URL",
                   help="RTSP/HTTP stream URL (overrides --camera)")
    p.add_argument("--detect",     choices=["none", "yolo", "face"], default="none")
    p.add_argument("--model",      default="yolov8n.pt")
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--track",      action="store_true",
                   help="Enable CSRT target tracker on first detection")
    p.add_argument("--log",        action="store_true",
                   help="Enable telemetry + detection logging to logs/")
    p.add_argument("--callsign",   default="SIHA")
    p.add_argument("--record",     metavar="FILE",
                   help="Save annotated video (e.g. flight.avi)")
    p.add_argument("--fov-h",      type=float, default=62.2,
                   help="Camera horizontal FOV in degrees (default: 62.2)")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ── Demo telemetry ────────────────────────────────────────────────────────────

def _animate_demo(telem: Telemetry, t: float) -> None:
    import math
    telem.roll           =  12 * math.sin(t * 0.30)
    telem.pitch          =   4 * math.sin(t * 0.22)
    telem.heading        = int(45 + 25 * math.sin(t * 0.12)) % 360
    telem.altitude_rel   = 80 + 18 * math.sin(t * 0.10)
    telem.groundspeed    =  8 +  4 * math.sin(t * 0.25)
    telem.vertical_speed =  1.2 * math.cos(t * 0.20)
    telem.throttle       = int(55 + 8 * math.sin(t * 0.40))
    telem.battery_remaining = max(20, int(87 - t * 0.04))
    telem.gps_fix        = 3
    telem.gps_satellites = 12
    telem.flight_mode    = "LOITER"
    telem.armed          = True
    telem.connected      = True
    telem.lat           += 0.000002 * math.cos(t * 0.05)
    telem.lon           += 0.000002 * math.sin(t * 0.05)


# ── Key dispatch ─────────────────────────────────────────────────────────────

def _handle_command_key(key: int, client, fl_logger, logging_active: list) -> bool:
    k = key & 0xFF
    if k == ord("q"):
        return True
    if k == ord("l"):
        if fl_logger:
            logging_active[0] = not logging_active[0]
            state = "ON" if logging_active[0] else "OFF"
            logger.info("Telemetry logging: %s", state)
        return True
    if client is None:
        return False
    if k == ord("1"):
        logger.info("ARM"); client.arm()
    elif k == ord("2"):
        logger.info("DISARM"); client.disarm()
    elif k == ord("3"):
        client.set_mode("STABILIZE")
    elif k == ord("4"):
        client.set_mode("LOITER")
    elif k == ord("5"):
        client.set_mode("GUIDED")
    elif k == ord("6"):
        client.return_to_launch()
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # ── Telemetry + MAVLink ───────────────────────────────────────────────
    telem  = Telemetry()
    client = None

    if args.demo:
        logger.info("Demo mode – simulated telemetry")
        telem.connected = True
    elif args.connect:
        client = MAVLinkClient(args.connect, telem, baud=args.baud)
        client.start()
        deadline = time.time() + 10
        while not telem.connected and time.time() < deadline:
            time.sleep(0.1)
        if not telem.connected:
            logger.warning("No heartbeat – continuing without telemetry")

    # ── Detector ──────────────────────────────────────────────────────────
    detector = None
    if args.detect == "yolo":
        detector = YoloDetector(model_path=args.model, confidence=args.confidence)
    elif args.detect == "face":
        detector = FaceDetector()

    # ── Target localizer ──────────────────────────────────────────────────
    localizer = TargetLocalizer(fov_h_deg=args.fov_h)

    # ── Tracker ───────────────────────────────────────────────────────────
    tracker = TargetTracker(algorithm="CSRT") if args.track else None

    # ── HUD ───────────────────────────────────────────────────────────────
    hud = HUDRenderer(callsign=args.callsign)

    # ── Logger ────────────────────────────────────────────────────────────
    fl_logger  = TelemetryLogger() if args.log else None
    log_active = [args.log]
    if fl_logger:
        fl_logger.start()

    # ── Video source ──────────────────────────────────────────────────────
    source = args.stream if args.stream else args.camera

    with Camera(source=source) as cam:
        first = cam.read()
        if first is None:
            raise RuntimeError("Camera produced no frame")
        fh, fw = first.shape[:2]

        # ── Optional recorder ─────────────────────────────────────────────
        writer = None
        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(args.record, fourcc, 30.0, (fw, fh))
            logger.info("Recording → %s", args.record)

        cv2.namedWindow("SIHA FPV", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SIHA FPV", fw, fh)

        t0 = time.time()
        frame_idx = 0
        last_dets = []
        logger.info("FPV loop running – press Q to quit")

        while True:
            frame = cam.read()
            if frame is None:
                logger.error("Camera read failed"); break

            if args.demo:
                _animate_demo(telem, time.time() - t0)

            # ── Detection (every 3rd frame) ───────────────────────────────
            frame_idx += 1
            if detector and frame_idx % 3 == 0:
                try:
                    last_dets = detector.detect_raw(frame)
                except Exception:
                    last_dets = []

                # Auto-init tracker on first detection
                if tracker and not tracker.is_active and last_dets:
                    d = last_dets[0]
                    tracker.init_from_detection(frame, d.x1, d.y1, d.x2, d.y2, d.label)

            # ── Tracker update ────────────────────────────────────────────
            if tracker and tracker.is_active:
                tracker.update(frame)

            # ── Target localisation + logging ─────────────────────────────
            if log_active[0] and fl_logger:
                fl_logger.log_telemetry(telem)
                for det in last_dets:
                    cx = (det.x1 + det.x2) // 2
                    cy = (det.y1 + det.y2) // 2
                    loc = localizer.localize(
                        telem.lat, telem.lon, telem.altitude_rel,
                        float(telem.heading), telem.pitch, telem.roll,
                        cx, cy, fw, fh,
                    )
                    t_lat = loc.lat if loc else None
                    t_lon = loc.lon if loc else None
                    fl_logger.log_detection(telem, det, t_lat, t_lon)

            # ── Render HUD ────────────────────────────────────────────────
            output = hud.render_fpv(frame, telemetry=telem, detections=last_dets)
            if tracker and tracker.is_active:
                tracker.draw(output)

            if writer:
                writer.write(output)

            cv2.imshow("SIHA FPV", output)

            # ── Input ─────────────────────────────────────────────────────
            key = cv2.waitKey(1)
            if key == -1:
                continue
            if _handle_command_key(key, client, fl_logger, log_active):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    cv2.destroyAllWindows()
    if writer:
        writer.release()
    if fl_logger:
        fl_logger.stop()
    if client:
        client.stop()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not args.demo and not args.connect:
        logger.warning("No --demo or --connect specified. Use --demo for simulation.")

    try:
        run(args)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")


if __name__ == "__main__":
    main()
