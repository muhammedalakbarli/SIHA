"""SIHA GCS – PyQt6 GUI entry point.

Launches the full graphical Ground Control Station with video feed,
telemetry dashboard, interactive map, and mission planner.

Usage::

    python main_gui.py
    python main_gui.py --connect 127.0.0.1:14550 --camera 0
    python main_gui.py --connect rtsp://192.168.1.1:554/live
"""

from __future__ import annotations

import argparse
import logging
import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="siha-gui",
        description="SIHA – Ground Control Station (GUI)",
    )
    p.add_argument("--connect", metavar="URI",
                   help="MAVLink connection URI (e.g. 127.0.0.1:14550)")
    p.add_argument("--baud",   type=int, default=57600)
    p.add_argument("--camera", default="0",
                   help="Camera index or RTSP URL (default: 0)")
    p.add_argument("--callsign", default="SIHA")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SIHA GCS")
    app.setOrganizationName("SIHA")

    window = MainWindow()

    # Auto-connect if URI supplied
    if args.connect:
        from drone.mavlink_client import MAVLinkClient
        from drone.telemetry import Telemetry
        from vision.hud import HUDRenderer
        window._telem  = Telemetry()
        window._hud    = HUDRenderer(callsign=args.callsign)
        window._client = MAVLinkClient(args.connect, window._telem, baud=args.baud)
        window._client.start()

    # Auto-open camera
    source = int(args.camera) if args.camera.isdigit() else args.camera
    window._start_camera(source)

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
