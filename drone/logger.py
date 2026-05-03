"""Flight data recorder – logs telemetry and detection events to disk.

Produces three files per flight session:
  * ``flight_YYYYMMDD_HHMMSS.csv``      – full telemetry at ~1 Hz
  * ``flight_YYYYMMDD_HHMMSS.db``       – SQLite (queryable)
  * ``flight_YYYYMMDD_HHMMSS.geojson``  – flight path + targets (GIS-ready)

Example::

    logger = TelemetryLogger(log_dir="logs")
    logger.start()
    # in loop:
    logger.log_telemetry(telem)
    logger.log_detection(telem, det, target_lat=40.412, target_lon=49.870)
    # on exit:
    logger.stop()
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from drone.telemetry import Telemetry
    from vision.hud import Detection

logger = logging.getLogger(__name__)


# ── Column definitions ────────────────────────────────────────────────────────

_TELEM_COLS = [
    "timestamp", "lat", "lon", "alt_rel", "alt_abs",
    "roll", "pitch", "yaw", "heading",
    "groundspeed", "airspeed", "vertical_speed", "throttle",
    "battery_pct", "battery_v",
    "gps_fix", "gps_sats",
    "flight_mode", "armed", "connected",
]

_DET_COLS = [
    "timestamp", "label", "confidence",
    "x1", "y1", "x2", "y2",
    "drone_lat", "drone_lon", "drone_alt",
    "target_lat", "target_lon",
]


class TelemetryLogger:
    """Asynchronous flight data recorder.

    All disk I/O happens in a background thread so the main loop is
    never blocked.

    Args:
        log_dir:         Directory for output files (created if missing).
        telem_interval:  Minimum seconds between consecutive telemetry rows.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        telem_interval: float = 1.0,
    ) -> None:
        self._log_dir       = Path(log_dir)
        self._interval      = telem_interval
        self._last_log_time = 0.0

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._csv_path     = self._log_dir / f"flight_{ts}.csv"
        self._db_path      = self._log_dir / f"flight_{ts}.db"
        self._geojson_path = self._log_dir / f"flight_{ts}.geojson"

        self._queue: deque = deque()
        self._lock  = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop  = threading.Event()

        self._trail:   List[Tuple[float, float]] = []
        self._targets: List[dict] = []

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create output files and start the background writer thread."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._init_csv()
        self._init_db()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._write_loop, daemon=True, name="flight-logger"
        )
        self._thread.start()
        logger.info("Flight logger started → %s", self._log_dir)

    def stop(self) -> None:
        """Flush remaining rows and stop the writer thread."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._export_geojson()
        logger.info("Flight logger stopped, GeoJSON exported → %s", self._geojson_path)

    # ── Public logging API ────────────────────────────────────────────────

    def log_telemetry(self, telem: "Telemetry") -> None:
        """Enqueue a telemetry snapshot (rate-limited to ``telem_interval``)."""
        now = time.time()
        if now - self._last_log_time < self._interval:
            return
        self._last_log_time = now

        row = {
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "lat":            telem.lat,
            "lon":            telem.lon,
            "alt_rel":        telem.altitude_rel,
            "alt_abs":        telem.altitude_abs,
            "roll":           telem.roll,
            "pitch":          telem.pitch,
            "yaw":            telem.yaw,
            "heading":        telem.heading,
            "groundspeed":    telem.groundspeed,
            "airspeed":       telem.airspeed,
            "vertical_speed": telem.vertical_speed,
            "throttle":       telem.throttle,
            "battery_pct":    telem.battery_remaining,
            "battery_v":      telem.battery_voltage,
            "gps_fix":        telem.gps_fix,
            "gps_sats":       telem.gps_satellites,
            "flight_mode":    telem.flight_mode,
            "armed":          int(telem.armed),
            "connected":      int(telem.connected),
        }
        with self._lock:
            self._queue.append(("telem", row))
            if telem.lat != 0 or telem.lon != 0:
                self._trail.append((telem.lat, telem.lon))

    def log_detection(
        self,
        telem: "Telemetry",
        det: "Detection",
        target_lat: Optional[float] = None,
        target_lon: Optional[float] = None,
    ) -> None:
        """Log a single detection event with optional GPS coordinates."""
        row = {
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "label":       det.label,
            "confidence":  round(det.confidence, 4),
            "x1": det.x1, "y1": det.y1, "x2": det.x2, "y2": det.y2,
            "drone_lat":   telem.lat,
            "drone_lon":   telem.lon,
            "drone_alt":   telem.altitude_rel,
            "target_lat":  target_lat,
            "target_lon":  target_lon,
        }
        with self._lock:
            self._queue.append(("detection", row))
            if target_lat is not None and target_lon is not None:
                self._targets.append({
                    "lat":   target_lat,
                    "lon":   target_lon,
                    "label": det.label,
                    "conf":  det.confidence,
                    "ts":    row["timestamp"],
                })

    # ── Background writer ─────────────────────────────────────────────────

    def _write_loop(self) -> None:
        while not self._stop.is_set() or self._queue:
            batch: list = []
            with self._lock:
                while self._queue:
                    batch.append(self._queue.popleft())
            for kind, row in batch:
                try:
                    if kind == "telem":
                        self._write_csv(row)
                        self._write_db_telem(row)
                    elif kind == "detection":
                        self._write_db_detection(row)
                except Exception:
                    logger.exception("Logger write error")
            time.sleep(0.05)

    # ── File initialisation ───────────────────────────────────────────────

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=_TELEM_COLS).writeheader()

    def _write_csv(self, row: dict) -> None:
        with open(self._csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_TELEM_COLS).writerow(
                {k: row.get(k, "") for k in _TELEM_COLS}
            )

    def _init_db(self) -> None:
        con = sqlite3.connect(self._db_path)
        con.execute(
            "CREATE TABLE IF NOT EXISTS telemetry ("
            + ", ".join(f"{c} TEXT" for c in _TELEM_COLS)
            + ")"
        )
        con.execute(
            "CREATE TABLE IF NOT EXISTS detections ("
            + ", ".join(f"{c} TEXT" for c in _DET_COLS)
            + ")"
        )
        con.commit()
        con.close()

    def _write_db_telem(self, row: dict) -> None:
        con = sqlite3.connect(self._db_path)
        vals = [str(row.get(c, "")) for c in _TELEM_COLS]
        con.execute(
            f"INSERT INTO telemetry VALUES ({','.join(['?']*len(_TELEM_COLS))})", vals
        )
        con.commit()
        con.close()

    def _write_db_detection(self, row: dict) -> None:
        con = sqlite3.connect(self._db_path)
        vals = [str(row.get(c, "")) for c in _DET_COLS]
        con.execute(
            f"INSERT INTO detections VALUES ({','.join(['?']*len(_DET_COLS))})", vals
        )
        con.commit()
        con.close()

    # ── GeoJSON export ────────────────────────────────────────────────────

    def _export_geojson(self) -> None:
        features = []

        # Flight trail as a LineString
        if len(self._trail) >= 2:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lat, lon in self._trail],
                },
                "properties": {"name": "Flight path"},
            })

        # Take-off point
        if self._trail:
            lat0, lon0 = self._trail[0]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon0, lat0]},
                "properties": {"name": "Take-off", "marker-symbol": "airport"},
            })

        # Detected targets
        for t in self._targets:
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [t["lon"], t["lat"]],
                },
                "properties": {
                    "name":       f"{t['label']} ({t['conf']:.0%})",
                    "label":      t["label"],
                    "confidence": t["conf"],
                    "timestamp":  t["ts"],
                    "marker-color": "#ff0000",
                },
            })

        geojson = {"type": "FeatureCollection", "features": features}
        self._geojson_path.write_text(json.dumps(geojson, indent=2))
