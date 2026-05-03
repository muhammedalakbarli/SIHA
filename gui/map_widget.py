"""Map widget – Leaflet.js map embedded in a QWebEngineView.

Displays the drone's real-time position on an OpenStreetMap base layer,
draws the flight trail, detected targets, and geofence polygon.

Python → JavaScript bridge:
    All map updates are sent via ``page().runJavaScript()``, calling
    the functions defined in ``data/map.html``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

from PyQt6.QtCore import QUrl, pyqtSlot
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from drone.telemetry import Telemetry

logger = logging.getLogger(__name__)

_MAP_HTML = Path(__file__).resolve().parent.parent / "data" / "map.html"


class MapWidget(QWidget):
    """Leaflet map with drone position, trail, targets, and geofence.

    Falls back to a plain label if QtWebEngine is not installed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._web_available = False
        self._view = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            self._view = QWebEngineView(self)
            self._view.load(QUrl.fromLocalFile(str(_MAP_HTML)))
            layout.addWidget(self._view)
            self._web_available = True
            logger.info("Map widget: Leaflet map loaded from %s", _MAP_HTML)
        except ImportError:
            placeholder = QLabel(
                "Map unavailable.\n\nInstall PyQt6-WebEngine:\n"
                "  pip install PyQt6-WebEngine"
            )
            placeholder.setStyleSheet(
                "background:#0a0a0a; color:#007020;"
                " font-family:monospace; font-size:12px;"
            )
            layout.addWidget(placeholder)
            logger.warning("PyQt6-WebEngine not installed – map disabled")

        self._home_set = False

    # ── Public ───────────────────────────────────────────────────────────

    @pyqtSlot(object)
    def update_telemetry(self, telem: "Telemetry") -> None:
        """Update drone position, heading, altitude, and speed on the map."""
        if not self._web_available:
            return
        js = (
            f"updateDrone({telem.lat}, {telem.lon}, "
            f"{telem.heading}, {telem.altitude_rel}, {telem.groundspeed})"
        )
        self._run_js(js)

        if not self._home_set and telem.connected:
            self._run_js(f"setHome({telem.lat}, {telem.lon})")
            self._home_set = True

    def add_target(
        self,
        target_id: str,
        lat: float,
        lon: float,
        label: str,
    ) -> None:
        """Add or update a target marker on the map."""
        safe_id    = json.dumps(target_id)
        safe_label = json.dumps(label)
        self._run_js(f"addTarget({safe_id}, {lat}, {lon}, {safe_label})")

    def remove_target(self, target_id: str) -> None:
        self._run_js(f"removeTarget({json.dumps(target_id)})")

    def clear_targets(self) -> None:
        self._run_js("clearTargets()")

    def set_geofence(self, polygon: List[Tuple[float, float]]) -> None:
        """Draw a geofence polygon.  ``polygon`` is a list of (lat, lon) tuples."""
        coords_js = json.dumps([[lat, lon] for lat, lon in polygon])
        self._run_js(f"setGeofence({coords_js})")

    def clear_geofence(self) -> None:
        self._run_js("clearGeofence()")

    def clear_trail(self) -> None:
        self._run_js("clearTrail()")

    def set_auto_follow(self, enabled: bool) -> None:
        self._run_js(f"setAutoFollow({'true' if enabled else 'false'})")

    # ── Private ───────────────────────────────────────────────────────────

    def _run_js(self, js: str) -> None:
        if self._web_available and self._view:
            self._view.page().runJavaScript(js)
