"""Telemetry dashboard widget – live values + scrolling matplotlib graphs."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from drone.telemetry import Telemetry

# Rolling window length (seconds at ~10 Hz update = 600 samples = 60 s)
_WINDOW = 600

_STYLE_VALUE  = "color: #00ff41; font-family: monospace; font-size: 13px; font-weight: bold;"
_STYLE_LABEL  = "color: #007020; font-family: monospace; font-size: 11px;"
_STYLE_ALERT  = "color: #ff2020; font-family: monospace; font-size: 13px; font-weight: bold;"
_STYLE_CAUTION = "color: #ffcc00; font-family: monospace; font-size: 13px; font-weight: bold;"
_STYLE_BG     = "background: #0a0a0a;"


class TelemetryWidget(QWidget):
    """Two-panel telemetry display.

    Top panel: live numeric readouts in a grid.
    Bottom panel: three scrolling matplotlib graphs (altitude, speed, battery).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(_STYLE_BG)

        # ── Rolling data buffers ──────────────────────────────────────────
        self._t   = deque(maxlen=_WINDOW)
        self._alt = deque(maxlen=_WINDOW)
        self._spd = deque(maxlen=_WINDOW)
        self._bat = deque(maxlen=_WINDOW)
        self._t0  = time.time()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(self._build_readout_panel())
        layout.addWidget(self._build_graph_panel())

    # ── Public ───────────────────────────────────────────────────────────

    def update_telemetry(self, telem: "Telemetry") -> None:
        """Refresh all readouts and graphs from *telem*."""
        self._update_readouts(telem)
        self._update_graphs(telem)

    # ── Readout panel ─────────────────────────────────────────────────────

    def _build_readout_panel(self) -> QGroupBox:
        box = QGroupBox("Telemetry")
        box.setStyleSheet(
            "QGroupBox { color: #00a030; font-family: monospace; font-size: 11px;"
            " border: 1px solid #003010; margin-top: 6px; }"
            " QGroupBox::title { subcontrol-origin: margin; left: 6px; }"
        )
        grid = QGridLayout(box)
        grid.setSpacing(4)

        def _row(label, attr):
            lbl = QLabel(label)
            lbl.setStyleSheet(_STYLE_LABEL)
            val = QLabel("--")
            val.setStyleSheet(_STYLE_VALUE)
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            return lbl, val

        self._v: dict[str, QLabel] = {}

        fields = [
            ("LAT",  "lat"),   ("LON",  "lon"),
            ("ALT",  "alt"),   ("VSP",  "vsp"),
            ("SPD",  "spd"),   ("HDG",  "hdg"),
            ("ROLL", "roll"),  ("PTCH", "pitch"),
            ("BAT",  "bat"),   ("VOLT", "volt"),
            ("GPS",  "gps"),   ("MODE", "mode"),
        ]
        for i, (label, key) in enumerate(fields):
            row, col = divmod(i, 2)
            lbl, val = _row(label, key)
            grid.addWidget(lbl, row, col * 2)
            grid.addWidget(val, row, col * 2 + 1)
            self._v[key] = val

        # ARM / CONN status bar
        self._arm_lbl = QLabel("DISARMED")
        self._arm_lbl.setStyleSheet(_STYLE_ALERT)
        self._arm_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self._arm_lbl, len(fields) // 2, 0, 1, 4)

        return box

    def _update_readouts(self, t: "Telemetry") -> None:
        self._v["lat"].setText(f"{t.lat:.5f}°")
        self._v["lon"].setText(f"{t.lon:.5f}°")
        self._v["alt"].setText(f"{t.altitude_rel:.1f} m")
        self._v["vsp"].setText(f"{t.vertical_speed:+.1f} m/s")
        self._v["spd"].setText(f"{t.groundspeed_kmh:.1f} km/h")
        self._v["hdg"].setText(f"{t.heading:03d}°")
        self._v["roll"].setText(f"{t.roll:+.1f}°")
        self._v["pitch"].setText(f"{t.pitch:+.1f}°")
        self._v["gps"].setText(f"{t.gps_fix_label} {t.gps_satellites}sat")
        self._v["mode"].setText(t.flight_mode)

        bat = t.battery_remaining
        bat_style = _STYLE_ALERT if bat < 20 else (_STYLE_CAUTION if bat < 40 else _STYLE_VALUE)
        self._v["bat"].setStyleSheet(bat_style)
        self._v["bat"].setText(f"{bat}%")
        self._v["volt"].setText(f"{t.battery_voltage:.2f} V" if t.battery_voltage else "--")

        if t.armed:
            self._arm_lbl.setText("✔ ARMED")
            self._arm_lbl.setStyleSheet(_STYLE_VALUE)
        else:
            self._arm_lbl.setText("✘ DISARMED")
            self._arm_lbl.setStyleSheet(_STYLE_ALERT)

    # ── Graph panel ───────────────────────────────────────────────────────

    def _build_graph_panel(self) -> QWidget:
        try:
            import matplotlib
            matplotlib.use("QtAgg")
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(4, 3), tight_layout=True, facecolor="#0a0a0a")
            self._ax_alt = fig.add_subplot(3, 1, 1)
            self._ax_spd = fig.add_subplot(3, 1, 2)
            self._ax_bat = fig.add_subplot(3, 1, 3)

            for ax, ylabel in [
                (self._ax_alt, "ALT m"),
                (self._ax_spd, "SPD km/h"),
                (self._ax_bat, "BAT %"),
            ]:
                ax.set_facecolor("#0a0a0a")
                ax.tick_params(colors="#007020", labelsize=7)
                ax.set_ylabel(ylabel, color="#007020", fontsize=7)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#003010")

            self._canvas = FigureCanvasQTAgg(fig)
            self._canvas.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
            self._has_mpl = True
            return self._canvas

        except Exception:
            self._has_mpl = False
            placeholder = QLabel("Install matplotlib for graphs")
            placeholder.setStyleSheet(_STYLE_LABEL)
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return placeholder

    def _update_graphs(self, t: "Telemetry") -> None:
        if not self._has_mpl:
            return
        elapsed = time.time() - self._t0
        self._t.append(elapsed)
        self._alt.append(t.altitude_rel)
        self._spd.append(t.groundspeed_kmh)
        self._bat.append(t.battery_remaining)

        xs = list(self._t)
        for ax, data, color in [
            (self._ax_alt, self._alt, "#00ff41"),
            (self._ax_spd, self._spd, "#00ccff"),
            (self._ax_bat, self._bat, "#ffcc00"),
        ]:
            ax.clear()
            ax.set_facecolor("#0a0a0a")
            ax.plot(xs, list(data), color=color, linewidth=1)
            ax.tick_params(colors="#007020", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#003010")

        self._canvas.draw_idle()
