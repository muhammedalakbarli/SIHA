"""SIHA GCS – Main application window.

Full PyQt6 ground-control station with:
  * Live FPV video feed (left panel)
  * Telemetry dashboard + graphs (right panel, Telemetry tab)
  * Interactive Leaflet map (right panel, Map tab)
  * Mission planner with waypoint table (right panel, Mission tab)
  * Toolbar: connect, ARM/DISARM, mode buttons, detect toggle, record
  * Status bar: connection, mode, GPS, FPS, target count
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, QTimer, pyqtSlot
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from drone.mavlink_client import MAVLinkClient
from drone.telemetry import Telemetry
from gui.map_widget import MapWidget
from gui.telemetry_widget import TelemetryWidget
from gui.video_widget import VideoWidget
from gui.workers import CameraWorker, make_camera_thread
from vision.hud import HUDRenderer

logger = logging.getLogger(__name__)

_STYLE = """
QMainWindow, QDialog { background: #0d0d0d; }
QToolBar { background: #111; border-bottom: 1px solid #003010; spacing: 4px; }
QPushButton {
    background: #111; color: #00ff41; border: 1px solid #00a030;
    padding: 4px 10px; font-family: monospace; border-radius: 2px;
}
QPushButton:hover  { background: #003010; }
QPushButton:pressed { background: #005020; }
QComboBox {
    background: #111; color: #00ff41; border: 1px solid #00a030;
    padding: 2px 6px; font-family: monospace;
}
QTabWidget::pane  { border: 1px solid #003010; }
QTabBar::tab      { background: #111; color: #007020; font-family: monospace;
                    padding: 4px 12px; border: 1px solid #003010; }
QTabBar::tab:selected { background: #003010; color: #00ff41; }
QStatusBar        { background: #0a0a0a; color: #007020; font-family: monospace; }
QLabel            { color: #007020; }
QTableWidget      { background: #0a0a0a; color: #00cc40; gridline-color: #003010;
                    font-family: monospace; font-size: 11px; }
QHeaderView::section { background: #111; color: #007020; font-family: monospace;
                        border: 1px solid #003010; }
"""


class MainWindow(QMainWindow):
    """Top-level GCS window."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SIHA – Ground Control Station")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(_STYLE)

        # ── App state ─────────────────────────────────────────────────────
        self._telem:  Telemetry          = Telemetry()
        self._client: Optional[MAVLinkClient] = None
        self._hud:    HUDRenderer        = HUDRenderer(callsign="SIHA")
        self._cam_worker: Optional[CameraWorker] = None
        self._cam_thread: Optional[QThread]      = None
        self._recording   = False
        self._video_writer = None
        self._target_counter = 0

        # ── Widgets ───────────────────────────────────────────────────────
        self._video_widget    = VideoWidget()
        self._telem_widget    = TelemetryWidget()
        self._map_widget      = MapWidget()
        self._mission_widget  = MissionWidget()

        self._setup_layout()
        self._setup_menu()
        self._setup_toolbar()
        self._setup_statusbar()

        # ── Telemetry polling timer (10 Hz) ───────────────────────────────
        self._telem_timer = QTimer(self)
        self._telem_timer.timeout.connect(self._on_telem_tick)
        self._telem_timer.start(100)

    # ── Layout ────────────────────────────────────────────────────────────

    def _setup_layout(self) -> None:
        splitter = QSplitter()
        splitter.setStyleSheet("QSplitter::handle { background: #003010; }")

        splitter.addWidget(self._video_widget)

        tabs = QTabWidget()
        tabs.addTab(self._telem_widget,   "Telemetry")
        tabs.addTab(self._map_widget,     "Map")
        tabs.addTab(self._mission_widget, "Mission")
        splitter.addWidget(tabs)

        splitter.setSizes([820, 420])
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

    # ── Menu bar ──────────────────────────────────────────────────────────

    def _setup_menu(self) -> None:
        mb = self.menuBar()
        mb.setStyleSheet(
            "QMenuBar { background:#0d0d0d; color:#007020; font-family:monospace; }"
            " QMenuBar::item:selected { background:#003010; color:#00ff41; }"
            " QMenu { background:#0d0d0d; color:#007020; font-family:monospace; }"
            " QMenu::item:selected { background:#003010; color:#00ff41; }"
        )

        # File
        fm = mb.addMenu("File")
        fm.addAction("Open Video File…", self._on_open_file)
        fm.addAction("Start Recording…", self._on_record_start)
        fm.addAction("Stop Recording",   self._on_record_stop)
        fm.addSeparator()
        fm.addAction("Quit", QApplication.quit)

        # Connection
        cm = mb.addMenu("Connection")
        cm.addAction("Connect to Drone…", self._on_connect_dialog)
        cm.addAction("Disconnect",        self._on_disconnect)
        cm.addSeparator()
        cm.addAction("Open Camera…",      self._on_open_camera_dialog)

        # Mission
        mm = mb.addMenu("Mission")
        mm.addAction("Add Waypoint",    self._mission_widget.add_waypoint)
        mm.addAction("Upload Mission",  self._on_upload_mission)
        mm.addAction("Download Mission",self._on_download_mission)
        mm.addAction("Clear Mission",   self._mission_widget.clear)

        # View
        vm = mb.addMenu("View")
        vm.addAction("Clear Map Trail", self._map_widget.clear_trail)
        vm.addAction("Clear Targets",   self._map_widget.clear_targets)

        # Help
        hm = mb.addMenu("Help")
        hm.addAction("About SIHA", self._on_about)

    # ── Toolbar ───────────────────────────────────────────────────────────

    def _setup_toolbar(self) -> None:
        tb: QToolBar = self.addToolBar("Main")
        tb.setMovable(False)

        def btn(text, slot, tooltip=""):
            b = QPushButton(text)
            b.setToolTip(tooltip)
            b.clicked.connect(slot)
            tb.addWidget(b)
            return b

        btn("Connect",  self._on_connect_dialog, "Connect to drone via MAVLink")
        tb.addSeparator()
        btn("ARM",      self._on_arm,    "Arm motors")
        btn("DISARM",   self._on_disarm, "Disarm motors")
        tb.addSeparator()

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["STABILIZE", "ALT_HOLD", "LOITER", "GUIDED", "AUTO", "RTL", "LAND"])
        self._mode_combo.setToolTip("Set flight mode")
        self._mode_combo.currentTextChanged.connect(self._on_mode_change)
        tb.addWidget(QLabel(" Mode: "))
        tb.addWidget(self._mode_combo)
        tb.addSeparator()

        self._detect_combo = QComboBox()
        self._detect_combo.addItems(["No Detection", "YOLO", "Face"])
        self._detect_combo.currentTextChanged.connect(self._on_detect_change)
        tb.addWidget(QLabel(" Detect: "))
        tb.addWidget(self._detect_combo)
        tb.addSeparator()

        self._rec_btn = btn("⏺ REC", self._on_record_toggle, "Toggle video recording")
        btn("RTL",  lambda: self._client and self._client.return_to_launch(), "Return to launch")
        btn("LAND", lambda: self._client and self._client.land(), "Land")

    # ── Status bar ────────────────────────────────────────────────────────

    def _setup_statusbar(self) -> None:
        sb: QStatusBar = self.statusBar()
        self._sb_conn   = QLabel("Disconnected")
        self._sb_mode   = QLabel("--")
        self._sb_gps    = QLabel("GPS: --")
        self._sb_fps    = QLabel("FPS: --")
        self._sb_targets = QLabel("Targets: 0")
        for w in [self._sb_conn, self._sb_mode, self._sb_gps, self._sb_fps, self._sb_targets]:
            sb.addWidget(w)
            sb.addWidget(_sep())

    # ── Slots – connection ────────────────────────────────────────────────

    @pyqtSlot()
    def _on_connect_dialog(self) -> None:
        dlg = ConnectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        uri, baud, callsign = dlg.values()
        self._hud = HUDRenderer(callsign=callsign)
        self._telem = Telemetry()
        if self._client:
            self._client.stop()
        self._client = MAVLinkClient(uri, self._telem, baud=baud)
        self._client.start()
        self._sb_conn.setText(f"Connecting → {uri}")
        logger.info("Connecting to %s", uri)

    @pyqtSlot()
    def _on_disconnect(self) -> None:
        if self._client:
            self._client.stop()
            self._client = None
        self._telem.connected = False
        self._sb_conn.setText("Disconnected")

    @pyqtSlot()
    def _on_open_camera_dialog(self) -> None:
        text, ok = QInputDialog.getText(
            self, "Open Camera",
            "Camera index (0, 1…) or URL (rtsp://…):",
            text="0",
        )
        if ok and text.strip():
            source = int(text) if text.strip().isdigit() else text.strip()
            self._start_camera(source)

    @pyqtSlot()
    def _on_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", "Video (*.mp4 *.avi *.mkv *.mov)"
        )
        if path:
            self._start_camera(path)

    # ── Slots – drone commands ────────────────────────────────────────────

    @pyqtSlot()
    def _on_arm(self) -> None:
        if self._client:
            self._client.arm()

    @pyqtSlot()
    def _on_disarm(self) -> None:
        if self._client:
            self._client.disarm()

    @pyqtSlot(str)
    def _on_mode_change(self, mode: str) -> None:
        if self._client and self._telem.connected:
            self._client.set_mode(mode)

    # ── Slots – detection ─────────────────────────────────────────────────

    @pyqtSlot(str)
    def _on_detect_change(self, choice: str) -> None:
        if self._cam_worker is None:
            return
        if choice == "YOLO":
            try:
                from vision.yolo_detector import YoloDetector
                self._cam_worker.set_detector(YoloDetector())
            except Exception as exc:
                logger.error("YOLO load failed: %s", exc)
        elif choice == "Face":
            from vision.detection import FaceDetector
            self._cam_worker.set_detector(FaceDetector())
        else:
            self._cam_worker.set_detector(None)

    # ── Slots – recording ─────────────────────────────────────────────────

    @pyqtSlot()
    def _on_record_toggle(self) -> None:
        if self._recording:
            self._on_record_stop()
        else:
            self._on_record_start()

    def _on_record_start(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Recording", f"flight_{int(time.time())}.avi",
            "Video (*.avi *.mp4)"
        )
        if not path:
            return
        import cv2
        self._video_writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"XVID"), 30.0, (1280, 720)
        )
        self._recording = True
        self._rec_btn.setStyleSheet("color: #ff2020; border-color: #ff2020;")
        logger.info("Recording started → %s", path)

    def _on_record_stop(self) -> None:
        if self._video_writer:
            self._video_writer.release()
            self._video_writer = None
        self._recording = False
        self._rec_btn.setStyleSheet("")
        logger.info("Recording stopped")

    # ── Slots – mission ───────────────────────────────────────────────────

    def _on_upload_mission(self) -> None:
        QMessageBox.information(self, "Mission", "Mission upload: connect a drone first.")

    def _on_download_mission(self) -> None:
        QMessageBox.information(self, "Mission", "Mission download: connect a drone first.")

    # ── Slots – frame ready ───────────────────────────────────────────────

    @pyqtSlot(object)
    def _on_frame_ready(self, frame) -> None:
        self._video_widget.update_frame(frame)
        if self._recording and self._video_writer:
            import cv2
            resized = cv2.resize(frame, (1280, 720))
            self._video_writer.write(resized)

    # ── Telemetry tick ────────────────────────────────────────────────────

    @pyqtSlot()
    def _on_telem_tick(self) -> None:
        t = self._telem
        self._telem_widget.update_telemetry(t)
        self._map_widget.update_telemetry(t)

        self._sb_conn.setText("CONNECTED" if t.connected else "Disconnected")
        self._sb_mode.setText(t.flight_mode)
        self._sb_gps.setText(f"GPS: {t.gps_fix_label} {t.gps_satellites}sat")

    # ── Camera helpers ────────────────────────────────────────────────────

    def _start_camera(self, source) -> None:
        self._stop_camera()
        self._cam_worker = CameraWorker(source, self._telem, self._hud)
        self._cam_thread = make_camera_thread(self._cam_worker)
        self._cam_worker.frame_ready.connect(self._on_frame_ready)
        self._cam_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Camera Error", msg)
        )
        self._cam_thread.start()
        logger.info("Camera started: %s", source)

    def _stop_camera(self) -> None:
        if self._cam_worker:
            self._cam_worker.stop()
        if self._cam_thread:
            self._cam_thread.quit()
            self._cam_thread.wait(3000)
        self._cam_worker = None
        self._cam_thread = None

    # ── About ─────────────────────────────────────────────────────────────

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "About SIHA",
            "<b>SIHA – AI Vision & FPV Ground Control Station</b><br/>"
            "YOLOv8 · MAVLink · OpenCV · PyQt6<br/><br/>"
            "Built for professional FPV drone operations.",
        )

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._stop_camera()
        if self._client:
            self._client.stop()
        if self._video_writer:
            self._video_writer.release()
        event.accept()


# ── Helper widgets ────────────────────────────────────────────────────────────

class ConnectDialog(QDialog):
    """Dialog for entering a MAVLink connection URI."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect to Drone")
        self.setStyleSheet(parent.styleSheet() if parent else "")
        self.setFixedWidth(400)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        def _row(label, widget):
            box = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(90)
            box.addWidget(lbl)
            box.addWidget(widget)
            return box

        self._uri   = QLineEdit("127.0.0.1:14550")
        self._baud  = QLineEdit("57600")
        self._call  = QLineEdit("SIHA")

        layout.addLayout(_row("URI / Port:", self._uri))
        layout.addLayout(_row("Baud rate:", self._baud))
        layout.addLayout(_row("Callsign:", self._call))

        # Quick-preset buttons
        presets = QHBoxLayout()
        for label, val in [("SITL", "127.0.0.1:14550"), ("UDP 14550", "udpin:0.0.0.0:14550")]:
            b = QPushButton(label)
            b.clicked.connect(lambda _, v=val: self._uri.setText(v))
            presets.addWidget(b)
        layout.addLayout(presets)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        try:
            baud = int(self._baud.text())
        except ValueError:
            baud = 57600
        return self._uri.text().strip(), baud, self._call.text().strip()


class MissionWidget(QWidget):
    """Simple waypoint table for mission planning."""

    _HEADERS = ["#", "Type", "Lat", "Lon", "Alt (m)", "Speed (m/s)"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        btn_row = QHBoxLayout()
        for label, slot in [
            ("+ Waypoint", self.add_waypoint),
            ("- Remove",   self.remove_selected),
            ("Clear All",  self.clear),
        ]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._table = QTableWidget(0, len(self._HEADERS))
        self._table.setHorizontalHeaderLabels(self._HEADERS)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)

    def add_waypoint(self) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        defaults = [str(row + 1), "WAYPOINT", "0.0", "0.0", "50", "10"]
        for col, val in enumerate(defaults):
            self._table.setItem(row, col, QTableWidgetItem(val))

    def remove_selected(self) -> None:
        for idx in sorted(
            {i.row() for i in self._table.selectedItems()}, reverse=True
        ):
            self._table.removeRow(idx)

    def clear(self) -> None:
        self._table.setRowCount(0)

    def waypoints(self) -> list:
        wps = []
        for r in range(self._table.rowCount()):
            def _v(c):
                item = self._table.item(r, c)
                return item.text() if item else ""
            wps.append({
                "type":  _v(1),
                "lat":   float(_v(2) or 0),
                "lon":   float(_v(3) or 0),
                "alt":   float(_v(4) or 50),
                "speed": float(_v(5) or 10),
            })
        return wps


def _sep() -> QLabel:
    s = QLabel("|")
    s.setStyleSheet("color: #003010;")
    return s
