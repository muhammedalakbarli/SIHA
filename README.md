# SIHA — UAV AI Vision & Ground Control Station

> Real-time object detection, military-style HUD, MAVLink drone control, target tracking, and geofencing — all in one open-source GCS.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![MAVLink](https://img.shields.io/badge/MAVLink-ArduPilot%2FPX4-green)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![CI](https://github.com/muhammedalakbarli/SIHA/actions/workflows/ci.yml/badge.svg)

---

## Overview

**SIHA** is a Python-based AI vision system and Ground Control Station (GCS) designed for Unmanned Aerial Vehicles (UAVs). It combines real-time computer vision with full drone telemetry and control via the MAVLink protocol, targeting both ArduPilot and PX4 autopilot platforms.

The system offers three operational modes:

| Mode | Entry Point | Description |
|---|---|---|
| CLI Detection | `main.py` | Webcam/stream object & face detection with optional HUD |
| Terminal FPV | `main_fpv.py` | Full drone piloting, detection, tracking, and logging in terminal |
| GUI GCS | `main_gui.py` | PyQt6 Ground Control Station with live video, map, and telemetry |

---

## Features

### Computer Vision
- **YOLOv8** real-time object detection — 80 COCO classes (person, vehicle, aircraft, boat, etc.)
- **Haar Cascade** face detection — CPU-only, no GPU required
- **CSRT / KCF / MIL** multi-algorithm target tracker with auto-init from detections
- **Target GPS localization** — pixel coordinates → GPS lat/lon via flat-earth approximation with gimbal pitch compensation

### HUD & Instruments
- Green phosphor CRT-style military HUD overlay (scanlines, vignette, corner brackets)
- Full glass-cockpit instruments: ADI (Artificial Horizon), altitude tape, speed tape, heading tape
- Live HUD data: callsign, timestamp, GPS fix, REC indicator, FPS, battery, altitude, speed, heading
- Mini compass rose

### Drone Control & Telemetry (MAVLink)
- Async threaded MAVLink client — serial, UDP, and TCP connection support
- Live telemetry: position, attitude, velocity, battery, GPS fix, RC channels
- Commands: ARM / DISARM, set flight mode, RC override, takeoff, land, RTL
- Keyboard control (Mode 2): W/S pitch, A/D roll, arrows throttle/yaw — self-centering roll/pitch, sticky throttle

### Safety & Logging
- **Geofencing** — polygon boundary with ray-casting algorithm; configurable breach action (RTL / DISARM)
- **Altitude limits** — min/max altitude constraints with breach callbacks
- **Flight logger** — async CSV + SQLite + GeoJSON recording of telemetry and detection events

### GUI (PyQt6)
- Live FPV video panel with detection overlays
- Telemetry graphs (matplotlib)
- Interactive Leaflet map widget
- Mission planner with waypoint table
- Toolbar: connect, ARM/DISARM, detect toggle, record

---

## Architecture

```
SIHA/
├── main.py                  # CLI object/face detection
├── main_fpv.py              # Terminal FPV + drone control
├── main_gui.py              # PyQt6 Ground Control Station
│
├── drone/
│   ├── mavlink_client.py    # Async MAVLink telemetry + commands
│   ├── telemetry.py         # Telemetry dataclass (position, attitude, power, GPS)
│   ├── controller.py        # Keyboard → RC channel mapping (Mode 2)
│   ├── geofence.py          # Polygon containment + breach callbacks
│   ├── logger.py            # CSV / SQLite / GeoJSON flight recorder
│   └── gimbal.py            # Gimbal control
│
├── vision/
│   ├── camera.py            # USB / RTSP / HTTP / GStreamer capture
│   ├── yolo_detector.py     # YOLOv8 detector (annotated frames + raw detections)
│   ├── detection.py         # Haar Cascade face detector
│   ├── hud.py               # Military HUD renderer
│   ├── attitude.py          # Glass-cockpit ADI + instrument tapes
│   ├── tracker.py           # CSRT / KCF / MIL target tracker
│   └── target_localizer.py  # Pixel → GPS coordinate estimation
│
├── gui/
│   ├── main_window.py       # Main GCS window + toolbars
│   ├── video_widget.py      # Live video feed panel
│   ├── telemetry_widget.py  # Telemetry graphs + data display
│   ├── map_widget.py        # Interactive Leaflet map
│   └── workers.py           # QThread background workers
│
├── utils/
│   └── config.py            # Dataclass-based configuration
│
├── scripts/
│   └── train.py             # Custom YOLOv8 model training
│
├── data/
│   └── fpv.yaml             # YOLOv8 dataset config (6 classes)
│
├── tests/                   # Pytest unit tests
├── Dockerfile               # Multi-stage Docker build
└── docker-compose.yml       # FPV + SITL + Training services
```

---

## Requirements

- Python 3.10+
- Webcam or RTSP/HTTP video stream
- (Optional) ArduPilot or PX4 autopilot — physical or SITL simulation

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/muhammedalakbarli/SIHA.git
cd SIHA

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically by Ultralytics on first run.

---

## Usage

### CLI Detection (`main.py`)

```bash
# Default: YOLO object detection on camera 0
python main.py

# Face detection mode
python main.py --mode face

# RTSP stream with custom model and confidence
python main.py --camera rtsp://192.168.1.1/stream --model yolov8s.pt --confidence 0.6

# GPU acceleration
python main.py --device cuda:0

# Enable HUD overlay
python main.py --hud
```

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera index or stream URL (RTSP/HTTP) |
| `--model` | `yolov8n.pt` | Path to YOLOv8 weights |
| `--confidence` | `0.5` | Detection confidence threshold (0–1) |
| `--device` | `cpu` | Torch device: `cpu`, `cuda:0` |
| `--mode` | `yolo` | `yolo` or `face` |
| `--hud` | off | Enable military HUD overlay |

---

### Terminal FPV Mode (`main_fpv.py`)

```bash
# Demo mode (simulated telemetry, no real drone needed)
python main_fpv.py --demo

# Connect to real drone via UDP (ArduPilot SITL or physical)
python main_fpv.py --connection udp:127.0.0.1:14550

# Connect via serial (physical autopilot)
python main_fpv.py --connection /dev/ttyUSB0,57600

# Enable YOLO detection + CSRT tracking
python main_fpv.py --demo --detect --tracker csrt
```

**Keyboard Controls (Mode 2):**

| Key | Action |
|---|---|
| `W / S` | Pitch forward / backward |
| `A / D` | Roll left / right |
| `↑ / ↓` | Throttle up / down |
| `← / →` | Yaw left / right |
| `Space` | ARM / DISARM |
| `L` | Land |
| `R` | Return to Launch (RTL) |
| `Q` | Quit |

---

### PyQt6 GCS (`main_gui.py`)

```bash
python main_gui.py
```

Use the toolbar to connect to your drone, toggle detection, and start recording. The left panel shows the live FPV feed; the right panel provides tabbed access to telemetry graphs, interactive map, and mission planner.

---

## Docker

```bash
# Build and run FPV mode
docker-compose up fpv

# Run ArduPilot SITL simulation
docker-compose up sitl

# Run model training
docker-compose up training
```

---

## Custom Model Training

The dataset config at `data/fpv.yaml` defines 6 UAV-specific classes:
`person`, `vehicle`, `aircraft`, `boat`, `weapon`, `unknown`

```bash
python scripts/train.py --data data/fpv.yaml --epochs 100 --model yolov8n.pt
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Technology | Role |
|---|---|
| Python 3.10+ | Core language |
| OpenCV 4.8+ | Camera capture, Haar Cascade, image processing |
| Ultralytics YOLOv8 | Real-time object detection |
| PyMAVLink 2.4+ | MAVLink protocol (ArduPilot / PX4) |
| PyQt6 | Desktop GUI and GCS interface |
| Pygame | Joystick input support |
| NumPy | Numerical operations |
| Matplotlib | Telemetry graphs |
| Docker | Containerization |
| GitHub Actions | CI/CD (test + lint + docker build) |

---

## License

MIT

---

## Author

**Muhamməd Alakbarlı**
GitHub: [@muhammedalakbarli](https://github.com/muhammedalakbarli)
