# SIHA – AI Vision System

Real-time object and face detection from a live webcam feed, powered by
[YOLOv8](https://github.com/ultralytics/ultralytics) and OpenCV.

---

## Features

| Feature | Details |
|---|---|
| Object detection | YOLOv8 nano model – 80 COCO classes |
| Face detection | OpenCV Haar Cascade (no GPU required) |
| Configurable CLI | Camera index, model path, confidence threshold, device |
| Structured logging | Timestamps, log-level control |
| Graceful shutdown | Press `q` or `Ctrl+C` |

---

## Project Structure

```
SIHA/
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
├── .gitignore
├── utils/
│   ├── __init__.py
│   └── config.py         # Dataclass-based configuration
├── vision/
│   ├── __init__.py
│   ├── camera.py         # Camera capture (context-manager support)
│   ├── detection.py      # Haar Cascade face detector
│   └── yolo_detector.py  # YOLOv8 object detector
└── tests/
    ├── test_camera.py
    ├── test_config.py
    └── test_detection.py
```

---

## Requirements

- Python 3.10+
- Webcam

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd SIHA

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The YOLOv8 nano weights (`yolov8n.pt`) are downloaded automatically by
`ultralytics` on first run if not already present.

---

## Usage

```bash
# Default: YOLO object detection on camera 0
python main.py

# Face detection mode
python main.py --mode face

# Custom camera, model, and confidence
python main.py --camera 1 --model yolov8s.pt --confidence 0.6

# GPU acceleration
python main.py --device cuda:0

# Verbose debug output
python main.py --log-level DEBUG
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--model` | `yolov8n.pt` | Path to YOLOv8 weights |
| `--confidence` | `0.5` | Detection confidence threshold (0–1) |
| `--device` | `cpu` | Torch device (`cpu`, `cuda:0`, …) |
| `--mode` | `yolo` | `yolo` or `face` |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

Press **`q`** to quit the video window.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT
