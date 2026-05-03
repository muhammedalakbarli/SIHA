# ── SIHA FPV – Docker image ──────────────────────────────────────────────────
# Multi-stage build: keeps the final image lean.
#
# Build:
#   docker build -t siha-fpv .
#
# Run (terminal FPV, camera passthrough required):
#   docker run --rm -it \
#     --device /dev/video0:/dev/video0 \
#     --network host \
#     siha-fpv python main_fpv.py --demo
#
# Run (GUI – needs X11 forwarding on Linux):
#   docker run --rm -it \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     --device /dev/video0:/dev/video0 \
#     --network host \
#     siha-fpv python main_gui.py

# ── Stage 1: build wheels ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed to build some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="SIHA FPV Project" \
      description="SIHA AI Vision & FPV Ground Control Station" \
      version="1.0"

WORKDIR /app

# Runtime OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        ffmpeg \
        v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages from pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
 && rm -rf /wheels

# Copy application source
COPY . .

# Create log directory
RUN mkdir -p /app/logs

# Default: terminal FPV in demo mode
CMD ["python", "main_fpv.py", "--demo", "--log-level", "INFO"]
