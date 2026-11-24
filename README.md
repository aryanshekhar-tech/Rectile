# Rectile — UAV Target Acquisition & Optical-Flow Guidance

**Rectile** is a low-latency target acquisition and tracking system for **NVIDIA Jetson Orin**. It pairs on-demand multi-signal AI acquisition with continuous **60 FPS Pyramidal Lucas-Kanade (KLT) optical-flow tracking**, streaming live annotated video and telemetry over **ZeroMQ** to a PySide6/PyQt Ground Control Station.

## Architecture

- **Capture & Streaming (`VideoSource`, `AsyncJpegPool`):** 1080p @ 60 FPS GStreamer SHM / video / RTSP capture decoupled from AI inference.
- **Hybrid Target Acquisition (`run_hybrid_search`):**
  - *Moving Vehicles:* Sparse optical flow + affine RANSAC ego-motion cancellation (<15 ms) with vegetation filtering and HSV color verification.
  - *General Objects:* CUDA **Grounding DINO** (`./weights`) with center-boresight and full-frame passes.
  - *Single Characters:* Morphological Top-Hat filtering + **EasyOCR**.
- **Core-Snapped FB-KLT Tracker (`AnchoredKLTTracker`):** Tightens initial boxes onto the target core, tracks via forward-backward KLT (`fb_err < 2.8 px`), and uses local template-matching rescue during motion blur.

## Repository Layout

- `jetson_node.py` — Edge perception, hybrid acquisition & 60 FPS KLT tracking node
- `tracker_nanosam.py` — Ground station launcher & ZMQ client
- `uav_gui/` — PySide6 / PyQt operator dashboard
- `weights/` & `models/` — Local model weights

## Usage & ZMQ Protocol

- **Run Edge Node:** `python3 jetson_node.py` (or `--source FVC_6_1080p.mp4`)
- **Run Ground Station:** `python3 tracker_nanosam.py`
- **Port 5555 (`PUB`):** Multipart `[JSON telemetry, JPEG frame]`
- **Port 5556 (`SUB`):** JSON commands (`deploy`, `clear`, `optic`)
