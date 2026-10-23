# 🎯 Rectile: High-Speed UAV Guidance & Ground Station Dashboard

An edge-optimized UAV visual target acquisition, terminal guidance, and optical tracking system. Designed for terminal dive guidance (up to 60 m/s), multi-target detection, and multi-spectral optic filtering over ZeroMQ.

This repository includes the **Rectile Edge Node** (for NVIDIA Jetson) and the **Rectile Ground Station GUI** (a PyQt-based viewer and command dashboard).

---

## ✨ System Features

### Core Tracking & Perception (Edge Node)
* **2-Stage Adaptive Acquisition:** Fast boresight center-crop with automatic full-frame fallback for edge targets.
* **Anchored KLT Tracking:** Pyramidal Lucas-Kanade optical flow locked to physical pixels, synchronized to camera frame IDs to eliminate drift during rapid pitch/dive maneuvers.
* **Optic Filter Simulation:** White Hot, Black Hot, Ironbow, and Rainbow thermal colormaps applied dynamically.

### Ground Station Dashboard (GUI)
* **Dynamic Source Switching:** Instantly swap between ZMQ video feeds, local webcams (e.g., Camera 0, Camera 1), local MP4 files, and live RTSP network streams (e.g., `rtsp://{remote_ip}:8554/stream`).
* **Intelligent MP4 Recording:** Dedicated capture to `recording_<timestamp>_<id>.mp4`. Features a bounded pending queue that prevents application crashes by safely skipping frames (and reporting them) if the video encoder overloads. 
* **Paced FPS Processing:** Video file playback is automatically paced to the source's nominal FPS for realistic viewing, rather than decoding at maximum CPU speed. 
* **Command History & Pinning:** Target prompts (e.g., "white car", "helipad") are saved locally. Right-click history chips to pin your most-used commands for instant deployment.
* **Tactical UI:** Full object rectangles replaced with minimalist fixed-green corner brackets for a cleaner, unobstructed target view.

---

## 📦 Dependencies & Requirements

To run Rectile, ensure your environment meets the following requirements. 

**Hardware:**
* **Edge Node:** NVIDIA Jetson (Orin Nano/NX, Xavier NX, Nano) running JetPack 5.x/6.x.
* **Ground Station:** Linux, Windows (WSL2), or macOS with a GUI environment.

**Python Dependencies:**
* `PySide6` / `PyQt` (For the Ground Station Dashboard)
* `opencv-python` (For frame capture, encoding, and display)
* `pyzmq` (For edge-to-ground telemetry and video streaming)
* `transformers`, `torch` (For Grounding DINO / AI perception)
* `pytest` (Optional: For running GUI test suites)

*To install core requirements:*
`pip3 install -r requirements.txt`

---

## 🛠️ Installation & GUI Patching

### 1. Standard Installation
Clone the repository to your Jetson or Ground Station laptop:
`git clone https://github.com/aryanshekhar-tech/Rectile.git`
`cd Rectile`

### 2. Applying GUI Updates (If utilizing a patch archive)
If you have received a local `uav_gui` update patch, extract it outside the project folder (e.g., to `Downloads`) and apply it using the built-in update helper. This safely replaces the GUI without touching the AI models, ZMQ server, or C++ sources.

`python3 apply_gui_update.py --target ~/Rectile`
*Note: The helper automatically backs up your previous GUI in `~/Rectile/backups/gui_viewer_fix_<timestamp>/`.*

---

## 🚀 Execution Guide

### Launching the Ground Station (Laptop)

Navigate to the Rectile directory and launch the dashboard.

**Option A: Connect to Live Edge Node (Default ZMQ)**
`python3 ~/Rectile/tracker_nanosam.py`
*(Defaults to pulling video from `http://localhost:5000/video` if the Edge Node is running).*

**Option B: Play Local Video Simulation**
`python3 ~/Rectile/tracker_nanosam.py /path/to/flight_video.mp4`

### Dashboard Configuration & Settings
Click the **Settings (Gear)** icon in the GUI to configure:
1. **Camera Sources:** Map `Camera 1` and `Camera 2` to local devices or network streams.
2. **RTSP Setup:** Enter full RTSP URLs and assign a remote IP for instant network connections.
3. **Save vs. Apply:** Click *Save* to store preferences, or *Apply* to immediately switch the live viewer source. *(Note: Input switching clears the current local analysis session).*

---

## ⏪ Rollback & Verification

**Running Tests:**
To verify file decoding, recording boundaries, history persistence, and source switching on your local machine:
`cd ~/Rectile`
`python3 -m pytest -q`

**Restoring a Backup GUI:**
If a GUI patch causes issues, you can easily restore the automatically generated backup:
`cd ~/Rectile`
`mv uav_gui "uav_gui.disabled.$(date +%Y%m%d_%H%M%S)"`
`cp -a "backups/YOUR_BACKUP_DIRECTORY/uav_gui" ./uav_gui`
