import argparse
import os
import queue
import threading
import time
from collections import deque
import cv2
import zmq
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import easyocr

parser = argparse.ArgumentParser(description="Rectile Guidance Engine - Anchored KLT Tracker")
parser.add_argument("--video", type=str, default=None, help="Path to video file")
args = parser.parse_args()

print("[RECTILE ENGINE] Initializing Anchored Optical-Flow Guidance Engine...")

# ---------------------------------------------------------
# 1. 1-Second Sliding Window FPS Meter
# ---------------------------------------------------------
class SlidingWindowFPSMeter:
    def __init__(self, window_sec=1.0):
        self.window_sec = window_sec
        self.timestamps = deque()
        self._lock = threading.Lock()

    def tick(self) -> float:
        now = time.perf_counter()
        with self._lock:
            self.timestamps.append(now)
            while self.timestamps and (now - self.timestamps[0]) > self.window_sec:
                self.timestamps.popleft()
            if len(self.timestamps) >= 2:
                elapsed = self.timestamps[-1] - self.timestamps[0]
                if elapsed > 0.01:
                    return round((len(self.timestamps) - 1) / elapsed, 1)
            return float(len(self.timestamps))

    def get_fps(self) -> float:
        now = time.perf_counter()
        with self._lock:
            while self.timestamps and (now - self.timestamps[0]) > self.window_sec:
                self.timestamps.popleft()
            if len(self.timestamps) >= 2:
                elapsed = self.timestamps[-1] - self.timestamps[0]
                if elapsed > 0.01:
                    return round((len(self.timestamps) - 1) / elapsed, 1)
            return 0.0


# ---------------------------------------------------------
# 2. Paced Video Source
# ---------------------------------------------------------
class VideoSource:
    def __init__(self, video_path=None):
        self.video_path = video_path
        self.latest_frame = None
        self.running = True

        if self.video_path and os.path.exists(self.video_path):
            self.cap = cv2.VideoCapture(self.video_path)
            fps_val = self.cap.get(cv2.CAP_PROP_FPS)
            self.target_fps = fps_val if (fps_val and 5.0 < fps_val <= 120.0) else 60.0
            self.frame_delay = 1.0 / self.target_fps
            self.is_file = True
        else:
            gst_pipeline = (
                "shmsrc socket-path=/run/newspace-imx477/frames.sock is-live=true do-timestamp=true ! "
                "video/x-raw,format=I420,width=1920,height=1080,framerate=60/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(
                    "shmsrc socket-path=/run/newspace-imx477/frames.sock ! "
                    "video/x-raw,format=I420,width=1920,height=1080,framerate=60/1 ! "
                    "videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false",
                    cv2.CAP_GSTREAMER
                )
            self.frame_delay = 0.001
            self.is_file = False

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            t0 = time.perf_counter()
            ret, frame = self.cap.read()

            if not ret or frame is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.002)
                continue

            h, w = frame.shape[:2]
            if w != 1920 or h != 1080:
                frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

            self.latest_frame = frame

            if self.is_file:
                elapsed = time.perf_counter() - t0
                sleep_needed = self.frame_delay - elapsed
                if sleep_needed > 0:
                    time.sleep(sleep_needed)

    def read(self):
        if self.latest_frame is None:
            return False, None
        return True, self.latest_frame

    def release(self):
        self.running = False
        self.cap.release()


# ---------------------------------------------------------
# 3. Anchored Lucas-Kanade + Template Hybrid Tracker
# ---------------------------------------------------------
class AnchoredKLTTracker:
    def __init__(self, frame, box):
        x, y, w, h = [int(v) for v in box]
        H, W = frame.shape[:2]
        
        self.x = max(0, min(W - 10, x))
        self.y = max(0, min(H - 10, y))
        self.w = max(16, min(W - self.x, w))
        self.h = max(16, min(H - self.y, h))
        
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.points = self._extract_points(self.prev_gray)
        self.lost_frames = 0
        self.max_lost_frames = 20

    def _extract_points(self, gray):
        H, W = gray.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        
        # Inset margin to avoid picking up moving background pixels
        pad_x = max(2, int(self.w * 0.10))
        pad_y = max(2, int(self.h * 0.10))
        
        x1 = max(0, self.x + pad_x)
        y1 = max(0, self.y + pad_y)
        x2 = min(W, self.x + self.w - pad_x)
        y2 = min(H, self.y + self.h - pad_y)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=50, 
            qualityLevel=0.01, 
            minDistance=4, 
            mask=mask
        )
        return pts

    def update(self, frame):
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Re-seed if points dropped
        if self.points is None or len(self.points) < 6:
            self.points = self._extract_points(self.prev_gray)
            if self.points is None or len(self.points) < 4:
                self.lost_frames += 1
                return (self.lost_frames <= self.max_lost_frames), [self.x, self.y, self.w, self.h], 0.30

        # Calculate Pyramidal Optical Flow
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        if new_pts is None or status is None:
            self.lost_frames += 1
            return (self.lost_frames <= self.max_lost_frames), [self.x, self.y, self.w, self.h], 0.30

        valid = status.ravel() == 1
        old_valid = self.points[valid]
        new_valid = new_pts[valid]
        
        if len(new_valid) < 4:
            self.lost_frames += 1
            return (self.lost_frames <= self.max_lost_frames), [self.x, self.y, self.w, self.h], 0.30

        # Discard displacement outliers via median filtering
        shifts = new_valid - old_valid
        dx = float(np.median(shifts[:, 0, 0]))
        dy = float(np.median(shifts[:, 0, 1]))
        
        # Anchored Translation: Box moves strictly with the target's physical pixels
        self.x = int(max(0, min(W - self.w, self.x + round(dx))))
        self.y = int(max(0, min(H - self.h, self.y + round(dy))))
        
        self.points = new_valid.reshape(-1, 1, 2)
        self.prev_gray = gray
        self.lost_frames = 0
        
        # Replenish points if getting sparse
        if len(self.points) < 18:
            more_pts = self._extract_points(gray)
            if more_pts is not None:
                self.points = np.vstack([self.points, more_pts])
                
        confidence = min(0.95, float(len(new_valid)) / 30.0 + 0.35)
        return True, [self.x, self.y, self.w, self.h], confidence


# ---------------------------------------------------------
# 4. Asynchronous Multi-Threaded JPEG Encoder Pool
# ---------------------------------------------------------
class AsyncJpegPool:
    def __init__(self, num_workers=2, quality=60):
        self.quality = quality
        self.in_queue = queue.Queue(maxsize=2)
        self.out_queue = queue.Queue(maxsize=4)
        self.running = True
        self.workers = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(num_workers)
        ]
        for w in self.workers:
            w.start()

    def _worker(self):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality, int(cv2.IMWRITE_JPEG_OPTIMIZE), 0]
        while self.running:
            try:
                frame, meta = self.in_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            _, buf = cv2.imencode('.jpg', frame, encode_params)
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put((buf.tobytes(), meta))
            self.in_queue.task_done()

    def submit(self, frame, meta):
        if self.in_queue.full():
            try:
                self.in_queue.get_nowait()
            except queue.Empty:
                pass
        self.in_queue.put((frame, meta))

    def get_latest(self):
        try:
            return self.out_queue.get(timeout=0.02)
        except queue.Empty:
            return None, None


# ---------------------------------------------------------
# 5. CUDA Models and ZeroMQ Setup
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[RECTILE ENGINE] Hardware Device: {device}")

print("[RECTILE ENGINE] Loading Grounding DINO...")
model_id = "./weights"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

print("[RECTILE ENGINE] Loading EasyOCR...")
ocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))

context = zmq.Context()
pub_sock = context.socket(zmq.PUB)
pub_sock.setsockopt(zmq.SNDHWM, 2)
pub_sock.bind("tcp://*:5555")

cmd_sock = context.socket(zmq.SUB)
cmd_sock.bind("tcp://*:5556")
cmd_sock.setsockopt_string(zmq.SUBSCRIBE, "")

cap = VideoSource(args.video)

sync_lock = threading.Lock()
shared_state = {
    "latest_frame": None,
    "frame_id": 0,
    "state": "IDLE",
    "prompt": "",
    "box": None,
    "score": 0.0,
    "process_fps": 0.0,
    "optic_mode": "none"
}
ai_command_queue = queue.Queue()
source_fps_meter = SlidingWindowFPSMeter(1.0)


# ---------------------------------------------------------
# FOVEATED TOP-HAT MORPHOLOGY FOR ALPHABETS
# ---------------------------------------------------------
def find_far_alphabet_mathematically(frame_bgr, target_letter):
    H_frame, W_frame = frame_bgr.shape[:2]
    cx_frame, cy_frame = W_frame // 2, H_frame // 2
    
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    
    _, thresh = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_cands = []
    for c in contours:
        area = cv2.contourArea(c)
        if 15 < area < 8000:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / max(1, h)
            hull = cv2.convexHull(c)
            solidity = area / max(1, cv2.contourArea(hull))
            
            if 0.15 < aspect_ratio < 2.5 and 0.15 < solidity < 0.95:
                ccx, ccy = x + w/2.0, y + h/2.0
                dist = np.hypot(ccx - cx_frame, ccy - cy_frame)
                valid_cands.append((dist, x, y, w, h))
                
    valid_cands.sort(key=lambda item: item[0])
    best_box = None
    best_score = 0.0
    
    for cand in valid_cands[:5]:
        _, x, y, w, h = cand
        pad = max(6, int(max(w, h) * 0.4))
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(W_frame, x + w + pad), min(H_frame, y + h + pad)
        
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        roi_up = cv2.resize(roi, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        _, bin_roi = cv2.threshold(roi_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bin_roi) < 127:
            bin_roi = cv2.bitwise_not(bin_roi)
            
        results = ocr_reader.readtext(bin_roi, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        for (bbox, text, prob) in results:
            if target_letter.lower() == text.lower().strip():
                if prob > best_score:
                    best_score = float(prob)
                    best_box = [x, y, w, h]
                    
    if best_box:
        return best_box, best_score
    return None, 0.0


# ---------------------------------------------------------
# 2-STAGE ADAPTIVE ACQUISITION SEARCH
# ---------------------------------------------------------
def run_hybrid_search(frame_bgr, prompt):
    prompt_clean = prompt.lower().strip()
    if prompt_clean.startswith("letter "):
        prompt_clean = prompt_clean.replace("letter ", "").strip()
        
    is_text = len(prompt_clean) == 1

    if is_text:
        box, score = find_far_alphabet_mathematically(frame_bgr, prompt_clean)
        if box:
            return box, score
        return None, 0.0

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]
    dino_prompt = prompt_clean if prompt_clean.endswith(".") else f"{prompt_clean} ."

    def _infer_dino(image_patch, thresh=0.18, offset_x=0, offset_y=0):
        try:
            inputs = processor(images=image_patch, text=dino_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            target_sizes = [image_patch.shape[:2]]
            try:
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=thresh, text_threshold=thresh, target_sizes=target_sizes
                )
            except TypeError:
                results = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=thresh, text_threshold=thresh, target_sizes=target_sizes
                )
            
            cands = []
            if results and len(results[0]["boxes"]) > 0:
                raw_boxes = results[0]["boxes"].detach().cpu().numpy()
                raw_scores = results[0]["scores"].detach().cpu().numpy()
                for b, s in zip(raw_boxes, raw_scores):
                    x1, y1, x2, y2 = b
                    w = max(10, x2 - x1)
                    h = max(10, y2 - y1)
                    if 10 <= w <= 800 and 10 <= h <= 800:
                        cands.append(([int(x1 + offset_x), int(y1 + offset_y), int(w), int(h)], float(s)))
            return cands
        except Exception as e:
            print(f"[JETSON DINO ERROR] {e}")
            return []

    # STAGE 1: Boresight Center Crop Pass
    crop_size = 960
    cx1, cy1 = int(W/2 - crop_size/2), int(H/2 - crop_size/2)
    center_crop = frame_rgb[cy1:cy1 + crop_size, cx1:cx1 + crop_size]

    cands_stage1 = _infer_dino(center_crop, thresh=0.18, offset_x=cx1, offset_y=cy1)
    if len(cands_stage1) > 0:
        cands_stage1.sort(key=lambda x: x[1], reverse=True)
        return cands_stage1[0][0], cands_stage1[0][1]

    # STAGE 2: Full-Frame Fallback Pass
    cands_stage2 = _infer_dino(frame_rgb, thresh=0.15, offset_x=0, offset_y=0)
    if len(cands_stage2) > 0:
        cands_stage2.sort(key=lambda x: x[1], reverse=True)
        return cands_stage2[0][0], cands_stage2[0][1]

    return None, 0.0


# ---------------------------------------------------------
# Universal Optic Filter Applier
# ---------------------------------------------------------
def apply_optic_filter(frame, mode_str):
    if not mode_str:
        return frame
    mode = str(mode_str).lower().strip()
    if not mode or mode in ["none", "rgb", "off", "0", "normal"]:
        return frame
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if "black" in mode or "inverted" in mode:
        return cv2.cvtColor(cv2.bitwise_not(gray), cv2.COLOR_GRAY2BGR)
    elif "white" in mode or "mono" in mode:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif "iron" in mode:
        return cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    elif "rain" in mode or "jet" in mode:
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
    return frame


# ---------------------------------------------------------
# 6. Non-Blocking AI Worker Thread (Synchronized with Fresh Frames)
# ---------------------------------------------------------
def tracker_worker():
    tracker = None
    last_processed_frame_id = -1

    while True:
        try:
            cmd_data = ai_command_queue.get_nowait()
            cmd_type = cmd_data.get("command")
            if cmd_type == "clear":
                tracker = None
                with sync_lock:
                    shared_state["state"] = "IDLE"
                    shared_state["box"] = None
                    shared_state["prompt"] = ""
                    shared_state["process_fps"] = 0.0
            elif cmd_type == "deploy":
                with sync_lock:
                    shared_state["state"] = "AI_SEARCH"
                    shared_state["prompt"] = cmd_data.get("prompt", "")
                    search_frame = shared_state["latest_frame"]

                if search_frame is not None:
                    box, score = run_hybrid_search(search_frame, cmd_data.get("prompt", ""))
                    if box:
                        tracker = AnchoredKLTTracker(search_frame, box)
                        with sync_lock:
                            shared_state["state"] = "TRACKING"
                            shared_state["box"] = box
                            shared_state["score"] = score
                    else:
                        tracker = None
                        with sync_lock:
                            shared_state["state"] = "IDLE"
                            shared_state["box"] = None
                            shared_state["process_fps"] = 0.0
        except queue.Empty:
            pass

        with sync_lock:
            current_state = shared_state["state"]
            frame = shared_state["latest_frame"]
            current_frame_id = shared_state["frame_id"]

        # Only execute optical-flow update if a genuinely NEW frame arrived from the camera
        if current_state == "TRACKING" and tracker and frame is not None:
            if current_frame_id != last_processed_frame_id:
                last_processed_frame_id = current_frame_id
                
                t_start = time.perf_counter()
                ok, box, score = tracker.update(frame)
                dt = time.perf_counter() - t_start
                
                if ok:
                    with sync_lock:
                        shared_state["box"] = box
                        shared_state["score"] = score
                        shared_state["process_fps"] = 1.0 / max(0.001, dt)
                else:
                    tracker = None
                    with sync_lock:
                        shared_state["state"] = "LOST"
                        shared_state["box"] = None
                        shared_state["process_fps"] = 0.0
            else:
                time.sleep(0.002)
        else:
            time.sleep(0.005)

threading.Thread(target=tracker_worker, daemon=True).start()


# ---------------------------------------------------------
# 7. High-Throughput Publisher Thread
# ---------------------------------------------------------
encoder_pool = AsyncJpegPool(num_workers=2, quality=60)

def publisher_worker():
    while True:
        buf, meta = encoder_pool.get_latest()
        if buf is not None:
            pub_sock.send_json(meta, zmq.SNDMORE)
            pub_sock.send(buf)
        else:
            time.sleep(0.001)

threading.Thread(target=publisher_worker, daemon=True).start()


# ---------------------------------------------------------
# 8. Uninterrupted 60 FPS Capture Engine
# ---------------------------------------------------------
print("[RECTILE ENGINE] Stream active on ZMQ Ports 5555/5556...")
frame_counter = 0

while True:
    if cmd_sock.poll(0):
        msg = cmd_sock.recv_json()
        
        extracted_mode = None
        if isinstance(msg, dict):
            if "mode" in msg:
                extracted_mode = msg["mode"]
            elif "filter" in msg:
                extracted_mode = msg["filter"]
            elif "optic" in msg and msg["optic"] != "optic":
                extracted_mode = msg["optic"]
            elif msg.get("command") == "optic":
                extracted_mode = msg.get("mode")

        if extracted_mode is not None:
            with sync_lock:
                shared_state["optic_mode"] = str(extracted_mode)
                print(f"[JETSON OPTICS] Filter Switched To: {extracted_mode}")
        else:
            ai_command_queue.put(msg)

    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.001)
        continue

    frame_counter += 1
    source_fps_meter.tick()

    with sync_lock:
        shared_state["latest_frame"] = frame
        shared_state["frame_id"] = frame_counter
        state = shared_state["state"]
        active_prompt = shared_state["prompt"]
        tracker_box = shared_state["box"]
        score = shared_state["score"]
        process_fps = shared_state["process_fps"]
        active_optic = shared_state["optic_mode"]

    annotated = apply_optic_filter(frame.copy(), active_optic)
    H, W = annotated.shape[:2]

    if state == "IDLE":
        cv2.drawMarker(annotated, (W//2, H//2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    if state == "TRACKING" and tracker_box:
        x, y, w, h = tracker_box
        x1, y1 = max(0, min(W - 1, x)), max(0, min(H - 1, y))
        x2, y2 = max(0, min(W - 1, x + w)), max(0, min(H - 1, y + h))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
        cv2.putText(annotated, f"{active_prompt.upper()} {score:.2f}", (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        pad_w, pad_h = int(w * 0.8), int(h * 0.8)
        crop_x1, crop_y1 = max(0, cx - pad_w), max(0, cy - pad_h)
        crop_x2, crop_y2 = min(W, cx + pad_w), min(H, cy + pad_h)
        
        target_crop = frame[max(0, crop_y1):min(H, crop_y2), max(0, crop_x1):min(W, crop_x2)]
        if target_crop.size > 0:
            inset_w, inset_h = 240, 180
            inset_resized = cv2.resize(target_crop, (inset_w, inset_h), interpolation=cv2.INTER_LINEAR)
            inset_filtered = apply_optic_filter(inset_resized, active_optic)

            ins_x1 = W - inset_w - 20
            ins_y1 = 20
            ins_x2 = ins_x1 + inset_w
            ins_y2 = ins_y1 + inset_h

            annotated[ins_y1:ins_y2, ins_x1:ins_x2] = inset_filtered
            cv2.rectangle(annotated, (ins_x1, ins_y1), (ins_x2, ins_y2), (0, 255, 255), 2)
            cv2.drawMarker(annotated, (ins_x1 + inset_w // 2, ins_y1 + inset_h // 2), (0, 255, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.putText(annotated, "ANCHORED LOCK", (ins_x1 + 6, ins_y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    meta = {
        "state": state,
        "prompt": active_prompt,
        "box": tracker_box,
        "score": round(float(score), 2),
        "process_fps": round(float(process_fps), 1) if state == "TRACKING" else 0.0,
        "message": f"Rectile Guidance Engine: {state}"
    }

    encoder_pool.submit(annotated, meta)
