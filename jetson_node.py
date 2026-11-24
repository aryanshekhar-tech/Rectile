import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import argparse
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
parser.add_argument("--source", type=str, default=None, help="Camera index (e.g., 0, 2), RTSP URL, or Path to video file")
args = parser.parse_args()

# Parse the input (Convert to integer if it is a USB camera index)
if args.source is not None:
    video_input = int(args.source) if args.source.isdigit() else args.source
else:
    video_input = None

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
    def __init__(self, source=None):
        self.source = source
        self.latest_frame = None
        self.running = True

        if self.source is not None:
            self.cap = cv2.VideoCapture(self.source)
            if isinstance(self.source, str) and os.path.exists(self.source):
                fps_val = self.cap.get(cv2.CAP_PROP_FPS)
                self.target_fps = fps_val if (fps_val and 5.0 < fps_val <= 120.0) else 60.0
                self.frame_delay = 1.0 / self.target_fps
                self.is_file = True
            else:
                self.frame_delay = 0.001
                self.is_file = False
        else:
            # Tap into the background service's Shared Memory (SHM) socket
            gst_pipeline = (
                "shmsrc socket-path=/run/newspace-imx477/frames.sock is-live=true do-timestamp=true ! "
                "video/x-raw,format=I420,width=1920,height=1080,framerate=60/1 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true max-buffers=1 sync=false"
            )
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
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
# 3. Anchored Lucas-Kanade + Core-Snapped Hybrid Tracker
# ---------------------------------------------------------
def _refine_box_to_target_core(frame_bgr, box):
    """Tightens oversized boxes onto the high-contrast target core so KLT doesn't grab road/grass."""
    H, W = frame_bgr.shape[:2]
    x, y, w, h = [int(v) for v in box]
    x1 = max(0, min(W - 10, x))
    y1 = max(0, min(H - 10, y))
    x2 = max(x1 + 10, min(W, x + w))
    y2 = max(y1 + 10, min(H, y + h))
    bw, bh = x2 - x1, y2 - y1

    if bw < 36 and bh < 26:
        return [x1, y1, bw, bh]

    roi = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    border_vals = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
    bg_med = float(np.median(border_vals))
    diff = np.abs(roi - bg_med)

    thresh_val = max(18.0, float(np.percentile(diff, 72)))
    mask = (diff >= thresh_val).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    ys, xs = np.where(mask > 0)
    if len(xs) < 12:
        return [x1, y1, bw, bh]

    nx1 = max(x1, x1 + int(np.percentile(xs, 5)) - 3)
    ny1 = max(y1, y1 + int(np.percentile(ys, 5)) - 3)
    nx2 = min(x2, x1 + int(np.percentile(xs, 95)) + 4)
    ny2 = min(y2, y1 + int(np.percentile(ys, 95)) + 4)

    nw = max(20, nx2 - nx1)
    nh = max(16, ny2 - ny1)
    if nw >= bw and nh >= bh:
        return [x1, y1, bw, bh]
    return [nx1, ny1, nw, nh]


class AnchoredKLTTracker:
    def __init__(self, frame, box):
        refined = _refine_box_to_target_core(frame, box)
        x, y, w, h = [int(v) for v in refined]
        H, W = frame.shape[:2]

        self.x = max(0, min(W - 10, x))
        self.y = max(0, min(H - 10, y))
        self.w = max(18, min(W - self.x, w))
        self.h = max(14, min(H - self.y, h))

        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template = self.prev_gray[self.y:self.y + self.h, self.x:self.x + self.w].copy()
        self.points = self._extract_points(self.prev_gray, inset_ratio=0.08)
        self.lost_frames = 0
        self.max_lost_frames = 45

    def _extract_points(self, gray, inset_ratio=0.08, motion_mask=None):
        H, W = gray.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        pad_x = max(1, int(self.w * inset_ratio))
        pad_y = max(1, int(self.h * inset_ratio))

        x1 = max(0, self.x + pad_x)
        y1 = max(0, self.y + pad_y)
        x2 = min(W, self.x + self.w - pad_x)
        y2 = min(H, self.y + self.h - pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        mask[y1:y2, x1:x2] = 255
        if motion_mask is not None:
            combined = cv2.bitwise_and(mask, motion_mask)
            if cv2.countNonZero(combined) >= 8:
                mask = combined

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=50,
            qualityLevel=0.003,
            minDistance=2,
            mask=mask
        )
        if pts is None or len(pts) < 4:
            mask.fill(0)
            mask[max(0, self.y):min(H, self.y + self.h),
                 max(0, self.x):min(W, self.x + self.w)] = 255
            pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=50, qualityLevel=0.002, minDistance=2, mask=mask
            )
        return pts

    def _template_fallback(self, gray):
        """Fast local template match rescue when KLT points drop on small/fast targets."""
        if self.template is None or self.template.size == 0:
            return False, 0.0
        H, W = gray.shape[:2]
        th, tw = self.template.shape[:2]
        if tw >= W or th >= H:
            return False, 0.0

        margin = 34
        sx1 = max(0, self.x - margin)
        sy1 = max(0, self.y - margin)
        sx2 = min(W, self.x + self.w + margin)
        sy2 = min(H, self.y + self.h + margin)

        search_win = gray[sy1:sy2, sx1:sx2]
        if search_win.shape[0] <= th or search_win.shape[1] <= tw:
            return False, 0.0

        res = cv2.matchTemplate(search_win, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.46:
            self.x = int(max(0, min(W - self.w, sx1 + max_loc[0])))
            self.y = int(max(0, min(H - self.h, sy1 + max_loc[1])))
            return True, float(max_val)
        return False, float(max_val)

    def update(self, frame):
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        klt_ok = False
        new_valid = np.empty((0, 1, 2), dtype=np.float32)

        if self.points is not None and len(self.points) >= 2:
            lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            new_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.points, None, **lk_params
            )
            if new_pts is not None and status_fwd is None:
                new_pts = None
            if new_pts is not None:
                back_pts, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.prev_gray, new_pts, None, **lk_params
                )
                if back_pts is not None and status_bwd is not None:
                    fb_err = np.linalg.norm(self.points - back_pts, axis=2).ravel()
                    valid = (status_fwd.ravel() == 1) & (status_bwd.ravel() == 1) & (fb_err < 2.8)
                else:
                    valid = status_fwd.ravel() == 1

                old_valid = self.points[valid]
                new_valid = new_pts[valid]

                if len(new_valid) >= 2:
                    shifts = new_valid - old_valid
                    dx = float(np.median(shifts[:, 0, 0]))
                    dy = float(np.median(shifts[:, 0, 1]))
                    self.x = int(max(0, min(W - self.w, self.x + round(dx))))
                    self.y = int(max(0, min(H - self.h, self.y + round(dy))))
                    klt_ok = True

        if not klt_ok:
            tpl_ok, tpl_score = self._template_fallback(gray)
            if not tpl_ok:
                self.lost_frames += 1
                self.prev_gray = gray
                return (self.lost_frames <= self.max_lost_frames), [self.x, self.y, self.w, self.h], 0.30
            self.points = self._extract_points(gray, inset_ratio=0.08)
            self.prev_gray = gray
            self.lost_frames = 0
            return True, [self.x, self.y, self.w, self.h], min(0.85, tpl_score)

        # Keep only points inside updated target box
        pts_flat = new_valid.reshape(-1, 2)
        inside = (
            (pts_flat[:, 0] >= self.x) &
            (pts_flat[:, 0] <= self.x + self.w) &
            (pts_flat[:, 1] >= self.y) &
            (pts_flat[:, 1] <= self.y + self.h)
        )
        kept_pts = pts_flat[inside]
        self.points = kept_pts.reshape(-1, 1, 2) if len(kept_pts) > 0 else np.empty((0, 1, 2), dtype=np.float32)

        # Replenish points when getting sparse
        if len(self.points) < 16:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, np.ones((3, 3), np.uint8), iterations=1)
            more_pts = self._extract_points(gray, inset_ratio=0.08, motion_mask=motion_mask)
            if more_pts is not None:
                if len(self.points) > 0:
                    self.points = np.vstack([self.points, more_pts])
                else:
                    self.points = more_pts

        # Refresh template when KLT hold is strong
        if len(new_valid) >= 8:
            patch = gray[self.y:self.y + self.h, self.x:self.x + self.w]
            if patch.shape == self.template.shape:
                self.template = patch.copy()

        self.prev_gray = gray
        self.lost_frames = 0

        confidence = min(0.95, float(len(new_valid)) / 25.0 + 0.40)
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

cap = VideoSource(video_input)

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
                ccx, ccy = x + w / 2.0, y + h / 2.0
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
# EGO-MOTION COMPENSATED MOVING TARGET + DINO HYBRID SEARCH
# ---------------------------------------------------------
COLOR_WORDS = {"white", "black", "red", "blue", "green", "yellow", "orange", "purple", "pink", "gray", "grey"}


def _is_vegetation_or_terrain(frame_bgr, box, requested_colors):
    """Rejects green grass/bushes and flat terrain patches unless 'green' was explicitly requested."""
    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return True

    inner = frame_bgr[y1:y2, x1:x2]

    # 1. Vegetation check: reject grass tufts/bushes when user did not ask for green
    if "green" not in requested_colors:
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green_ratio = float(np.mean((h_ch >= 34) & (h_ch <= 88) & (s_ch >= 38) & (v_ch >= 35)))
        if green_ratio > 0.45:
            return True

    # 2. Flat terrain check: reject patches with low contrast relative to surrounding ground
    inner_f = inner.astype(np.float32)
    inner_std = float(np.mean(np.std(inner_f, axis=(0, 1))))

    pad_x, pad_y = max(14, w), max(14, h)
    ox1, oy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ox2, oy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
    outer_f = frame_bgr[oy1:oy2, ox1:ox2].astype(np.float32)

    inner_area = (x2 - x1) * (y2 - y1)
    outer_area = (ox2 - ox1) * (oy2 - oy1)
    if outer_area > inner_area:
        inner_mean = np.mean(inner_f, axis=(0, 1))
        ring_mean = (np.sum(outer_f, axis=(0, 1)) - np.sum(inner_f, axis=(0, 1))) / float(outer_area - inner_area)
        color_contrast = float(np.linalg.norm(inner_mean - ring_mean))
        if inner_std < 22.0 and color_contrast < 24.0:
            return True

    return False


def _score_color_match(frame_bgr, box, requested_colors):
    """Verifies candidate pixels match any requested color in the prompt."""
    if not requested_colors:
        return 1.0

    H, W = frame_bgr.shape[:2]
    x, y, w, h = box
    px, py = int(w * 0.10), int(h * 0.10)
    x1, y1 = max(0, x + px), max(0, y + py)
    x2, y2 = min(W, x + w - px), min(H, y + h - py)
    if x2 <= x1 or y2 <= y1:
        return 1.0

    roi = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    color = requested_colors[0]
    if color == "white":
        ratio = float(np.mean((v_ch > 155) & (s_ch < 75)))
    elif color == "black":
        ratio = float(np.mean(v_ch < 85))
    elif color == "red":
        ratio = float(np.mean(((h_ch <= 10) | (h_ch >= 170)) & (s_ch > 50) & (v_ch > 50)))
    elif color == "orange":
        ratio = float(np.mean((h_ch > 10) & (h_ch <= 24) & (s_ch > 60) & (v_ch > 60)))
    elif color == "yellow":
        ratio = float(np.mean((h_ch > 24) & (h_ch <= 36) & (s_ch > 50) & (v_ch > 60)))
    elif color == "green":
        ratio = float(np.mean((h_ch > 36) & (h_ch <= 86) & (s_ch > 45) & (v_ch > 45)))
    elif color == "blue":
        ratio = float(np.mean((h_ch > 86) & (h_ch <= 135) & (s_ch > 50) & (v_ch > 45)))
    else:
        return 1.0

    if ratio < 0.08:
        return 0.35
    if ratio < 0.18:
        return 0.78
    return 1.15


def _find_moving_vehicle_blobs(prev_bgr, curr_bgr, requested_colors):
    """Cancels camera ego-motion via affine RANSAC and isolates independently moving vehicles (<15ms)."""
    if prev_bgr is None or curr_bgr is None:
        return []

    H, W = curr_bgr.shape[:2]
    sw, sh = 960, 540
    g1 = cv2.resize(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)
    g2 = cv2.resize(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY), (sw, sh), interpolation=cv2.INTER_AREA)

    pts1 = cv2.goodFeaturesToTrack(g1, maxCorners=160, qualityLevel=0.01, minDistance=20)
    if pts1 is None or len(pts1) < 12:
        return []

    pts2, status, _ = cv2.calcOpticalFlowPyrLK(
        g1, g2, pts1, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    if pts2 is None or status is None:
        return []

    ok = status.ravel() == 1
    p1 = pts1[ok].reshape(-1, 2)
    p2 = pts2[ok].reshape(-1, 2)
    if len(p1) < 10:
        return []

    M, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return []

    g1_aligned = cv2.warpAffine(g1, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid_mask = cv2.warpAffine(np.full((sh, sw), 255, dtype=np.uint8), M, (sw, sh), flags=cv2.INTER_NEAREST)
    valid_mask = cv2.erode(valid_mask, np.ones((9, 9), np.uint8), iterations=1)

    diff = cv2.absdiff(g1_aligned, g2)
    diff = cv2.bitwise_and(diff, valid_mask)
    diff_blur = cv2.GaussianBlur(diff, (3, 3), 0)

    _, motion_bin = cv2.threshold(diff_blur, 18, 255, cv2.THRESH_BINARY)
    motion_bin = cv2.morphologyEx(motion_bin, cv2.MORPH_CLOSE, np.ones((4, 4), np.uint8))

    contours, _ = cv2.findContours(motion_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moving_cands = []

    for c in contours:
        area_s = cv2.contourArea(c)
        if area_s < 9 or area_s > 6000:
            continue
        xs, ys, ws, hs = cv2.boundingRect(c)
        if ws < 4 or hs < 3 or ws > 140 or hs > 100:
            continue
        aspect = float(ws) / float(hs)
        if aspect < 0.35 or aspect > 4.2:
            continue

        patch_diff = diff_blur[ys:ys + hs, xs:xs + ws]
        peak_motion = float(np.max(patch_diff)) if patch_diff.size > 0 else 0.0
        if peak_motion < 22.0:
            continue

        # Map back to 1920x1080 with small padding so vehicle is cleanly framed
        pad = 6
        x = max(0, xs * 2 - pad)
        y = max(0, ys * 2 - pad)
        w = min(W - x, ws * 2 + pad * 2)
        h = min(H - y, hs * 2 + pad * 2)
        box = [int(x), int(y), int(w), int(h)]

        if _is_vegetation_or_terrain(curr_bgr, box, requested_colors):
            continue

        color_mult = _score_color_match(curr_bgr, box, requested_colors)
        if requested_colors and color_mult < 0.6:
            continue

        score = min(0.96, (0.45 + min(0.45, peak_motion / 120.0)) * color_mult)
        moving_cands.append((score, box))

    moving_cands.sort(key=lambda item: item[0], reverse=True)
    return moving_cands


def run_hybrid_search(frame_bgr, prompt, prev_frame_bgr=None):
    prompt_clean = prompt.lower().strip()
    if prompt_clean.startswith("letter "):
        prompt_clean = prompt_clean.replace("letter ", "").strip()

    is_text = len(prompt_clean) == 1

    if is_text:
        box, score = find_far_alphabet_mathematically(frame_bgr, prompt_clean)
        if box:
            return box, score
        return None, 0.0

    words = prompt_clean.replace(".", "").split()
    requested_colors = [w for w in words if w in COLOR_WORDS]
    noun_words = [w for w in words if w not in COLOR_WORDS]
    object_query = " ".join(noun_words).strip() if noun_words else prompt_clean

    vehicle_words = {"car", "vehicle", "truck", "van", "bus", "suv", "bike", "motorcycle"}
    is_vehicle_prompt = bool(set(words) & vehicle_words)

    # 1. Fast Ego-Motion Moving Vehicle Detection (<15ms, locks moving cars immediately)
    if is_vehicle_prompt and prev_frame_bgr is not None:
        moving_cands = _find_moving_vehicle_blobs(prev_frame_bgr, frame_bgr, requested_colors)
        if moving_cands:
            best_score, best_box = moving_cands[0]
            print(f"[MOTION LOCK] Moving '{prompt}' locked: box={best_box}, score={best_score:.2f}")
            return best_box, best_score

    # 2. Grounding DINO Fallback (stricter 0.36 gate for vehicles so 0.30 grass/road never false-locks)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    H, W = frame_rgb.shape[:2]
    dino_prompt = object_query if object_query.endswith(".") else f"{object_query} ."
    dino_min_thresh = 0.36 if is_vehicle_prompt else 0.30

    def _infer_dino(image_patch, thresh=0.30, offset_x=0, offset_y=0):
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
                    aspect = float(w) / max(1.0, float(h))
                    if 10 <= w <= 800 and 10 <= h <= 800 and 0.25 <= aspect <= 4.5:
                        box = [int(x1 + offset_x), int(y1 + offset_y), int(w), int(h)]
                        if _is_vegetation_or_terrain(frame_bgr, box, requested_colors):
                            continue
                        adj = float(s) * _score_color_match(frame_bgr, box, requested_colors)
                        if adj >= thresh:
                            cands.append((adj, box))
            return cands
        except Exception as e:
            print(f"[JETSON DINO ERROR] {e}")
            return []

    crop_size = 960
    cx1, cy1 = int(W / 2 - crop_size / 2), int(H / 2 - crop_size / 2)
    center_crop = frame_rgb[cy1:cy1 + crop_size, cx1:cx1 + crop_size]

    cands_s1 = _infer_dino(center_crop, thresh=dino_min_thresh, offset_x=cx1, offset_y=cy1)
    if cands_s1:
        cands_s1.sort(key=lambda item: item[0], reverse=True)
        return cands_s1[0][1], min(0.99, cands_s1[0][0])

    cands_s2 = _infer_dino(frame_rgb, thresh=dino_min_thresh, offset_x=0, offset_y=0)
    if cands_s2:
        cands_s2.sort(key=lambda item: item[0], reverse=True)
        return cands_s2[0][1], min(0.99, cands_s2[0][0])

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
                    frame_1 = shared_state["latest_frame"]
                    id_1 = shared_state["frame_id"]

                # Sample ~6 frames (~100ms) for ego-motion compensated moving target detection
                frame_2 = frame_1
                t_wait = time.perf_counter()
                while (time.perf_counter() - t_wait) < 0.14:
                    with sync_lock:
                        cur_id = shared_state["frame_id"]
                        cur_f = shared_state["latest_frame"]
                    if cur_id >= id_1 + 6 and cur_f is not None:
                        frame_2 = cur_f
                        break
                    time.sleep(0.005)

                if frame_2 is not None:
                    box, score = run_hybrid_search(
                        frame_2,
                        cmd_data.get("prompt", ""),
                        prev_frame_bgr=frame_1 if frame_2 is not frame_1 else None
                    )
                    if box:
                        tracker = AnchoredKLTTracker(frame_2, box)
                        with sync_lock:
                            shared_state["state"] = "TRACKING"
                            shared_state["box"] = [tracker.x, tracker.y, tracker.w, tracker.h]
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
                extracted_mode = msg["mode"]

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
        cv2.drawMarker(annotated, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

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
