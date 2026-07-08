"""
Air Canvas — Python Backend (Flask + SocketIO)
Compatible with mediapipe >= 0.10 (Tasks API)

Usage:
    pip install opencv-python mediapipe flask flask-socketio
    python server.py
Then open: http://localhost:5000
"""

import cv2
import mediapipe as mp
import numpy as np
import base64
import time
import threading
import os
import urllib.request
from flask import Flask
from flask_socketio import SocketIO

# ── Auto-download hand landmarker model ──────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

def download_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 10_000:
        print(f"✅ Model found: {MODEL_PATH}")
        return
    print("⬇️  Downloading hand landmarker model (~6 MB)...")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(MODEL_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
        print(f"✅ Model downloaded: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("   Manual download:")
        print(f"   {MODEL_URL}")
        print(f"   Save as '{MODEL_PATH}' in this folder, then re-run.")
        raise SystemExit(1)

download_model()

# ── MediaPipe Tasks API setup ─────────────────────────────────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, RunningMode
)

latest_result: HandLandmarkerResult = None
result_lock = threading.Lock()

def result_callback(result: HandLandmarkerResult, output_image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result

options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6,
    result_callback=result_callback,
)
landmarker = HandLandmarker.create_from_options(options)

# ── Hand connections for drawing skeleton ─────────────────────────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm
]

TIP_IDS = [8, 12, 16, 20]
MCP_IDS = [6, 10, 14, 18]

def fingers_up(lm):
    up = []
    up.append(1 if lm[4].x < lm[3].x else 0)
    for tip, mcp in zip(TIP_IDS, MCP_IDS):
        up.append(1 if lm[tip].y < lm[mcp].y else 0)
    return up

def finger_spread(lm):
    dx = lm[8].x - lm[12].x
    dy = lm[8].y - lm[12].y
    return (dx**2 + dy**2) ** 0.5

def draw_skeleton(frame, lm_list, w, h):
    pts = [(int(l.x * w), int(l.y * h)) for l in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 120, 220), 2)
    for i, (x, y) in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (x, y), r, (0, 220, 255), -1)

# ── Flask + SocketIO ──────────────────────────────────────────────────────────
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

running         = True
last_color_time = 0
COLOR_COOLDOWN  = 0.55

@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@socketio.on("connect")
def on_connect():
    print("🌐 Browser connected")

@socketio.on("disconnect")
def on_disconnect():
    print("🌐 Browser disconnected")

# ── Camera thread ─────────────────────────────────────────────────────────────
def camera_loop():
    global running, last_color_time, latest_result

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ts_ms = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Feed frame to landmarker (async callback fills latest_result)
        ts_ms += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        landmarker.detect_async(mp_img, ts_ms)

        payload = {"mode": "NONE", "x": None, "y": None, "color_advance": False}

        with result_lock:
            result = latest_result

        if result and result.hand_landmarks:
            lm = result.hand_landmarks[0]

            draw_skeleton(frame, lm, w, h)

            fu = fingers_up(lm)
            ix = int(lm[8].x * w)
            iy = int(lm[8].y * h)
            nx = lm[8].x
            ny = lm[8].y

            payload["x"] = nx
            payload["y"] = ny

            if sum(fu) == 5:
                payload["mode"] = "CLEAR"
            elif sum(fu) == 0:
                payload["mode"] = "ERASE"
            elif fu == [1, 0, 0, 0, 0]:
                payload["mode"] = "UNDO"
            elif fu[1] == 1 and fu[2] == 1 and fu[3] == 0 and fu[4] == 0:
                payload["mode"] = "HOVER"
                spread = finger_spread(lm)
                now    = time.time()
                if spread > 0.08 and (now - last_color_time) > COLOR_COOLDOWN:
                    payload["color_advance"] = True
                    last_color_time = now
                mx = int(lm[12].x * w)
                my = int(lm[12].y * h)
                cv2.line(frame, (ix, iy), (mx, my), (255, 220, 0), 2)
            elif fu[1] == 1 and fu[2] == 0:
                payload["mode"] = "DRAW"
                cv2.circle(frame, (ix, iy), 10, (0, 255, 120), -1)
            else:
                payload["mode"] = "HOVER"

        _, buf   = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        b64      = base64.b64encode(buf).decode("utf-8")
        payload["frame"] = b64

        socketio.emit("frame", payload)
        time.sleep(1 / 30)

    cap.release()
    landmarker.close()

if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()
    print("\n🎨 Air Canvas server running → http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False)
