# 🎨 Air Canvas

Air Canvas is a real-time, touchless drawing application that allows you to paint in the air using hand gestures captured via your webcam. Utilizing a robust **Flask + Socket.IO** Python backend, the app pairs advanced computer vision powered by **MediaPipe Tasks API (>= 0.10)** with a responsive, high-performance HTML5 Canvas frontend.

---

## ✨ Features

* **Real-Time Tracking:** Low-latency tracking powered by modern MediaPipe Tasks API.
* **Intuitive Gesture Control:** Control every action seamlessly with predefined hand shapes.
* **Polished Frontend:** Futuristic neon-glow canvas strokes with a customizable sidebar UI.
* **State Management:** Built-in multi-step canvas **Undo (↩)** and **Clear (🗑)** operations.
* **Export Creation:** Save your masterpiece locally as a high-quality `.png` at any time.
* **Dual-View:** Toggle between viewing your raw camera feed with an overlaid tracking skeleton, or drawing on a clean dark background.

---

## 🖐️ Gesture Reference Guide

| Gesture | Action | Description |
| :---: | :--- | :--- |
| ☝️ | **Draw** | Raise *only* your index finger to begin drawing on the canvas. |
| ✌️ | **Hover** | Raise both Index & Middle fingers to navigate without drawing. |
| 🤙 | **Next Color** | Spread your index and middle fingers wide while in Hover mode to cycle colors. |
| ✊ | **Erase** | Close your hand into a full fist to erase pixels locally. |
| 🖐️ | **Clear All** | Extend all 5 fingers fully to wipe the entire canvas. |
| 👍 | **Undo** | Raise *only* your thumb to roll back your last stroke. |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Freddyz-coder/air-canvas.git](https://github.com/Freddyz-coder/air-canvas.git)
cd air-canvas

Make sure you have python 3.8+
pip install opencv-python mediapipe flask flask-socketio

Run the server on cmd
cd download
python server.py
