import sys
import os
from pathlib import Path

# Ensure project root directory is in sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import time
import signal
import threading
import logging
from flask import Flask, Response, render_template_string

from visionguard.config import settings
from visionguard.core.camera import CameraSource
from visionguard.core.detector import YOLODetector
from visionguard.core.pipeline import VideoPipeline

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global streaming variables
camera: CameraSource = None
detector: YOLODetector = None
current_frame = None
is_running = True

def signal_handler(sig, frame):
    global is_running, camera
    logger.info("Exiting Flask server gracefully...")
    is_running = False
    if camera is not None:
        camera.release()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def detection_loop():
    global current_frame, is_running, camera, detector
    camera = CameraSource(source=settings.DEFAULT_WEBCAM_INDEX)
    detector = YOLODetector()
    pipeline = VideoPipeline()

    while is_running and camera.is_connected:
        t0 = time.time()
        success, frame = camera.read()
        if not success or frame is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(frame)
        t_inf = time.time() - t0

        frame = pipeline.draw_detections(frame, detections)
        fps = pipeline.fps_meter.update(t_inf)
        frame = pipeline.draw_overlay(frame, fps, t_inf * 1000.0, "Flask Server", "Live Stream")

        current_frame = frame.copy()

def generate_mjpeg():
    global current_frame
    while is_running:
        if current_frame is None:
            time.sleep(0.01)
            continue
        ret, buffer = cv2.imencode('.jpg', current_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

HTML_INDEX = """
<!DOCTYPE html>
<html>
<head>
    <title>VisionGuard AI Live Stream</title>
    <style>
        body { background-color: #0E1117; color: #FFFFFF; font-family: Arial, sans-serif; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #4CAF50; }
        .stream-container { margin-top: 20px; }
        img { border: 4px solid #1E2640; border-radius: 8px; max-width: 90%; height: auto; }
    </style>
</head>
<body>
    <h1>🛡️ VisionGuard AI — Live Detection Server</h1>
    <div class="stream-container">
        <img src="/video" alt="Live Detection Feed">
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_INDEX)

@app.route('/video')
def video():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_server(port: int = 5000):
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    run_server()
