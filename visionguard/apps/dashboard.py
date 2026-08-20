import sys
import os
from pathlib import Path

# Ensure project root directory is in sys.path for direct script execution
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import cv2
import cvzone
import time
import math
import tempfile
import pandas as pd
import numpy as np
import streamlit as st

from visionguard.config import settings
from visionguard.core.device import get_device_info
from visionguard.core.camera import CameraSource
from visionguard.core.detector import YOLODetector
from visionguard.core.pipeline import VideoPipeline, FPSMeter
from visionguard.assistive.spatial_analyzer import SpatialAnalyzer, ThreatLevel
from visionguard.assistive.priority_engine import ThreatPriorityEngine
from visionguard.assistive.alert_manager import AlertManager
from visionguard.traffic.vehicle_counter import VehicleCounter
from visionguard.traffic.people_counter import PeopleCounter
from visionguard.traffic.analytics import TrafficAnalytics
from visionguard.drowsiness.detector import DrowsinessDetector, DrowsinessState
from visionguard.services.weather import WeatherService

# Streamlit Page Setup
st.set_page_config(
    page_title="VisionGuard AI Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; color: #4CAF50; font-weight: 700; margin-bottom: 0px; }
    .sub-title { font-size: 1.0rem; color: #B0BEC5; margin-bottom: 20px; }
    .card { background-color: #1E2640; padding: 18px; border-radius: 10px; border-left: 5px solid #4CAF50; margin-bottom: 15px; }
    .metric-value { font-size: 1.8rem; font-weight: bold; color: #00E676; }
    .metric-label { font-size: 0.85rem; color: #90A4AE; }
    .stButton>button { width: 100%; border-radius: 6px; height: 42px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Singleton Cached Resources
@st.cache_resource
def get_cached_detector(model_name: str, conf_thresh: float):
    return YOLODetector(model_name_or_path=model_name, conf_threshold=conf_thresh)

@st.cache_resource
def get_cached_weather_service():
    return WeatherService()

@st.cache_resource
def get_cached_alert_manager():
    return AlertManager()

def main():
    st.markdown('<div class="main-title">🛡️ VisionGuard AI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Intelligent Computer Vision, Assistive Perception & Safety Analytics</div>', unsafe_allow_html=True)

    device_info = get_device_info()
    weather_svc = get_cached_weather_service()
    weather_info = weather_svc.get_weather_summary()

    # Sidebar Navigation & Settings
    st.sidebar.image("https://img.icons8.com/color/96/000000/security-checked.png", width=64)
    st.sidebar.title("Navigation")
    
    app_mode = st.sidebar.radio(
        "Select Platform Module:",
        [
            "🏠 Home Overview",
            "🎥 Object Detection",
            "🚦 Traffic Intelligence",
            "♿ VisionGuard Assistive",
            "🚗 Driver Safety (EAR)",
            "📊 Session Analytics",
            "⚙️ Platform Settings"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Hardware & Environment")
    st.sidebar.info(f"**Device:** `{device_info['device_name']}`\n\n**PyTorch:** `{device_info['pytorch_version']}`\n\n**Weather ({weather_info['city']}):** `{weather_info['temp_celsius']}°C`")

    # Selected Model Options
    selected_model = st.sidebar.selectbox("YOLO Model", list(settings.YOLO_MODELS.keys()), index=0)
    conf_slider = st.sidebar.slider("Confidence Threshold", 0.10, 0.90, settings.CONFIDENCE_THRESHOLD, 0.05)

    detector = get_cached_detector(settings.YOLO_MODELS[selected_model], conf_slider)
    detector.conf_threshold = conf_slider

    # ==========================================
    # 🏠 HOME OVERVIEW
    # ==========================================
    if app_mode == "🏠 Home Overview":
        st.subheader("Platform Architecture & Modules")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="card"><div class="metric-label">INFERENCE DEVICE</div><div class="metric-value">' + device_info['device'].upper() + '</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="card"><div class="metric-label">ACTIVE MODEL</div><div class="metric-value">' + selected_model.split()[0] + '</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="card"><div class="metric-label">WEATHER TEMP</div><div class="metric-value">' + str(weather_info['temp_celsius']) + '°C</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="card"><div class="metric-label">PLATFORM STATUS</div><div class="metric-value">READY 🟢</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        ### Four Pillars of VisionGuard AI
        1. **🎥 Real-time Object Detection:** High-speed COCO object recognition supporting webcam, video uploads, and streaming.
        2. **🚦 Traffic Intelligence & Analytics:** Vehicle tracking via SORT, ROI zone masking, line-crossing counts, and directional pedestrian metrics.
        3. **♿ VisionGuard Assistive Perception:** Wireless mobile camera streaming, spatial threat zoning (head-level obstacles, vehicle proximity, path blocks), and non-blocking speech notifications.
        4. **🚗 Driver Safety Monitoring:** Facial landmark EAR (Eye Aspect Ratio), PERCLOS, and head posture tilt monitoring for fatigue warning.
        """)

    # ==========================================
    # 🎥 OBJECT DETECTION
    # ==========================================
    elif app_mode == "🎥 Object Detection":
        st.subheader("Real-Time Object Detection Engine")
        source_type = st.radio("Select Input Source:", ["Webcam", "Video File Upload"], horizontal=True)
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if source_type == "Webcam":
            run_detection = st.checkbox("Start Live Detection Feed", value=False)
            if run_detection:
                cam = CameraSource(source=settings.DEFAULT_WEBCAM_INDEX)
                pipeline = VideoPipeline()

                while run_detection and cam.is_connected:
                    t0 = time.time()
                    success, frame = cam.read()
                    if not success or frame is None:
                        st.error("Failed to read webcam frame")
                        break

                    detections = detector.detect(frame)
                    t_inf = time.time() - t0

                    frame = pipeline.draw_detections(frame, detections)
                    fps = pipeline.fps_meter.update(t_inf)
                    frame = pipeline.draw_overlay(frame, fps, t_inf * 1000.0, device_info['device_name'], "Detection")

                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                    metrics_placeholder.caption(f"Active Detections: {len(detections)} | FPS: {fps:.1f} | Latency: {t_inf*1000.0:.1f}ms")

                cam.release()

        else:
            uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                cam = CameraSource(source=temp_path)
                pipeline = VideoPipeline()

                if st.button("Process Video"):
                    while cam.is_connected:
                        t0 = time.time()
                        success, frame = cam.read()
                        if not success or frame is None:
                            break

                        detections = detector.detect(frame)
                        t_inf = time.time() - t0

                        frame = pipeline.draw_detections(frame, detections)
                        fps = pipeline.fps_meter.update(t_inf)
                        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                    cam.release()
                    os.unlink(temp_path)

    # ==========================================
    # 🚦 TRAFFIC INTELLIGENCE
    # ==========================================
    elif app_mode == "🚦 Traffic Intelligence":
        st.subheader("Traffic Analysis & Counting Engine")
        
        traffic_mode = st.radio("Traffic Module:", ["Vehicle Counting Dashboard", "Pedestrian Flow Counter"], horizontal=True)
        frame_placeholder = st.empty()
        
        if traffic_mode == "Vehicle Counting Dashboard":
            uploaded_video = st.file_uploader("Upload Traffic Video (.mp4, .avi)", type=["mp4", "avi"])
            uploaded_mask = st.file_uploader("Upload Optional Mask Image (.png, .jpg)", type=["png", "jpg", "jpeg"])

            if uploaded_video is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
                    tmp_v.write(uploaded_video.read())
                    v_path = tmp_v.name

                mask_img = None
                if uploaded_mask is not None:
                    bytes_data = uploaded_mask.read()
                    np_arr = np.frombuffer(bytes_data, np.uint8)
                    mask_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if st.button("Start Traffic Analysis"):
                    counter = VehicleCounter()
                    cam = CameraSource(source=v_path)

                    col1, col2 = st.columns(2)
                    m1 = col1.empty()
                    m2 = col2.empty()

                    while cam.is_connected:
                        success, frame = cam.read()
                        if not success or frame is None:
                            break

                        detections = detector.detect(frame)
                        processed_frame, total_cnt, class_breakdown = counter.process_frame(frame, detections, mask=mask_img)

                        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                        m1.metric("Total Vehicles Counted", total_cnt)
                        m2.json(class_breakdown)

                    cam.release()
                    os.unlink(v_path)

        else:
            uploaded_video = st.file_uploader("Upload Pedestrian Video", type=["mp4", "avi"])
            if uploaded_video is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
                    tmp_v.write(uploaded_video.read())
                    v_path = tmp_v.name

                if st.button("Start Pedestrian Analysis"):
                    counter = PeopleCounter()
                    cam = CameraSource(source=v_path)
                    
                    c1, c2 = st.columns(2)
                    m1 = c1.empty()
                    m2 = c2.empty()

                    while cam.is_connected:
                        success, frame = cam.read()
                        if not success or frame is None:
                            break

                        detections = detector.detect(frame)
                        processed_frame, up_cnt, down_cnt = counter.process_frame(frame, detections)

                        frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
                        m1.metric("Pedestrians Going Up", up_cnt)
                        m2.metric("Pedestrians Going Down", down_cnt)

                    cam.release()
                    os.unlink(v_path)

    # ==========================================
    # ♿ VISIONGUARD ASSISTIVE
    # ==========================================
    elif app_mode == "♿ VisionGuard Assistive":
        st.subheader("Assistive Perception for Visually Impaired Users")

        ip_url = st.text_input("Mobile IP Camera Stream URL", value=settings.DEFAULT_IP_WEBCAM_URL)
        enable_speech = st.checkbox("Enable Non-Blocking Voice Alerts", value=True)
        enable_beeps = st.checkbox("Enable Auditory Pitch Beeps", value=True)

        alert_mgr = get_cached_alert_manager()
        alert_mgr.voice_enabled = enable_speech
        alert_mgr.beep_enabled = enable_beeps

        if st.checkbox("Start VisionGuard Assistive Feed", value=False):
            spatial_analyzer = SpatialAnalyzer()
            priority_engine = ThreatPriorityEngine()
            
            target_source = ip_url if ip_url else 0
            cam = CameraSource(source=target_source, auto_reconnect=True, max_reconnect_attempts=2)
            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            if not cam.is_connected:
                st.error(f"⚠️ Unable to connect to IP Camera at '{target_source}'. Please verify that IP Webcam app is running on your mobile device and connected to the same Wi-Fi network.")
            else:
                fail_counter = 0
                while cam.is_connected:
                    success, frame = cam.read()
                    if not success or frame is None:
                        fail_counter += 1
                        status_placeholder.warning(f"Connecting to camera stream (attempt {fail_counter}/5)...")
                        time.sleep(0.5)
                        if fail_counter >= 5:
                            status_placeholder.error(f"⚠️ Lost connection to camera stream at '{target_source}'.")
                            break
                        continue

                    fail_counter = 0

                detections = detector.detect(frame)
                hazards = spatial_analyzer.analyze_detections(detections)
                urgent_hazards = priority_engine.process_hazards(hazards)

                # Draw spatial threat visualization line for head-level top zone
                head_zone_y = spatial_analyzer.head_level_zone
                cv2.line(frame, (0, head_zone_y), (frame.shape[1], head_zone_y), (255, 0, 0), 2)
                cv2.putText(frame, "HEAD LEVEL ZONE", (10, head_zone_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                for hazard in hazards:
                    det = hazard.detection
                    cvzone.cornerRect(frame, (det.x1, det.y1, det.width, det.height), colorR=hazard.color_bgr, t=3)
                    cvzone.putTextRect(frame, f"{hazard.threat_label} ({hazard.approx_distance_m}m)", (max(0, det.x1), max(35, det.y1)), scale=1, colorR=hazard.color_bgr)

                # Trigger urgent audio & speech notifications
                for h in urgent_hazards:
                    alert_mgr.play_beep(h.threat_label)
                    alert_mgr.speak(h.spoken_message)

                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                status_placeholder.info(f"Active Hazards Identified: {len(hazards)} | Urgent Speech Alerts Triggered: {len(urgent_hazards)}")

            cam.release()

    # ==========================================
    # 🚗 DRIVER SAFETY (EAR)
    # ==========================================
    elif app_mode == "🚗 Driver Safety (EAR)":
        st.subheader("Driver Drowsiness & Attention Monitoring")
        
        if st.checkbox("Start Drowsiness Monitoring Webcam Feed", value=False):
            drowsiness_detector = DrowsinessDetector()
            cam = CameraSource(source=0)
            alert_mgr = get_cached_alert_manager()

            frame_placeholder = st.empty()
            metric_placeholder = st.empty()

            while cam.is_connected:
                success, frame = cam.read()
                if not success or frame is None:
                    break

                processed_frame, res = drowsiness_detector.process_frame(frame)
                frame_placeholder.image(processed_frame, channels="BGR", use_container_width=True)

                metric_placeholder.markdown(f"**Driver Status:** `{res.state.value}` | **EAR:** `{res.ear_smoothed:.2f}` | **PERCLOS:** `{res.perclos_pct}%` | **Head Tilt:** `{res.head_tilted}`")

                if res.alert_triggered:
                    alert_mgr.play_beep("CRITICAL")

            cam.release()

    # ==========================================
    # 📊 SESSION ANALYTICS
    # ==========================================
    elif app_mode == "📊 Session Analytics":
        st.subheader("System Performance & Session Metrics")
        
        st.json({
            "Platform Version": settings.VERSION,
            "Hardware Device": device_info['device_name'],
            "PyTorch CUDA": device_info['cuda_available'],
            "Default Confidence": settings.CONFIDENCE_THRESHOLD,
            "Active Authors": settings.AUTHORS
        })

    # ==========================================
    # ⚙️ PLATFORM SETTINGS
    # ==========================================
    elif app_mode == "⚙️ Platform Settings":
        st.subheader("Platform Configuration")
        st.text_input("OpenWeather API Key", value=settings.OPENWEATHER_API_KEY, type="password")
        st.text_input("Default Weather City", value=settings.DEFAULT_CITY)
        st.slider("Assistive Alert Cooldown (sec)", 1.0, 10.0, settings.ALERT_COOLDOWN_SEC)

if __name__ == "__main__":
    main()
