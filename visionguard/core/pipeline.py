import cv2
import cvzone
import math
import time
from typing import List, Tuple, Optional, Any
from visionguard.core.detector import DetectionResult

class FPSMeter:
    """Accurate FPS and latency calculation tracker."""
    
    def __init__(self, smoothing_factor: float = 0.9):
        self.smoothing_factor = smoothing_factor
        self.prev_time = time.time()
        self.current_fps = 0.0
        self.inference_time_ms = 0.0

    def update(self, inference_duration_sec: float = 0.0) -> float:
        curr_time = time.time()
        delta = curr_time - self.prev_time
        self.prev_time = curr_time
        
        instant_fps = 1.0 / delta if delta > 0 else 0.0
        if self.current_fps == 0.0:
            self.current_fps = instant_fps
        else:
            self.current_fps = (self.smoothing_factor * self.current_fps) + ((1 - self.smoothing_factor) * instant_fps)
            
        self.inference_time_ms = inference_duration_sec * 1000.0
        return self.current_fps

class VideoPipeline:
    """High-performance real-time video processing pipeline."""
    
    def __init__(self, frame_skip: int = 0):
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.fps_meter = FPSMeter()

    def draw_detections(
        self,
        frame: Any,
        detections: List[DetectionResult],
        color: Tuple[int, int, int] = (255, 0, 255),
        thickness: int = 2
    ) -> Any:
        """Render detection bounding boxes and labels cleanly onto frame."""
        for det in detections:
            w, h = det.width, det.height
            cvzone.cornerRect(
                frame,
                (det.x1, det.y1, w, h),
                l=9,
                rt=thickness,
                colorR=color,
                colorC=(0, 255, 0)
            )
            label = f"{det.cls_name} {det.conf:.2f}"
            cvzone.putTextRect(
                frame,
                label,
                (max(0, det.x1), max(35, det.y1)),
                scale=1,
                thickness=1,
                offset=3,
                colorR=color
            )
        return frame

    def draw_overlay(
        self,
        frame: Any,
        fps: float,
        inference_ms: float = 0.0,
        device_name: str = "CPU",
        active_mode: str = "Object Detection"
    ) -> Any:
        """Draw modern performance metrics overlay bar onto frame."""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
        
        text_fps = f"FPS: {fps:.1f} | Latency: {inference_ms:.1f}ms | Device: {device_name} | Mode: {active_mode}"
        cv2.putText(
            frame,
            text_fps,
            (15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 200),
            2,
            cv2.LINE_AA
        )
        return frame
