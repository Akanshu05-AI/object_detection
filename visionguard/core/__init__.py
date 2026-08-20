"""
VisionGuard Core Subsystem — Device, Camera, Object Detector & Video Pipeline
"""

from .device import get_device, get_device_info
from .camera import CameraSource
from .detector import ObjectDetector, DetectionResult
from .pipeline import VideoPipeline

__all__ = [
    "get_device",
    "get_device_info",
    "CameraSource",
    "ObjectDetector",
    "DetectionResult",
    "VideoPipeline",
]
