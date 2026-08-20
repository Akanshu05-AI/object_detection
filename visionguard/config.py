import os
from pathlib import Path
from typing import List, Dict, Any

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Try to load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

class Settings:
    """Centralized configuration for VisionGuard AI Platform."""
    
    PROJECT_NAME: str = "VisionGuard AI"
    VERSION: str = "2.0.0"
    AUTHORS: List[str] = ["Akanshu Goel", "Yash Mahindroo"]
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Model Weights Paths
    DEFAULT_YOLO_MODEL: str = "yolov8n.pt"
    YOLO_MODELS: Dict[str, str] = {
        "YOLOv8 Nano": "yolov8n.pt",
        "YOLOv11 Nano": "yolo11n.pt",
        "YOLOv5 Medium": "yolov5m.pt",
        "YOLOv5m Updated": "yolov5mu.pt",
    }
    
    # Drowsiness Models
    SHAPE_PREDICTOR_PATH: Path = MODELS_DIR / "shape_predictor_68_face_landmarks_1.dat"
    HAAR_CASCADE_PATH: Path = MODELS_DIR / "haarcascade_frontalface_default.xml"
    
    # COCO Class Names (80 standard classes)
    COCO_CLASSES: List[str] = [
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
        "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.40
    IOU_THRESHOLD: float = 0.45
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    
    # Camera Defaults
    DEFAULT_WEBCAM_INDEX: int = 0
    DEFAULT_IP_WEBCAM_URL: str = os.getenv("IP_WEBCAM_URL", "http://192.168.1.5:8080/video")
    
    # VisionGuard Assistive Threat Thresholds
    HEAD_LEVEL_ZONE_RATIO: float = 0.30     # Top 30% of frame is head level zone
    VEHICLE_PROXIMITY_RATIO: float = 0.35   # Bounding box height > 35% frame height = close proximity
    ANIMAL_PATH_BLOCK_RATIO: float = 0.25   # Animal box height > 25% = path blocked
    
    # Audio Alert Settings
    VOICE_ENABLED: bool = True
    BEEP_ENABLED: bool = True
    ALERT_COOLDOWN_SEC: float = 2.0
    SPEECH_RATE: int = 175
    
    # Driver Drowsiness Settings
    EAR_THRESH: float = 0.20
    EAR_CONSEC_FRAMES: int = 15
    CLOSED_EYE_TIME_SEC: float = 1.5
    HEAD_TILT_SHIFT_PX: int = 15
    PERCLOS_WINDOW_FRAMES: int = 100
    
    # Weather Service Settings
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    DEFAULT_CITY: str = os.getenv("WEATHER_CITY", "Delhi")
    WEATHER_REFRESH_SEC: int = 600 # 10 minutes

settings = Settings()
