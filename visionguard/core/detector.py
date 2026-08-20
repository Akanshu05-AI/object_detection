import os
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import logging

warnings.filterwarnings("ignore", category=FutureWarning)

from ultralytics import YOLO
import torch

from visionguard.config import settings
from visionguard.core.device import get_device

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Standardized object detection output structure."""
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    cls_id: int
    cls_name: str
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
        
    @property
    def height(self) -> int:
        return self.y2 - self.y1
        
    @property
    def center_x(self) -> int:
        return self.x1 + self.width // 2
        
    @property
    def center_y(self) -> int:
        return self.y1 + self.height // 2

    def to_sort_box(self) -> List[float]:
        """Format as [x1, y1, x2, y2, conf] for SORT tracker."""
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2), float(self.conf)]

class ObjectDetector:
    """Base class interface for object detection models."""
    def detect(self, frame: Any) -> List[DetectionResult]:
        raise NotImplementedError

class YOLODetector(ObjectDetector):
    """Centralized YOLO Object Detector supporting YOLOv8, YOLOv11, and custom models."""
    
    _model_cache: Dict[str, YOLO] = {}

    def __init__(
        self,
        model_name_or_path: str = settings.DEFAULT_YOLO_MODEL,
        conf_threshold: float = settings.CONFIDENCE_THRESHOLD,
        iou_threshold: float = settings.IOU_THRESHOLD,
        device: Optional[str] = None
    ):
        self.model_path = self._resolve_model_path(model_name_or_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or get_device()
        self.model = self._get_or_load_model()
        self.class_names = self.model.names if hasattr(self.model, "names") and self.model.names else settings.COCO_CLASSES

    def _resolve_model_path(self, path_str: str) -> str:
        """Resolve model path against project directories."""
        # 1. Check direct path
        if os.path.exists(path_str):
            return path_str
            
        # 2. Check MODELS_DIR
        models_path = settings.MODELS_DIR / path_str
        if models_path.exists():
            return str(models_path)

        # 3. Check BASE_DIR
        base_path = settings.BASE_DIR / path_str
        if base_path.exists():
            return str(base_path)

        # Fallback to model string (Ultralytics auto-download)
        return path_str

    def _get_or_load_model(self) -> YOLO:
        """Load YOLO model into memory with singleton caching per path."""
        cache_key = f"{self.model_path}_{self.device}"
        if cache_key in YOLODetector._model_cache:
            logger.info(f"Reusing cached YOLO model: {self.model_path}")
            return YOLODetector._model_cache[cache_key]

        logger.info(f"Loading YOLO model from {self.model_path} onto {self.device}...")
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            YOLODetector._model_cache[cache_key] = model
            logger.info(f"Successfully loaded YOLO model: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading YOLO model from {self.model_path}: {e}")
            raise

    def detect(
        self,
        frame: Any,
        target_classes: Optional[List[int]] = None,
        conf_override: Optional[float] = None
    ) -> List[DetectionResult]:
        """Perform real-time detection on a video frame."""
        conf = conf_override if conf_override is not None else self.conf_threshold
        results = self.model(
            frame,
            conf=conf,
            iou=self.iou_threshold,
            classes=target_classes,
            verbose=False,
            device=self.device
        )
        
        detections: List[DetectionResult] = []
        
        if not results:
            return detections

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, coords)
                conf_score = float(box.conf[0].cpu().item())
                cls_id = int(box.cls[0].cpu().item())
                cls_name = self.class_names[cls_id] if cls_id in self.class_names else str(cls_id)

                detections.append(
                    DetectionResult(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        conf=round(conf_score, 2),
                        cls_id=cls_id,
                        cls_name=cls_name
                    )
                )

        return detections
