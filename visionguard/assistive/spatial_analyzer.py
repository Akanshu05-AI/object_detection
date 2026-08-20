from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional
from visionguard.config import settings
from visionguard.core.detector import DetectionResult

class ThreatLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"
    SAFE = "SAFE"

@dataclass
class SpatialHazard:
    """Evaluated hazard analysis result for an object."""
    detection: DetectionResult
    threat_level: ThreatLevel
    threat_label: str
    zone: str                 # "LEFT", "CENTER", "RIGHT"
    is_head_level: bool
    approx_distance_m: float
    color_bgr: Tuple[int, int, int]
    spoken_message: str

class SpatialAnalyzer:
    """Spatial zone hazard analyzer for assistive vision guidance."""

    VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}
    ANIMAL_CLASSES = {"dog", "cow", "horse", "sheep", "cat"}

    def __init__(
        self,
        frame_width: int = settings.FRAME_WIDTH,
        frame_height: int = settings.FRAME_HEIGHT,
        head_zone_ratio: float = settings.HEAD_LEVEL_ZONE_RATIO,
        proximity_ratio: float = settings.VEHICLE_PROXIMITY_RATIO,
        animal_block_ratio: float = settings.ANIMAL_PATH_BLOCK_RATIO
    ):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.head_level_zone = int(frame_height * head_zone_ratio)
        self.proximity_limit = int(frame_height * proximity_ratio)
        self.animal_limit = int(frame_height * animal_block_ratio)

    def analyze_detections(self, detections: List[DetectionResult]) -> List[SpatialHazard]:
        """Analyze detections and return identified hazards sorted by threat severity."""
        hazards: List[SpatialHazard] = []

        for det in detections:
            hazard = self.analyze_single(det)
            if hazard.threat_level != ThreatLevel.SAFE:
                hazards.append(hazard)

        # Sort hazards so highest priority is first
        severity_order = {
            ThreatLevel.CRITICAL: 0,
            ThreatLevel.HIGH: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.LOW: 3,
            ThreatLevel.INFORMATIONAL: 4,
            ThreatLevel.SAFE: 5
        }
        hazards.sort(key=lambda h: severity_order[h.threat_level])
        return hazards

    def analyze_single(self, det: DetectionResult) -> SpatialHazard:
        """Perform spatial hazard analysis on a single object detection."""
        # 1. Zone determination (LEFT, CENTER, RIGHT)
        if det.center_x < self.frame_w * 0.33:
            zone = "LEFT"
        elif det.center_x > self.frame_w * 0.67:
            zone = "RIGHT"
        else:
            zone = "CENTER"

        # 2. Distance approximation (calibration heuristic: height ratio)
        h_ratio = det.height / float(self.frame_h)
        approx_dist = round(max(0.5, 3.0 * (0.35 / max(0.05, h_ratio))), 1)

        is_head_level = (det.y1 < self.head_level_zone) and (det.height > self.frame_h * 0.20)
        
        threat_level = ThreatLevel.SAFE
        threat_label = "SAFE"
        color_bgr = (0, 255, 0)
        spoken_msg = f"{det.cls_name} detected"

        # Rule 1: CRITICAL Head-Level Obstacle
        if is_head_level:
            threat_level = ThreatLevel.CRITICAL
            threat_label = "CRITICAL: HEAD-LEVEL"
            color_bgr = (0, 0, 255) # Red
            spoken_msg = f"Head-level hazard ahead in {zone.lower()} zone, watch out!"

        # Rule 2: HIGH Vehicle Proximity
        elif det.cls_name in self.VEHICLE_CLASSES and det.height > self.proximity_limit:
            threat_level = ThreatLevel.HIGH
            threat_label = "HIGH: VEHICLE"
            color_bgr = (0, 165, 255) # Orange
            spoken_msg = f"Close vehicle in {zone.lower()} zone, {approx_dist} meters."

        # Rule 3: Animal / Path Blocked (Indian context)
        elif det.cls_name in self.ANIMAL_CLASSES and det.height > self.animal_limit:
            threat_level = ThreatLevel.MEDIUM
            threat_label = f"PATH BLOCKED: {det.cls_name.upper()}"
            color_bgr = (255, 0, 255) # Purple
            spoken_msg = f"{det.cls_name.capitalize()} blocking {zone.lower()} path."

        # Rule 4: Pedestrians / Obstacles in Center corridor
        elif zone == "CENTER" and det.height > self.frame_h * 0.25:
            threat_level = ThreatLevel.LOW
            threat_label = f"OBSTACLE: {det.cls_name.upper()}"
            color_bgr = (0, 255, 255) # Yellow
            spoken_msg = f"{det.cls_name.capitalize()} in center path."

        else:
            threat_level = ThreatLevel.INFORMATIONAL
            threat_label = f"INFO: {det.cls_name.upper()}"
            color_bgr = (200, 200, 200) # Gray
            spoken_msg = f"{det.cls_name} on {zone.lower()}"

        return SpatialHazard(
            detection=det,
            threat_level=threat_level,
            threat_label=threat_label,
            zone=zone,
            is_head_level=is_head_level,
            approx_distance_m=approx_dist,
            color_bgr=color_bgr,
            spoken_message=spoken_msg
        )
