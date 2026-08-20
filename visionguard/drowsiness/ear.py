import numpy as np
from scipy.spatial import distance
from typing import List, Tuple, Dict, Any
from visionguard.config import settings

# 68-point facial landmark indices for eyes
LEFT_EYE_INDICES = list(range(42, 48))
RIGHT_EYE_INDICES = list(range(36, 42))

def eye_aspect_ratio(eye_pts: List[Tuple[int, int]]) -> float:
    """
    Calculate Eye Aspect Ratio (EAR) from 6 2D eye landmark points.
    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    if len(eye_pts) < 6:
        return 0.0

    a = distance.euclidean(eye_pts[1], eye_pts[5])
    b = distance.euclidean(eye_pts[2], eye_pts[4])
    c = distance.euclidean(eye_pts[0], eye_pts[3])

    if c == 0.0:
        return 0.0

    return (a + b) / (2.0 * c)

def get_eye_landmarks(shape_68_pts: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Extract left and right eye coordinate lists from 68 face landmarks."""
    left_eye = [shape_68_pts[i] for i in LEFT_EYE_INDICES]
    right_eye = [shape_68_pts[i] for i in RIGHT_EYE_INDICES]
    return left_eye, right_eye

class EARTracker:
    """Temporal EAR tracker and PERCLOS calculator."""

    def __init__(
        self,
        ear_thresh: float = settings.EAR_THRESH,
        smoothing_frames: int = 10,
        perclos_window_size: int = settings.PERCLOS_WINDOW_FRAMES
    ):
        self.ear_thresh = ear_thresh
        self.smoothing_frames = smoothing_frames
        self.perclos_window_size = perclos_window_size
        
        self.ear_history: List[float] = []
        self.perclos_history: List[bool] = []

    def update(self, raw_ear: float) -> Tuple[float, float, float]:
        """
        Update tracker with frame EAR.
        Returns (raw_ear, smoothed_ear, perclos_percentage).
        """
        self.ear_history.append(raw_ear)
        if len(self.ear_history) > self.smoothing_frames:
            self.ear_history.pop(0)

        smoothed_ear = float(np.mean(self.ear_history))
        
        # PERCLOS tracking (is eye closed?)
        is_closed = smoothed_ear < self.ear_thresh
        self.perclos_history.append(is_closed)
        if len(self.perclos_history) > self.perclos_window_size:
            self.perclos_history.pop(0)

        closed_count = sum(1 for c in self.perclos_history if c)
        perclos_pct = round((closed_count / float(len(self.perclos_history))) * 100.0, 1)

        return raw_ear, smoothed_ear, perclos_pct

    def reset(self):
        self.ear_history.clear()
        self.perclos_history.clear()
