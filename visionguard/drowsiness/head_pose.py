import time
from typing import Tuple, Optional
from visionguard.config import settings

class HeadPoseTracker:
    """Monitors head position shifts and downward tilts indicative of drowsiness."""

    def __init__(self, tilt_threshold_px: int = settings.HEAD_TILT_SHIFT_PX):
        self.tilt_threshold_px = tilt_threshold_px
        self.baseline_y: Optional[int] = None
        self.tilt_start_time: Optional[float] = None

    def update(self, face_y: int) -> Tuple[bool, int, float]:
        """
        Update head pose tracker with face bounding box top Y.
        Returns (is_tilted_down, vertical_shift_px, duration_sec).
        """
        if self.baseline_y is None:
            self.baseline_y = face_y

        shift = face_y - self.baseline_y
        is_tilted = shift > self.tilt_threshold_px

        duration = 0.0
        if is_tilted:
            if self.tilt_start_time is None:
                self.tilt_start_time = time.time()
            duration = time.time() - self.tilt_start_time
        else:
            self.tilt_start_time = None

        return is_tilted, shift, round(duration, 1)

    def calibrate(self, current_y: int):
        """Reset baseline calibration."""
        self.baseline_y = current_y
        self.tilt_start_time = None
