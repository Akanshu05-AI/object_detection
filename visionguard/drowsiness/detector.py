import cv2
import time
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any

from visionguard.config import settings
from visionguard.drowsiness.ear import eye_aspect_ratio, get_eye_landmarks, EARTracker
from visionguard.drowsiness.head_pose import HeadPoseTracker

logger = logging.getLogger(__name__)

# Try loading dlib
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logger.warning("dlib library not found. Drowsiness detection fallback mode will be active.")

class DrowsinessState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    DROWSY = "DROWSY"
    SEVERE = "SEVERE"

@dataclass
class DrowsinessResult:
    state: DrowsinessState
    ear_raw: float
    ear_smoothed: float
    perclos_pct: float
    head_tilted: bool
    head_shift_px: int
    alert_triggered: bool
    alert_message: str

class DrowsinessDetector:
    """Full pipeline for driver drowsiness & attention monitoring."""

    def __init__(
        self,
        predictor_path: str = str(settings.SHAPE_PREDICTOR_PATH),
        haar_path: str = str(settings.HAAR_CASCADE_PATH),
        ear_thresh: float = settings.EAR_THRESH,
        ear_consec_frames: int = settings.EAR_CONSEC_FRAMES
    ):
        self.ear_thresh = ear_thresh
        self.ear_consec_frames = ear_consec_frames

        self.ear_tracker = EARTracker(ear_thresh=ear_thresh)
        self.head_tracker = HeadPoseTracker()

        self.detector = None
        self.predictor = None
        self.face_cascade = None
        self.is_initialized = False

        self._init_models(predictor_path, haar_path)

        self.closed_frame_counter = 0
        self.drowsy_start_time: Optional[float] = None
        self.last_alert_time = 0.0

    def _init_models(self, predictor_path: str, haar_path: str):
        """Initialize dlib face detector, shape predictor & Haar cascade classifier."""
        if DLIB_AVAILABLE:
            try:
                self.detector = dlib.get_frontal_face_detector()
                if predictor_path and len(predictor_path) > 0:
                    self.predictor = dlib.shape_predictor(predictor_path)
                logger.info("Successfully initialized dlib facial landmark models.")
            except Exception as e:
                logger.error(f"Error loading dlib models: {e}")

        try:
            if haar_path and len(haar_path) > 0:
                self.face_cascade = cv2.CascadeClassifier(haar_path)
        except Exception as e:
            logger.error(f"Error loading Haar cascade: {e}")

        self.is_initialized = (self.detector is not None and self.predictor is not None) or (self.face_cascade is not None)

    def process_frame(self, frame: Any) -> Tuple[Any, DrowsinessResult]:
        """Process video frame and compute EAR, PERCLOS, head tilt & drowsiness status."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        raw_ear = 0.35
        smoothed_ear = 0.35
        perclos_pct = 0.0
        head_tilted = False
        head_shift = 0
        state = DrowsinessState.NORMAL
        alert_triggered = False
        alert_msg = ""

        if not self.is_initialized or not DLIB_AVAILABLE:
            # Render notice if models missing
            cv2.putText(
                frame,
                "Drowsiness Detector: Model File Required",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return frame, DrowsinessResult(
                state=DrowsinessState.NORMAL,
                ear_raw=0.35,
                ear_smoothed=0.35,
                perclos_pct=0.0,
                head_tilted=False,
                head_shift_px=0,
                alert_triggered=False,
                alert_message="Model Missing"
            )

        # Detect faces using dlib
        faces = self.detector(gray)
        if len(faces) > 0:
            face = faces[0]
            # Draw face box
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
            
            # Head pose update
            head_tilted, head_shift, tilt_duration = self.head_tracker.update(face.top())

            # Landmark shape
            shape = self.predictor(gray, face)
            pts = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            left_eye, right_eye = get_eye_landmarks(pts)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            raw_ear = (left_ear + right_ear) / 2.0

            # Draw eye contours
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

            raw_ear, smoothed_ear, perclos_pct = self.ear_tracker.update(raw_ear)

            # Evaluate Drowsiness State
            dynamic_thresh = max(self.ear_thresh, smoothed_ear * 0.85)

            if smoothed_ear < dynamic_thresh:
                self.closed_frame_counter += 1
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = time.time()
            else:
                self.closed_frame_counter = 0
                self.drowsy_start_time = None

            # State transition rules
            if self.closed_frame_counter >= self.ear_consec_frames or perclos_pct > 40.0:
                state = DrowsinessState.SEVERE
                alert_msg = "SEVERE DROWSINESS ALERT! WAKE UP!"
                alert_triggered = True
            elif self.closed_frame_counter >= (self.ear_consec_frames // 2) or head_tilted:
                state = DrowsinessState.DROWSY
                alert_msg = "WARNING: DROWSINESS / HEAD DROOP DETECTED!"
                alert_triggered = True
            elif smoothed_ear < self.ear_thresh:
                state = DrowsinessState.WARNING
                alert_msg = "EYES CLOSING WARNING"

        # Render On-Screen Display Overlay
        color_map = {
            DrowsinessState.NORMAL: (0, 255, 0),
            DrowsinessState.WARNING: (0, 255, 255),
            DrowsinessState.DROWSY: (0, 165, 255),
            DrowsinessState.SEVERE: (0, 0, 255)
        }
        badge_color = color_map[state]

        cv2.putText(frame, f"Driver State: {state.value}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, badge_color, 2)
        cv2.putText(frame, f"EAR: {smoothed_ear:.2f} | PERCLOS: {perclos_pct}%", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if alert_triggered:
            cv2.rectangle(frame, (10, 95), (frame.shape[1] - 10, 140), (0, 0, 200), -1)
            cv2.putText(frame, alert_msg, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, DrowsinessResult(
            state=state,
            ear_raw=round(raw_ear, 2),
            ear_smoothed=round(smoothed_ear, 2),
            perclos_pct=perclos_pct,
            head_tilted=head_tilted,
            head_shift_px=head_shift,
            alert_triggered=alert_triggered,
            alert_message=alert_msg
        )
