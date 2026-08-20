import cv2
import cvzone
import numpy as np
from typing import List, Tuple, Optional, Set, Any
from visionguard.core.detector import DetectionResult
from visionguard.tracking.sort_tracker import SortTracker

class PeopleCounter:
    """Directional pedestrian counter (Up/Down or In/Out)."""

    def __init__(
        self,
        limits_up: Optional[Tuple[int, int, int, int]] = None,
        limits_down: Optional[Tuple[int, int, int, int]] = None,
        tracker: Optional[SortTracker] = None
    ):
        self.tracker = tracker or SortTracker(max_age=20, min_hits=3, iou_threshold=0.3)
        self.limits_up = limits_up
        self.limits_down = limits_down
        
        self.counted_up_ids: Set[int] = set()
        self.counted_down_ids: Set[int] = set()

    def process_frame(
        self,
        frame: Any,
        detections: List[DetectionResult],
        mask: Optional[Any] = None
    ) -> Tuple[Any, int, int]:
        """Process video frame and count pedestrians crossing upper/lower lines."""
        h, w = frame.shape[:2]
        
        line_up = self.limits_up or (int(w * 0.1), int(h * 0.25), int(w * 0.4), int(h * 0.25))
        line_down = self.limits_down or (int(w * 0.45), int(h * 0.70), int(w * 0.8), int(h * 0.70))

        # Filter for person detections
        person_dets = [d for d in detections if d.cls_name == "person" and d.conf > 0.30]
        
        if len(person_dets) > 0:
            dets_array = np.array([d.to_sort_box() for d in person_dets])
        else:
            dets_array = np.empty((0, 5))

        tracked_objects = self.tracker.update(dets_array)

        # Draw counting lines
        cv2.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 0, 255), 4)
        cv2.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 0, 255), 4)

        for trk in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, trk)
            track_w, track_h = x2 - x1, y2 - y1
            cx, cy = x1 + track_w // 2, y1 + track_h // 2

            cvzone.cornerRect(frame, (x1, y1, track_w, track_h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(frame, f"Person #{obj_id}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Check Upper Line crossing (Going Up / In)
            if line_up[0] <= cx <= line_up[2] and (line_up[1] - 15) <= cy <= (line_up[1] + 15):
                if obj_id not in self.counted_up_ids:
                    self.counted_up_ids.add(obj_id)
                    cv2.line(frame, (line_up[0], line_up[1]), (line_up[2], line_up[3]), (0, 255, 0), 5)

            # Check Lower Line crossing (Going Down / Out)
            if line_down[0] <= cx <= line_down[2] and (line_down[1] - 15) <= cy <= (line_down[1] + 15):
                if obj_id not in self.counted_down_ids:
                    self.counted_down_ids.add(obj_id)
                    cv2.line(frame, (line_down[0], line_down[1]), (line_down[2], line_down[3]), (0, 255, 0), 5)

        # Draw metrics
        cvzone.putTextRect(frame, f"Going Up: {len(self.counted_up_ids)}", (30, 70), scale=1.2, thickness=2, colorR=(139, 195, 75))
        cvzone.putTextRect(frame, f"Going Down: {len(self.counted_down_ids)}", (30, 120), scale=1.2, thickness=2, colorR=(50, 50, 230))

        return frame, len(self.counted_up_ids), len(self.counted_down_ids)

    def reset(self):
        """Reset pedestrian counts."""
        self.counted_up_ids.clear()
        self.counted_down_ids.clear()
        self.tracker.reset()
