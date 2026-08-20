import cv2
import cvzone
import numpy as np
from typing import List, Tuple, Dict, Optional, Set, Any
from visionguard.core.detector import DetectionResult
from visionguard.tracking.sort_tracker import SortTracker

class VehicleCounter:
    """Vehicle counting system using SORT tracking & line-crossing detection."""

    VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

    def __init__(
        self,
        line_coords: Optional[Tuple[int, int, int, int]] = None,
        tracker: Optional[SortTracker] = None
    ):
        self.tracker = tracker or SortTracker(max_age=20, min_hits=3, iou_threshold=0.3)
        self.line_coords = line_coords  # (x1, y1, x2, y2)
        
        self.counted_ids: Set[int] = set()
        self.class_counts: Dict[str, int] = {
            "car": 0,
            "truck": 0,
            "bus": 0,
            "motorbike": 0
        }
        self.total_count = 0

    def process_frame(
        self,
        frame: Any,
        detections: List[DetectionResult],
        mask: Optional[Any] = None,
        line_override: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[Any, int, Dict[str, int]]:
        """Process video frame, update tracker, check line crossing, and render UI overlay."""
        h, w = frame.shape[:2]
        line = line_override or self.line_coords or (w // 4, h // 2, (3 * w) // 4, h // 2)

        # 1. Apply ROI mask if provided
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
            if len(mask.shape) == 2 and len(frame.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame_roi = cv2.bitwise_and(frame, mask)
        else:
            frame_roi = frame

        # 2. Filter detections for vehicles
        vehicle_dets = [d for d in detections if d.cls_name in self.VEHICLE_CLASSES and d.conf > 0.30]
        
        if len(vehicle_dets) > 0:
            dets_array = np.array([d.to_sort_box() for d in vehicle_dets])
        else:
            dets_array = np.empty((0, 5))

        # Map bbox to class name lookup
        det_class_map = { (d.x1, d.y1, d.x2, d.y2): d.cls_name for d in vehicle_dets }

        # 3. Update SORT tracker
        tracked_objects = self.tracker.update(dets_array)

        # 4. Draw counting line
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 4)

        # 5. Check line crossing for tracked objects
        for trk in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, trk)
            track_w, track_h = x2 - x1, y2 - y1
            cx, cy = x1 + track_w // 2, y1 + track_h // 2
            
            # Find closest class name
            cls_name = "car"
            for (bx1, by1, bx2, by2), cname in det_class_map.items():
                if abs(x1 - bx1) < 50 and abs(y1 - by1) < 50:
                    cls_name = cname
                    break

            # Draw tracked bounding box
            cvzone.cornerRect(frame, (x1, y1, track_w, track_h), l=9, rt=2, colorR=(255, 0, 0))
            cvzone.putTextRect(frame, f"ID:{obj_id} {cls_name}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Line crossing check (within 15px threshold vertical tolerance)
            min_x, max_x = min(line[0], line[2]), max(line[0], line[2])
            line_y = line[1]

            if min_x <= cx <= max_x and (line_y - 15) <= cy <= (line_y + 15):
                if obj_id not in self.counted_ids:
                    self.counted_ids.add(obj_id)
                    self.total_count += 1
                    if cls_name in self.class_counts:
                        self.class_counts[cls_name] += 1
                    else:
                        self.class_counts[cls_name] = 1
                    # Flash green on line crossing
                    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 5)

        # Render counter badge overlay
        cvzone.putTextRect(
            frame,
            f"Vehicles Counted: {self.total_count}",
            (30, 80),
            scale=1.5,
            thickness=2,
            colorR=(0, 150, 0)
        )

        return frame, self.total_count, self.class_counts

    def reset(self):
        """Reset total vehicle count and tracker."""
        self.counted_ids.clear()
        self.class_counts = {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}
        self.total_count = 0
        self.tracker.reset()
