import time
from typing import Dict, List, Any

class TrafficAnalytics:
    """Traffic Flow and Vehicle Metrics Aggregator."""

    def __init__(self):
        self.start_time = time.time()
        self.history: List[Dict[str, Any]] = []

    def log_snapshot(
        self,
        vehicle_count: int,
        vehicle_breakdown: Dict[str, int],
        people_up: int = 0,
        people_down: int = 0,
        current_fps: float = 30.0
    ):
        """Record traffic snapshot at current timestamp."""
        elapsed = time.time() - self.start_time
        snapshot = {
            "timestamp_sec": round(elapsed, 1),
            "total_vehicles": vehicle_count,
            "cars": vehicle_breakdown.get("car", 0),
            "trucks": vehicle_breakdown.get("truck", 0),
            "buses": vehicle_breakdown.get("bus", 0),
            "motorbikes": vehicle_breakdown.get("motorbike", 0),
            "people_up": people_up,
            "people_down": people_down,
            "fps": round(current_fps, 1)
        }
        self.history.append(snapshot)

    def get_summary(self) -> Dict[str, Any]:
        """Return analytical summary report."""
        if not self.history:
            return {
                "total_session_duration_sec": 0,
                "total_vehicles": 0,
                "total_pedestrians": 0,
                "vehicle_breakdown": {"car": 0, "truck": 0, "bus": 0, "motorbike": 0}
            }

        latest = self.history[-1]
        duration = round(time.time() - self.start_time, 1)
        
        return {
            "total_session_duration_sec": duration,
            "total_vehicles": latest["total_vehicles"],
            "total_pedestrians": latest["people_up"] + latest["people_down"],
            "vehicle_breakdown": {
                "car": latest["cars"],
                "truck": latest["trucks"],
                "bus": latest["buses"],
                "motorbike": latest["motorbikes"]
            },
            "snapshots_recorded": len(self.history)
        }

    def reset(self):
        """Reset analytical logs."""
        self.start_time = time.time()
        self.history.clear()
