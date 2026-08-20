import time
from typing import List, Dict, Optional
from visionguard.assistive.spatial_analyzer import SpatialHazard, ThreatLevel

class ThreatPriorityEngine:
    """Filters, deduplicates, and prioritizes spatial hazards to prevent alert spam."""

    def __init__(
        self,
        cooldown_sec: float = 3.0,
        min_threat_level: ThreatLevel = ThreatLevel.LOW
    ):
        self.cooldown_sec = cooldown_sec
        self.min_threat_level = min_threat_level
        self.recent_alerts: Dict[str, float] = {}

    def process_hazards(self, hazards: List[SpatialHazard]) -> List[SpatialHazard]:
        """Select top urgent hazards that pass rate-limiting and priority thresholds."""
        curr_time = time.time()
        
        # Clean expired alerts from recent cache
        expired_keys = [k for k, v in self.recent_alerts.items() if curr_time - v > self.cooldown_sec]
        for k in expired_keys:
            del self.recent_alerts[k]

        priority_hazards: List[SpatialHazard] = []

        for hazard in hazards:
            # Skip if below minimum threat level
            if self._level_rank(hazard.threat_level) > self._level_rank(self.min_threat_level):
                continue

            alert_key = f"{hazard.threat_label}_{hazard.zone}"
            
            # Allow critical threats with shorter cooldown
            cooldown = self.cooldown_sec * 0.5 if hazard.threat_level == ThreatLevel.CRITICAL else self.cooldown_sec
            
            if alert_key not in self.recent_alerts or (curr_time - self.recent_alerts[alert_key] > cooldown):
                priority_hazards.append(hazard)
                self.recent_alerts[alert_key] = curr_time

        return priority_hazards

    @staticmethod
    def _level_rank(level: ThreatLevel) -> int:
        ranks = {
            ThreatLevel.CRITICAL: 0,
            ThreatLevel.HIGH: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.LOW: 3,
            ThreatLevel.INFORMATIONAL: 4,
            ThreatLevel.SAFE: 5
        }
        return ranks.get(level, 5)
