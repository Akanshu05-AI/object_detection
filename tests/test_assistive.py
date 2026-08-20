import unittest
from visionguard.core.detector import DetectionResult
from visionguard.assistive.spatial_analyzer import SpatialAnalyzer, ThreatLevel
from visionguard.assistive.priority_engine import ThreatPriorityEngine

class TestAssistiveSubsystem(unittest.TestCase):

    def setUp(self):
        self.analyzer = SpatialAnalyzer(frame_width=1280, frame_height=720)
        self.priority_engine = ThreatPriorityEngine(cooldown_sec=1.0)

    def test_head_level_critical_hazard(self):
        # Object high up in frame (y1 < 216) with large height
        det = DetectionResult(x1=500, y1=50, x2=700, y2=350, conf=0.9, cls_id=0, cls_name="sign")
        hazard = self.analyzer.analyze_single(det)
        
        self.assertEqual(hazard.threat_level, ThreatLevel.CRITICAL)
        self.assertTrue(hazard.is_head_level)

    def test_vehicle_proximity_high_hazard(self):
        # Close car in center corridor
        det = DetectionResult(x1=400, y1=300, x2=800, y2=650, conf=0.9, cls_id=2, cls_name="car")
        hazard = self.analyzer.analyze_single(det)
        
        self.assertEqual(hazard.threat_level, ThreatLevel.HIGH)

    def test_priority_engine_deduplication(self):
        det = DetectionResult(x1=500, y1=50, x2=700, y2=350, conf=0.9, cls_id=0, cls_name="sign")
        hazard = self.analyzer.analyze_single(det)
        
        h1 = self.priority_engine.process_hazards([hazard])
        self.assertEqual(len(h1), 1)
        
        # Second immediate call should be rate-limited
        h2 = self.priority_engine.process_hazards([hazard])
        self.assertEqual(len(h2), 0)

if __name__ == "__main__":
    unittest.main()
