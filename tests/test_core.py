import unittest
import numpy as np
from visionguard.core.device import get_device, get_device_info
from visionguard.core.detector import DetectionResult

class TestCoreSubsystem(unittest.TestCase):

    def test_device_info(self):
        device = get_device()
        self.assertIn(device, ["cuda", "cpu"])
        
        info = get_device_info()
        self.assertIn("device", info)
        self.assertIn("pytorch_version", info)

    def test_detection_result_dataclass(self):
        det = DetectionResult(
            x1=10, y1=20, x2=110, y2=120, conf=0.85, cls_id=0, cls_name="person"
        )
        self.assertEqual(det.width, 100)
        self.assertEqual(det.height, 100)
        self.assertEqual(det.center_x, 60)
        self.assertEqual(det.center_y, 70)
        self.assertEqual(det.to_sort_box(), [10.0, 20.0, 110.0, 120.0, 0.85])

if __name__ == "__main__":
    unittest.main()
