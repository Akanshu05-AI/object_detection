import unittest
import numpy as np
from visionguard.tracking.sort_tracker import SortTracker

class TestTrackingSubsystem(unittest.TestCase):

    def test_sort_tracker_initialization(self):
        tracker = SortTracker(max_age=10, min_hits=2, iou_threshold=0.3)
        self.assertEqual(tracker.max_age, 10)
        self.assertEqual(len(tracker.trackers), 0)

    def test_sort_tracker_update(self):
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
        dummy_dets = np.array([[10, 10, 50, 50, 0.9], [100, 100, 150, 150, 0.8]])
        tracks = tracker.update(dummy_dets)
        
        self.assertTrue(isinstance(tracks, np.ndarray))
        if len(tracks) > 0:
            self.assertEqual(tracks.shape[1], 5)

if __name__ == "__main__":
    unittest.main()
