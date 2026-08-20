import unittest
from visionguard.drowsiness.ear import eye_aspect_ratio, EARTracker
from visionguard.drowsiness.head_pose import HeadPoseTracker

class TestDrowsinessSubsystem(unittest.TestCase):

    def test_ear_math(self):
        # Open eye coordinates
        open_eye = [(10, 10), (20, 20), (30, 20), (40, 10), (30, 0), (20, 0)]
        ear_val = eye_aspect_ratio(open_eye)
        self.assertGreater(ear_val, 0.0)

    def test_ear_tracker_smoothing_and_perclos(self):
        tracker = EARTracker(ear_thresh=0.20, smoothing_frames=5)
        for _ in range(5):
            _, smoothed, perclos = tracker.update(0.15) # closed eye
            
        self.assertLess(smoothed, 0.20)
        self.assertEqual(perclos, 100.0)

    def test_head_pose_tracker(self):
        tracker = HeadPoseTracker(tilt_threshold_px=10)
        tracker.calibrate(current_y=100)
        
        is_tilted, shift, _ = tracker.update(face_y=120)
        self.assertTrue(is_tilted)
        self.assertEqual(shift, 20)

if __name__ == "__main__":
    unittest.main()
