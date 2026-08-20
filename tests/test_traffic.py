import unittest
from visionguard.traffic.analytics import TrafficAnalytics
from visionguard.traffic.vehicle_counter import VehicleCounter
from visionguard.traffic.people_counter import PeopleCounter

class TestTrafficSubsystem(unittest.TestCase):

    def test_traffic_analytics(self):
        analytics = TrafficAnalytics()
        analytics.log_snapshot(
            vehicle_count=5,
            vehicle_breakdown={"car": 3, "truck": 1, "bus": 0, "motorbike": 1},
            people_up=2,
            people_down=1,
            current_fps=28.5
        )
        summary = analytics.get_summary()
        self.assertEqual(summary["total_vehicles"], 5)
        self.assertEqual(summary["total_pedestrians"], 3)
        self.assertEqual(summary["vehicle_breakdown"]["car"], 3)

    def test_vehicle_counter_init(self):
        counter = VehicleCounter()
        self.assertEqual(counter.total_count, 0)
        self.assertIn("car", counter.class_counts)

    def test_people_counter_init(self):
        counter = PeopleCounter()
        self.assertEqual(len(counter.counted_up_ids), 0)
        self.assertEqual(len(counter.counted_down_ids), 0)

if __name__ == "__main__":
    unittest.main()
