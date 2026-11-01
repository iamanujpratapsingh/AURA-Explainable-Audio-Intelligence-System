import unittest

from backend.alert_state import acknowledge_alert, active_alerts


class AlertStateTests(unittest.TestCase):
    def test_acknowledgment_is_recorded_once(self):
        self.assertEqual(acknowledge_alert("alert-1", []), ["alert-1"])
        self.assertEqual(acknowledge_alert("alert-1", ["alert-1"]), ["alert-1"])

    def test_acknowledged_and_expired_alerts_stop_blinking(self):
        history = [
            {"alert_id": "active", "created_at": 95},
            {"alert_id": "acknowledged", "created_at": 98},
            {"alert_id": "expired", "created_at": 80},
        ]
        active = active_alerts(history, ["acknowledged"], now=100, interval_seconds=15)
        self.assertEqual([item["alert_id"] for item in active], ["active"])


if __name__ == "__main__":
    unittest.main()