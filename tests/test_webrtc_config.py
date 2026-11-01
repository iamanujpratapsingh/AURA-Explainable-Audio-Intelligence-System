import os
import unittest
from unittest.mock import patch

from backend.webrtc_config import build_rtc_configuration


class WebRtcConfigurationTests(unittest.TestCase):
    def test_stun_is_used_without_turn_credentials(self):
        configuration = build_rtc_configuration({})
        self.assertEqual(len(configuration["iceServers"]), 1)
        self.assertIn("stun:", configuration["iceServers"][0]["urls"][0])

    def test_turn_is_added_from_secret_values(self):
        configuration = build_rtc_configuration(
            {
                "AURA_TURN_URLS": "turn:relay.example:3478, turns:relay.example:5349",
                "AURA_TURN_USERNAME": "user",
                "AURA_TURN_PASSWORD": "password",
            }
        )
        turn = configuration["iceServers"][1]
        self.assertEqual(turn["urls"], ["turn:relay.example:3478", "turns:relay.example:5349"])
        self.assertEqual(turn["username"], "user")
        self.assertEqual(turn["credential"], "password")

    def test_environment_values_are_used(self):
        with patch.dict(
            os.environ,
            {
                "AURA_TURN_URLS": "turn:env.example:3478",
                "AURA_TURN_USERNAME": "env-user",
                "AURA_TURN_PASSWORD": "env-password",
            },
            clear=False,
        ):
            configuration = build_rtc_configuration({})
        self.assertEqual(configuration["iceServers"][1]["username"], "env-user")

    def test_partial_turn_configuration_is_not_sent(self):
        configuration = build_rtc_configuration(
            {"AURA_TURN_URLS": "turn:relay.example:3478", "AURA_TURN_USERNAME": "user"}
        )
        self.assertEqual(len(configuration["iceServers"]), 1)


if __name__ == "__main__":
    unittest.main()
