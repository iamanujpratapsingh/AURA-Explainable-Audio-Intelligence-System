import os
import base64
import hashlib
import hmac
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
                "AURA_ALLOW_STATIC_TURN_CREDENTIALS": "true",
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
                "AURA_ALLOW_STATIC_TURN_CREDENTIALS": "true",
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

    def test_ephemeral_turn_credentials_use_hmac_and_expiry(self):
        with patch("backend.webrtc_config.time.time", return_value=1_000):
            configuration = build_rtc_configuration(
                {
                    "AURA_TURN_URLS": "turn:relay.example:3478",
                    "AURA_TURN_EPHEMERAL_SECRET": "shared-secret",
                    "AURA_TURN_CREDENTIAL_TTL": "600",
                }
            )
        turn = configuration["iceServers"][1]
        username = "aura-1600"
        expected = base64.b64encode(
            hmac.new(
                b"shared-secret",
                username.encode("utf-8"),
                hashlib.sha1,
            ).digest()
        ).decode("ascii")
        self.assertEqual(turn["username"], username)
        self.assertEqual(turn["credential"], expected)

    def test_ephemeral_credentials_take_precedence_over_static_values(self):
        with patch("backend.webrtc_config.time.time", return_value=1_000):
            configuration = build_rtc_configuration(
                {
                    "AURA_TURN_URLS": "turn:relay.example:3478",
                    "AURA_TURN_USERNAME": "long-lived-user",
                    "AURA_TURN_PASSWORD": "long-lived-password",
                    "AURA_TURN_EPHEMERAL_SECRET": "shared-secret",
                }
            )
        self.assertEqual(configuration["iceServers"][1]["username"], "aura-4600")


if __name__ == "__main__":
    unittest.main()
