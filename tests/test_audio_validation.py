import io
import tempfile
import unittest
import wave
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import call
from unittest.mock import patch

from backend.aura_engine import AuraEngine
from google.genai import types


class AudioValidationTests(unittest.TestCase):
    def test_live_client_timeout_uses_client_level_http_options(self):
        live_client = object()
        with patch("backend.aura_engine.GEMINI_API_KEY", "test-key"), patch(
            "backend.aura_engine.genai.Client",
            side_effect=[object(), live_client],
        ) as create_client:
            engine = AuraEngine()

        self.assertIs(engine.live_client, live_client)
        create_client.assert_has_calls([
            call(api_key="test-key"),
            call(api_key="test-key", http_options=types.HttpOptions(timeout=20_000)),
        ])

    def write_temp(self, directory, name, content):
        path = Path(directory) / name
        path.write_bytes(content)
        return str(path)

    def valid_wav(self):
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x00\x00" * 160)
        return output.getvalue()

    def test_empty_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temp(directory, "empty.wav", b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                AuraEngine._read_validated_audio(path)

    def test_corrupted_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temp(directory, "broken.wav", b"not a wave file")
            with self.assertRaisesRegex(ValueError, "malformed"):
                AuraEngine._read_validated_audio(path)

    def test_unsupported_extension_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temp(directory, "audio.ogg", b"OggS")
            with self.assertRaisesRegex(ValueError, "Unsupported audio format"):
                AuraEngine._read_validated_audio(path)

    def test_valid_wav_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = self.valid_wav()
            path = self.write_temp(directory, "valid.wav", expected)
            audio_bytes, mime_type = AuraEngine._read_validated_audio(path)
            self.assertEqual(audio_bytes, expected)
            self.assertEqual(mime_type, "audio/wav")

    def test_audio_size_limit_is_enforced_from_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temp(directory, "valid.wav", self.valid_wav())
            with patch.dict("os.environ", {"AURA_MAX_AUDIO_MB": "0.0001"}):
                with self.assertRaisesRegex(ValueError, "exceeds the configured"):
                    AuraEngine._read_validated_audio(path)

    def test_provider_timeout_is_reported_as_audio_processing_failure(self):
        class FakeModels:
            def generate_content(self, **kwargs):
                raise TimeoutError("request timed out")

        class FakeClient:
            models = FakeModels()

        engine = AuraEngine()
        engine._gemini_ready = True
        engine._client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_temp(directory, "valid.wav", self.valid_wav())
            with patch("backend.aura_engine.time.time", return_value=0):
                with self.assertRaisesRegex(ValueError, "Audio processing failed.*timed out"):
                    engine.process_audio(path)

    def test_transient_provider_capacity_errors_are_retried(self):
        class FakeModels:
            def __init__(self):
                self.calls = 0

            def generate_content(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("503 UNAVAILABLE: high demand")
                return SimpleNamespace(text="ok")

        fake_models = FakeModels()
        engine = AuraEngine()
        engine._client = SimpleNamespace(models=fake_models)
        with patch.dict("os.environ", {"AURA_PROVIDER_RETRIES": "1"}):
            response = engine._generate_content_with_retry(model="test", contents="test")

        self.assertEqual(response.text, "ok")
        self.assertEqual(fake_models.calls, 2)


if __name__ == "__main__":
    unittest.main()