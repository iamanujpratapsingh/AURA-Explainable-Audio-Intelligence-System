import asyncio
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from backend.live_transcription import LiveTranscriptionService, normalize_transcript


class FakeLiveSession:
    def __init__(self):
        self.responses = asyncio.Queue()
        self.sent_audio = []
        self.stream_ended = False

    async def send_realtime_input(self, audio=None, audio_stream_end=False):
        if audio is not None:
            self.sent_audio.append(audio)
        if audio_stream_end:
            self.stream_ended = True
            transcription = SimpleNamespace(text="Please help", language_code="en-US")
            await self.responses.put(SimpleNamespace(server_content=SimpleNamespace(
                interim_input_transcription=SimpleNamespace(text="Please help me"),
                input_transcription=transcription,
            )))
            await self.responses.put(SimpleNamespace(server_content=SimpleNamespace(
                interim_input_transcription=None,
                input_transcription=transcription,
            )))
            await self.responses.put(None)

    async def receive(self):
        while True:
            response = await self.responses.get()
            if response is None:
                return
            yield response


class SlowLiveSession(FakeLiveSession):
    async def send_realtime_input(self, audio=None, audio_stream_end=False):
        if audio is not None:
            await asyncio.sleep(0.1)
        await super().send_realtime_input(audio, audio_stream_end)


class FakeConnectContext:
    def __init__(self, session):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeClient:
    def __init__(self, session):
        self.connect_kwargs = None
        self.aio = SimpleNamespace(
            live=SimpleNamespace(
                connect=self._connect
            )
        )
        self.session = session

    def _connect(self, **kwargs):
        self.connect_kwargs = kwargs
        return FakeConnectContext(self.session)


class LiveTranscriptionTests(unittest.TestCase):
    def test_normalization_handles_punctuation_and_repeated_whitespace(self):
        self.assertEqual(normalize_transcript("  HELP... me!!  "), "help me")

    def test_live_session_sends_pcm_and_stops_cleanly_with_final_transcript(self):
        session = FakeLiveSession()
        client = FakeClient(session)
        service = LiveTranscriptionService(client, finalize_seconds=0.05)
        service.start()
        deadline = time.monotonic() + 2
        while not service.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(service.connected)

        capture_time = time.monotonic()
        self.assertTrue(service.submit_audio(b"\0" * 3200, 0.1, capture_time))
        self.assertTrue(service.stop(join_timeout=2))
        self.assertTrue(session.stream_ended)
        self.assertNotIn("http_options", client.connect_kwargs["config"].model_fields_set)
        self.assertIsNone(client.connect_kwargs["config"].http_options)
        self.assertEqual(sum(len(chunk.data) for chunk in session.sent_audio), 3200)
        self.assertTrue(all(chunk.mime_type == "audio/pcm;rate=16000" for chunk in session.sent_audio))

        events = service.drain_events()
        interim = [event for event in events if event["kind"] == "interim"]
        finals = [event for event in events if event["kind"] == "final"]
        self.assertEqual(len(interim), 1, events)
        self.assertEqual(len(finals), 1, events)
        self.assertEqual(finals[0]["text"], "Please help")
        self.assertEqual(finals[0]["capture_seconds"], 0.1)
        self.assertGreaterEqual(finals[0]["provider_latency_ms"], 0)

    def test_audio_queue_is_bounded_and_drops_oldest_without_blocking(self):
        service = LiveTranscriptionService(FakeClient(FakeLiveSession()), audio_queue_size=2)
        for index in range(3):
            service.submit_audio(bytes([index]), index / 10, time.monotonic())
        self.assertEqual(service.audio_queue.qsize(), 2)
        self.assertEqual(service.dropped_audio_chunks, 1)
        self.assertEqual(service.audio_queue.get_nowait()[0], b"\x01")

    def test_stop_before_start_is_safe(self):
        service = LiveTranscriptionService(FakeClient(FakeLiveSession()))
        self.assertTrue(service.stop())
        self.assertEqual(service.status, "stopped")
        self.assertFalse(service.is_running)

    def test_request_stop_is_nonblocking_and_prevents_later_audio_submission(self):
        service = LiveTranscriptionService(FakeClient(FakeLiveSession()))
        service.request_stop()
        self.assertEqual(service.status, "stopped")
        self.assertFalse(service.submit_audio(b"\0" * 320, 0.01, time.monotonic()))

    def test_request_stop_preserves_worker_error_status(self):
        def fail_connect(**kwargs):
            raise RuntimeError("connection refused")

        client = SimpleNamespace(
            aio=SimpleNamespace(live=SimpleNamespace(connect=fail_connect))
        )
        service = LiveTranscriptionService(client)
        service.start()
        deadline = time.monotonic() + 2
        while service.status != "error" and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(service.status, "error")
        service.request_stop()

        self.assertEqual(service.status, "error")
        self.assertIn("connection refused", service.error)

    def test_slow_provider_send_fails_with_a_timeout(self):
        service = LiveTranscriptionService(
            FakeClient(SlowLiveSession()),
            operation_timeout_seconds=0.01,
        )
        service.start()
        deadline = time.monotonic() + 2
        while not service.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(service.connected)
        service.submit_audio(b"\0" * 3200, 0.1, time.monotonic())

        deadline = time.monotonic() + 2
        while service.status != "error" and time.monotonic() < deadline:
            time.sleep(0.01)

        self.assertEqual(service.status, "error")
        self.assertIn("timed out", service.error.lower())
        self.assertTrue(service.stop(join_timeout=2))

    def test_repeated_final_transcriptions_for_same_audio_are_deduplicated(self):
        session = FakeLiveSession()
        service = LiveTranscriptionService(FakeClient(session), finalize_seconds=0.05)
        service.start()
        deadline = time.monotonic() + 2
        while not service.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(service.connected)

        service.submit_audio(b"\0" * 3200, 0.1, time.monotonic())
        self.assertTrue(service.stop(join_timeout=2))

        finals = [event for event in service.drain_events() if event["kind"] == "final"]
        self.assertEqual(len(finals), 1)

    def test_generated_pcm_can_be_submitted_before_audio_stream_stop(self):
        session = FakeLiveSession()
        service = LiveTranscriptionService(
            FakeClient(session),
            finalize_seconds=0.05,
        )
        self.assertTrue(service.start())
        deadline = time.monotonic() + 2
        while not service.connected and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(service.connected)
        self.assertTrue(service.submit_audio(b"\x01\x00" * 160, 0.01, time.monotonic()))
        self.assertGreaterEqual(service.diagnostics["submitted_audio_chunks"], 1)
        self.assertTrue(service.stop(join_timeout=2))
        events = service.drain_events()
        self.assertTrue(any(event["kind"] == "final" for event in events))


if __name__ == "__main__":
    unittest.main()
