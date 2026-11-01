import io
import queue
import unittest
import wave
from unittest.mock import Mock

import av
import numpy as np

from frontend.live_audio import LiveAudioCapture, consume_receiver_frames, process_audio_frame


class LiveAudioCaptureTests(unittest.TestCase):
    def make_frame(self, samples=160, sample_rate=16000):
        frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = sample_rate
        frame.planes[0].update(np.zeros(samples, dtype=np.int16).tobytes())
        return frame

    def make_signal_frame(self, samples=160, sample_rate=16000):
        frame = av.AudioFrame(format="s16", layout="mono", samples=samples)
        frame.sample_rate = sample_rate
        frame.planes[0].update(np.full(samples, 8192, dtype=np.int16).tobytes())
        return frame

    def test_capture_tracks_bounded_amplitude_and_exports_valid_wav(self):
        capture = LiveAudioCapture()
        self.assertTrue(capture.add_frame(self.make_frame()))
        self.assertEqual(len(capture.amplitude_snapshot()), 1)
        self.assertEqual(capture.amplitude_snapshot()[0], 0.0)
        self.assertEqual(capture.duration_seconds, 0.01)

        with wave.open(io.BytesIO(capture.to_wav_bytes()), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnframes(), 160)

    def test_capture_stops_appending_at_size_limit(self):
        capture = LiveAudioCapture()
        capture.max_bytes = 100
        capture.max_bytes = 400
        self.assertTrue(capture.add_frame(self.make_frame()))
        self.assertFalse(capture.add_frame(self.make_frame()))
        self.assertTrue(capture.limit_reached)
        self.assertEqual(capture.total_bytes, 320)

    def test_amplitude_history_uses_fixed_rolling_window(self):
        capture = LiveAudioCapture(max_amplitude_points=3)
        for _ in range(5):
            capture.add_frame(self.make_frame())
        self.assertEqual(len(capture.amplitude_snapshot()), 3)

    def test_pcm_sink_receives_frames_independently_of_waveform_buffer(self):
        sent = []
        capture = LiveAudioCapture(pcm_sink=lambda pcm, seconds, captured_at: sent.append((pcm, seconds, captured_at)))
        self.assertTrue(capture.add_frame(self.make_frame()))
        self.assertEqual(len(sent), 1)
        self.assertEqual(len(sent[0][0]), 320)
        self.assertEqual(sent[0][1], capture.duration_seconds)
        self.assertGreater(sent[0][2], 0)
        self.assertEqual(len(capture.amplitude_snapshot()), 1)

    def test_pause_and_resume_control_capture_and_pcm_sink(self):
        sent = []
        capture = LiveAudioCapture(pcm_sink=lambda *item: sent.append(item))
        capture.pause()
        self.assertFalse(capture.add_frame(self.make_frame()))
        self.assertEqual(capture.duration_seconds, 0)
        self.assertEqual(sent, [])

        capture.resume()
        self.assertTrue(capture.add_frame(self.make_frame()))
        self.assertEqual(len(sent), 1)

    def test_capture_diagnostics_report_received_frame_format(self):
        capture = LiveAudioCapture()
        capture.add_frame(self.make_signal_frame())
        diagnostics = capture.diagnostics_snapshot()
        self.assertEqual(diagnostics["frame_count"], 1)
        self.assertEqual(diagnostics["processed_frame_count"], 1)
        self.assertEqual(diagnostics["sample_rate"], 16000)
        self.assertEqual(diagnostics["channels"], 1)
        self.assertEqual(diagnostics["samples"], 160)
        self.assertEqual(diagnostics["frame_duration_ms"], 10.0)
        self.assertGreater(diagnostics["rms"], 0)
        self.assertIsNotNone(diagnostics["last_frame_monotonic"])

    def test_generated_audio_reaches_transcription_queue_sink(self):
        submitted = []
        capture = LiveAudioCapture(
            pcm_sink=lambda pcm, seconds, captured_at: submitted.append(
                (pcm, seconds, captured_at)
            )
        )
        self.assertTrue(capture.add_frame(self.make_signal_frame()))
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(submitted[0][0]), 320)
        self.assertGreater(submitted[0][1], 0)

    def test_audio_callback_returns_frame_and_records_callback_errors(self):
        capture = LiveAudioCapture()
        frame = self.make_frame()
        self.assertIs(process_audio_frame(frame, capture), frame)
        self.assertEqual(capture.diagnostics_snapshot()["frame_count"], 1)

        invalid = Mock()
        invalid.sample_rate = 16000
        invalid.layout.channels = ()
        self.assertIs(process_audio_frame(invalid, capture), invalid)
        self.assertEqual(capture.diagnostics_snapshot()["frame_count"], 2)

        class BrokenFrame:
            @property
            def sample_rate(self):
                raise RuntimeError("frame metadata unavailable")

        broken = BrokenFrame()
        self.assertIs(process_audio_frame(broken, capture), broken)
        diagnostics = capture.diagnostics_snapshot()
        self.assertGreaterEqual(diagnostics["callback_error_count"], 1)
        self.assertIn("frame metadata unavailable", diagnostics["callback_error"])

    def test_empty_nonblocking_receiver_poll_is_not_a_stream_failure(self):
        receiver = Mock()
        receiver.get_frames.side_effect = queue.Empty
        self.assertEqual(consume_receiver_frames(receiver, LiveAudioCapture()), 0)

    def test_real_receiver_error_is_not_swallowed(self):
        receiver = Mock()
        receiver.get_frames.side_effect = RuntimeError("network disconnected")
        with self.assertRaisesRegex(RuntimeError, "network disconnected"):
            consume_receiver_frames(receiver, LiveAudioCapture())


if __name__ == "__main__":
    unittest.main()