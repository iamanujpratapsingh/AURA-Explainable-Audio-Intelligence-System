import io
import logging
import queue
import threading
import time
import wave
from collections import deque

import av
import numpy as np


logger = logging.getLogger(__name__)


def consume_receiver_frames(receiver, capture):
    try:
        frames = receiver.get_frames(block=False)
    except queue.Empty:
        return 0
    logger.debug("WebRTC receiver returned %d audio frame(s)", len(frames))
    for frame in frames:
        capture.add_frame(frame)
    return len(frames)


class LiveAudioCapture:
    def __init__(self, max_duration_seconds=180, max_amplitude_points=120, pcm_sink=None):
        self.max_duration_seconds = max_duration_seconds
        self.max_bytes = max_duration_seconds * 32000
        self.pcm_chunks = []
        self.total_bytes = 0
        self.sample_rate = None
        self.channels = None
        self.resampler = None
        self.amplitudes = deque(maxlen=max_amplitude_points)
        self.amplitude_points = deque(maxlen=max_amplitude_points)
        self.limit_reached = False
        self.pcm_sink = pcm_sink
        self.paused = False
        self.frame_count = 0
        self.last_frame_monotonic = None
        self.last_frame_sample_rate = None
        self.last_frame_channels = None
        self.last_frame_format = None
        self.processed_frame_count = 0
        self.lock = threading.Lock()

    def pause(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False

    def add_frame(self, frame):
        frame_received_at = time.monotonic()
        sample_rate = getattr(frame, "sample_rate", None)
        layout = getattr(frame, "layout", None)
        channels = len(layout.channels) if layout and layout.channels else 0
        frame_format = str(getattr(frame, "format", "") or "")
        with self.lock:
            self.frame_count += 1
            self.last_frame_monotonic = frame_received_at
            self.last_frame_sample_rate = sample_rate
            self.last_frame_channels = channels
            self.last_frame_format = frame_format
        if not sample_rate or not channels:
            logger.warning(
                "Received invalid WebRTC audio frame: sample_rate=%r channels=%r format=%r",
                sample_rate,
                channels,
                frame_format,
            )
            return False
        emitted_chunks = []
        with self.lock:
            if self.paused:
                return False
            if self.resampler is None:
                self.resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            converted_frames = self.resampler.resample(frame)
            self.sample_rate = 16000
            self.channels = 1
            for converted in converted_frames:
                samples = converted.to_ndarray()
                pcm_bytes = np.ascontiguousarray(samples).tobytes()
                if not pcm_bytes:
                    continue
                normalized = samples.astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(np.square(normalized))))
                if self.total_bytes + len(pcm_bytes) > self.max_bytes:
                    self.limit_reached = True
                    return False
                self.pcm_chunks.append(pcm_bytes)
                self.total_bytes += len(pcm_bytes)
                amplitude = round(min(1.0, rms), 4)
                capture_seconds = self.total_bytes / 32000
                self.amplitudes.append(amplitude)
                self.amplitude_points.append((capture_seconds, amplitude))
                emitted_chunks.append((pcm_bytes, capture_seconds, time.monotonic()))
                self.processed_frame_count += 1
            pcm_sink = self.pcm_sink

        if pcm_sink is not None:
            for pcm_bytes, captured_seconds, captured_monotonic in emitted_chunks:
                pcm_sink(pcm_bytes, captured_seconds, captured_monotonic)
        return bool(emitted_chunks)

    def diagnostics_snapshot(self):
        with self.lock:
            bytes_per_second = (self.sample_rate or 0) * (self.channels or 0) * 2
            return {
                "frame_count": self.frame_count,
                "processed_frame_count": self.processed_frame_count,
                "last_frame_monotonic": self.last_frame_monotonic,
                "sample_rate": self.last_frame_sample_rate,
                "channels": self.last_frame_channels,
                "format": self.last_frame_format,
                "duration_seconds": self.total_bytes / bytes_per_second if bytes_per_second else 0.0,
            }

    def amplitude_snapshot(self):
        with self.lock:
            return list(self.amplitudes)

    def amplitude_points_snapshot(self):
        with self.lock:
            return list(self.amplitude_points)

    def to_wav_bytes(self):
        with self.lock:
            if not self.pcm_chunks or not self.sample_rate or not self.channels:
                return b""
            sample_rate = self.sample_rate
            channels = self.channels
            chunks = tuple(self.pcm_chunks)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            for chunk in chunks:
                wav_file.writeframesraw(chunk)
        return output.getvalue()

    @property
    def duration_seconds(self):
        with self.lock:
            bytes_per_second = (self.sample_rate or 0) * (self.channels or 0) * 2
            return self.total_bytes / bytes_per_second if bytes_per_second else 0.0