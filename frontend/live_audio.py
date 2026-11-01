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


def process_audio_frame(frame, capture):
    """Process one WebRTC frame in the streamlit-webrtc worker thread."""
    capture.record_callback_invocation(frame)
    frame_format = _safe_frame_attribute(frame, "format")
    frame_layout = _safe_frame_attribute(frame, "layout")
    logger.debug(
        "WebRTC audio callback #%d: type=%s sample_rate=%r samples=%r format=%s layout=%s",
        capture.callback_count,
        type(frame).__name__,
        _safe_frame_attribute(frame, "sample_rate"),
        _safe_frame_attribute(frame, "samples"),
        getattr(frame_format, "name", frame_format),
        getattr(frame_layout, "name", frame_layout),
    )
    try:
        capture.add_frame(frame)
    except Exception as exc:
        capture.record_callback_error(exc)
        logger.exception("WebRTC audio frame callback failed")
    return frame


def _safe_frame_attribute(frame, name):
    try:
        return getattr(frame, name, None)
    except Exception as exc:
        logger.exception("Could not read WebRTC frame.%s", name)
        return f"<error: {type(exc).__name__}>"


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
        self.last_frame_layout = None
        self.last_frame_samples = 0
        self.last_frame_duration_ms = 0.0
        self.last_rms = 0.0
        self.processed_frame_count = 0
        self.callback_error = ""
        self.callback_error_count = 0
        self.callback_count = 0
        self.last_array_shape = None
        self.last_array_dtype = None
        self.last_min = 0.0
        self.last_max = 0.0
        self.last_mean = 0.0
        self.lock = threading.Lock()

    def pause(self):
        with self.lock:
            self.paused = True

    def resume(self):
        with self.lock:
            self.paused = False

    def add_frame(self, frame):
        frame_received_at = time.monotonic()
        with self.lock:
            self.frame_count += 1
            self.last_frame_monotonic = frame_received_at
        sample_rate = getattr(frame, "sample_rate", None)
        layout = getattr(frame, "layout", None)
        try:
            channels = len(layout.channels) if layout and layout.channels else 0
        except (TypeError, AttributeError):
            channels = 0
        frame_format_object = getattr(frame, "format", None)
        frame_format = str(getattr(frame_format_object, "name", frame_format_object) or "")
        frame_layout_object = getattr(frame, "layout", None)
        frame_layout = str(getattr(frame_layout_object, "name", frame_layout_object) or "")
        try:
            frame_samples = int(getattr(frame, "samples", 0) or 0)
        except (TypeError, ValueError):
            frame_samples = 0
        frame_duration_ms = (
            frame_samples / float(sample_rate) * 1000
            if sample_rate and frame_samples
            else 0.0
        )
        with self.lock:
            self.last_frame_sample_rate = sample_rate
            self.last_frame_channels = channels
            self.last_frame_format = frame_format
            self.last_frame_layout = frame_layout
            self.last_frame_samples = frame_samples
            self.last_frame_duration_ms = frame_duration_ms
        if not sample_rate or not channels:
            logger.warning(
                "Received invalid WebRTC audio frame: sample_rate=%r channels=%r format=%r",
                sample_rate,
                channels,
                frame_format,
            )
            return False
        raw_array = np.asarray(frame.to_ndarray())
        if raw_array.size == 0:
            logger.warning("Received empty decoded WebRTC audio array: shape=%s dtype=%s", raw_array.shape, raw_array.dtype)
            return False
        raw_float = self._normalize_samples(raw_array)
        raw_flat = raw_float.reshape(-1)
        raw_rms = float(np.sqrt(np.mean(np.square(raw_flat))))
        with self.lock:
            self.last_array_shape = tuple(raw_array.shape)
            self.last_array_dtype = str(raw_array.dtype)
            self.last_min = float(np.min(raw_float))
            self.last_max = float(np.max(raw_float))
            self.last_mean = float(np.mean(raw_float))
            self.last_rms = round(min(1.0, raw_rms), 6)
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
                samples = np.asarray(converted.to_ndarray())
                pcm_bytes = np.ascontiguousarray(samples).tobytes()
                if not pcm_bytes:
                    continue
                normalized = self._normalize_samples(samples)
                rms = float(np.sqrt(np.mean(np.square(normalized.reshape(-1)))))
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

    @staticmethod
    def _normalize_samples(samples):
        samples = np.asarray(samples)
        if np.issubdtype(samples.dtype, np.floating):
            return samples.astype(np.float32, copy=False)
        if np.issubdtype(samples.dtype, np.signedinteger):
            info = np.iinfo(samples.dtype)
            scale = max(abs(info.min), info.max)
            return samples.astype(np.float32) / float(scale)
        if np.issubdtype(samples.dtype, np.unsignedinteger):
            info = np.iinfo(samples.dtype)
            midpoint = (info.max + 1) / 2.0
            return (samples.astype(np.float32) - midpoint) / midpoint
        raise TypeError(f"Unsupported audio sample dtype: {samples.dtype}")

    def record_callback_invocation(self, frame):
        with self.lock:
            self.callback_count += 1

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
                "layout": self.last_frame_layout,
                "samples": self.last_frame_samples,
                "frame_duration_ms": self.last_frame_duration_ms,
                "rms": self.last_rms,
                "duration_seconds": self.total_bytes / bytes_per_second if bytes_per_second else 0.0,
                "callback_error": self.callback_error,
                "callback_error_count": self.callback_error_count,
                "callback_count": self.callback_count,
                "array_shape": self.last_array_shape,
                "array_dtype": self.last_array_dtype,
                "minimum": self.last_min,
                "maximum": self.last_max,
                "mean": self.last_mean,
            }

    def record_callback_error(self, error):
        with self.lock:
            self.callback_error_count += 1
            self.callback_error = f"{type(error).__name__}: {error}"

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