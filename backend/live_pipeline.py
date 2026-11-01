import time
from collections import deque


def monitoring_state(active, stream_playing, has_audio, transcription_status, error=False, paused=False):
    if error:
        return "Error"
    if not active and not stream_playing:
        return "Idle"
    if paused:
        return "Paused"
    if (
        not stream_playing
        or not has_audio
        or transcription_status in {"connecting", "waiting_for_microphone"}
    ):
        return "Processing"
    return "Listening"


class LiveTranscriptProcessor:
    def __init__(self, matcher, max_segments=200, max_seen_events=512):
        self.matcher = matcher
        self.max_segments = max_segments
        self.segments = deque(maxlen=max_segments)
        self._seen_events = set()
        self._seen_order = deque(maxlen=max_seen_events)

    def process(self, event):
        if not isinstance(event, dict) or event.get("kind") != "final":
            return None
        event_id = event.get("event_id")
        if event_id is not None and event_id in self._seen_events:
            return None

        text = str(event.get("text") or "").strip()
        if not text:
            return None
        if event_id is not None:
            if len(self._seen_order) == self._seen_order.maxlen:
                self._seen_events.discard(self._seen_order.popleft())
            self._seen_order.append(event_id)
            self._seen_events.add(event_id)

        capture_seconds = max(0.0, float(event.get("capture_seconds") or 0.0))
        segment = {
            "start": capture_seconds,
            "end": capture_seconds,
            "text": text,
            "speaker": "Unspecified",
            "tone": "Not analyzed",
            "importance": "LOW",
            "is_urgent": False,
            "confidence": None,
            "event": "",
            "flag_reasons": [],
            "segment_index": len(self.segments),
            "live_event_id": event_id,
            "timestamp_basis": "approximate live stream position; utterance/word timestamps are not provided by Live Transcription",
        }
        self.segments.append(segment)
        current_segments = list(self.segments)
        for index, item in enumerate(current_segments):
            item["segment_index"] = index
        current_index = len(current_segments) - 1

        recognized_monotonic = event.get("recognized_monotonic")
        captured_monotonic = event.get("capture_monotonic")
        match_started = time.monotonic()
        matches = self.matcher.match_segments(current_segments) if self.matcher else []
        matched_monotonic = time.monotonic()
        detections = [item for item in matches if item.get("segment_index") == current_index]

        capture_to_recognition_ms = self.elapsed_ms(captured_monotonic, recognized_monotonic)
        recognition_to_match_ms = self.elapsed_ms(recognized_monotonic, matched_monotonic)
        capture_to_match_ms = self.elapsed_ms(captured_monotonic, matched_monotonic)
        for detection in detections:
            detection.update(
                {
                    "live_event_id": event_id,
                    "capture_monotonic": captured_monotonic,
                    "recognized_monotonic": recognized_monotonic,
                    "matched_monotonic": matched_monotonic,
                    "capture_to_recognition_ms": capture_to_recognition_ms,
                    "recognition_to_match_ms": recognition_to_match_ms,
                    "capture_to_match_ms": capture_to_match_ms,
                }
            )

        return {
            "segment": segment,
            "detections": detections,
            "capture_seconds": capture_seconds,
            "capture_to_recognition_ms": capture_to_recognition_ms,
            "recognition_to_match_ms": recognition_to_match_ms,
            "capture_to_match_ms": capture_to_match_ms,
            "match_started_monotonic": match_started,
            "matched_monotonic": matched_monotonic,
        }

    @staticmethod
    def elapsed_ms(start, end):
        if start is None or end is None:
            return None
        try:
            return max(0.0, (float(end) - float(start)) * 1000)
        except (TypeError, ValueError):
            return None

    def snapshot(self):
        return list(self.segments)
