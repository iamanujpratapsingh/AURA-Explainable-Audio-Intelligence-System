import asyncio
import logging
import os
import queue
import re
import threading
import time
import unicodedata
from collections import deque

from google.genai import types


MODEL_DEFAULT = "gemini-3.5-transcribe-live"
PCM_SAMPLE_RATE = 16000
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * 2
SEND_CHUNK_BYTES = PCM_BYTES_PER_SECOND // 10
logger = logging.getLogger(__name__)


def normalize_transcript(text):
    text = str(text or "").casefold()
    text = "".join(
        character
        if character.isspace() or character.isalnum() or unicodedata.category(character).startswith("M")
        else " "
        for character in text
    )
    return " ".join(text.split())


def transcription_text(response, field_name):
    server_content = getattr(response, "server_content", None)
    if server_content is None:
        return ""
    transcription = getattr(server_content, field_name, None)
    return str(getattr(transcription, "text", "") or "").strip()


class LiveTranscriptionService:
    """One background Gemini Live transcription session with bounded queues."""

    def __init__(
        self,
        client,
        model=None,
        audio_queue_size=120,
        event_queue_size=200,
        finalize_seconds=0.8,
        connection_timeout_seconds=20.0,
        operation_timeout_seconds=10.0,
    ):
        if connection_timeout_seconds <= 0 or operation_timeout_seconds <= 0:
            raise ValueError("Live transcription timeouts must be greater than zero.")
        self.client = client
        self.model = model or os.getenv("AURA_LIVE_TRANSCRIBE_MODEL", MODEL_DEFAULT)
        self.audio_queue = queue.Queue(maxsize=audio_queue_size)
        self.event_queue = queue.Queue(maxsize=event_queue_size)
        self.finalize_seconds = finalize_seconds
        self.connection_timeout_seconds = connection_timeout_seconds
        self.operation_timeout_seconds = operation_timeout_seconds
        self._stop_event = threading.Event()
        self._connected_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._status = "idle"
        self._error = ""
        self._dropped_audio_chunks = 0
        self._dropped_events = 0
        self._submitted_audio_chunks = 0
        self._sent_audio_chunks = 0
        self._last_audio_sent_monotonic = None
        self._final_signatures = set()
        self._final_signature_order = deque(maxlen=256)
        self._last_capture_seconds = 0.0
        self._last_capture_monotonic = None
        self._event_sequence = 0

    @property
    def status(self):
        with self._lock:
            return self._status

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def dropped_audio_chunks(self):
        with self._lock:
            return self._dropped_audio_chunks

    @property
    def audio_queue_size(self):
        return self.audio_queue.qsize()

    @property
    def diagnostics(self):
        with self._lock:
            return {
                "status": self._status,
                "audio_queue_size": self.audio_queue.qsize(),
                "submitted_audio_chunks": self._submitted_audio_chunks,
                "sent_audio_chunks": self._sent_audio_chunks,
                "last_audio_sent_monotonic": self._last_audio_sent_monotonic,
                "dropped_audio_chunks": self._dropped_audio_chunks,
                "dropped_events": self._dropped_events,
                "connected": self._connected_event.is_set(),
            }

    @property
    def connected(self):
        return self._connected_event.is_set() and self.status == "connected"

    @property
    def is_running(self):
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._connected_event.clear()
            self._status = "connecting"
            self._error = ""
            self._thread = threading.Thread(target=self._thread_main, name="aura-live-transcriber", daemon=True)
            self._thread.start()
        return True

    def submit_audio(self, pcm_bytes, captured_seconds=0.0, captured_monotonic=None):
        if not pcm_bytes or self._stop_event.is_set():
            return False
        item = (
            bytes(pcm_bytes),
            max(0.0, float(captured_seconds or 0.0)),
            captured_monotonic if captured_monotonic is not None else time.monotonic(),
        )
        try:
            self.audio_queue.put_nowait(item)
            with self._lock:
                self._submitted_audio_chunks += 1
            return True
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped_audio_chunks += 1
            try:
                self.audio_queue.put_nowait(item)
                with self._lock:
                    self._submitted_audio_chunks += 1
                return True
            except queue.Full:
                with self._lock:
                    self._dropped_audio_chunks += 1
                return False

    def drain_events(self, limit=100):
        events = []
        for _ in range(max(0, limit)):
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    def stop(self, join_timeout=3.0):
        self.request_stop()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
        return not (thread and thread.is_alive())

    def request_stop(self):
        self._stop_event.set()
        publish_stopped = False
        with self._lock:
            thread = self._thread
            worker_finished = not thread or not thread.is_alive()
            if worker_finished and self._status not in {"error", "stopped"}:
                self._status = "stopped"
                publish_stopped = True
        if publish_stopped:
            self._publish(
                {"kind": "status", "status": "stopped", "received_monotonic": time.monotonic()}
            )

    def _thread_main(self):
        try:
            asyncio.run(self._run_session())
        except Exception as exc:
            message = str(exc).replace(os.getenv("GEMINI_API_KEY", "\0"), "[redacted]").strip()
            if not message and isinstance(exc, TimeoutError):
                message = "Live transcription request timed out."
            with self._lock:
                self._status = "error"
                self._error = f"{type(exc).__name__}: {message or 'Live transcription connection failed.'}"
            logger.error("Live transcription worker failed (%s): %s", type(exc).__name__, message)
            self._publish({"kind": "error", "error": self.error, "received_monotonic": time.monotonic()})
        else:
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"
            self._publish({"kind": "status", "status": self.status, "received_monotonic": time.monotonic()})
        finally:
            self._connected_event.clear()

    async def _run_session(self):
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[]),
        )
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            self._connected_event.set()
            with self._lock:
                self._status = "connected"
            self._publish({"kind": "status", "status": "connected", "received_monotonic": time.monotonic()})

            receiver_task = asyncio.create_task(self._receive_transcriptions(session))
            await self._send_audio(session)
            try:
                await asyncio.sleep(self.finalize_seconds)
            finally:
                receiver_task.cancel()
                try:
                    await receiver_task
                except asyncio.CancelledError:
                    pass

    async def _send_audio(self, session):
        buffered = bytearray()
        buffered_capture_seconds = 0.0
        buffered_capture_monotonic = None
        while True:
            try:
                item = await asyncio.to_thread(self.audio_queue.get, True, 0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue
            pcm_bytes, captured_seconds, captured_monotonic = item
            buffered.extend(pcm_bytes)
            buffered_capture_seconds = captured_seconds
            buffered_capture_monotonic = captured_monotonic
            while len(buffered) >= SEND_CHUNK_BYTES:
                chunk = bytes(buffered[:SEND_CHUNK_BYTES])
                del buffered[:SEND_CHUNK_BYTES]
                await self._send_chunk(session, chunk, buffered_capture_seconds, buffered_capture_monotonic)

        if buffered:
            await self._send_chunk(session, bytes(buffered), buffered_capture_seconds, buffered_capture_monotonic)
        await asyncio.wait_for(
            session.send_realtime_input(audio_stream_end=True),
            timeout=self.operation_timeout_seconds,
        )

    async def _send_chunk(self, session, pcm_bytes, captured_seconds, captured_monotonic):
        await asyncio.wait_for(
            session.send_realtime_input(
                audio=types.Blob(data=pcm_bytes, mime_type=f"audio/pcm;rate={PCM_SAMPLE_RATE}")
            ),
            timeout=self.operation_timeout_seconds,
        )
        with self._lock:
            self._last_capture_seconds = captured_seconds
            self._last_capture_monotonic = captured_monotonic
            self._sent_audio_chunks += 1
            self._last_audio_sent_monotonic = time.monotonic()

    async def _receive_transcriptions(self, session):
        async for response in session.receive():
            server_content = getattr(response, "server_content", None)
            if server_content is None:
                continue
            received = time.monotonic()
            interim = transcription_text(response, "interim_input_transcription")
            if interim:
                self._publish_transcription("interim", interim, received)
            final = transcription_text(response, "input_transcription")
            if final:
                self._publish_transcription("final", final, received)

    def _publish_transcription(self, kind, text, received_monotonic):
        normalized = normalize_transcript(text)
        if not normalized:
            return
        with self._lock:
            capture_seconds = self._last_capture_seconds
            capture_monotonic = self._last_capture_monotonic
            if kind == "final":
                signature = (normalized, round(capture_seconds, 1))
                if signature in self._final_signatures:
                    return
                if len(self._final_signature_order) == self._final_signature_order.maxlen:
                    expired = self._final_signature_order.popleft()
                    self._final_signatures.discard(expired)
                self._final_signature_order.append(signature)
                self._final_signatures.add(signature)
                self._event_sequence += 1
                event_id = self._event_sequence
            else:
                event_id = None
        self._publish(
            {
                "kind": kind,
                "text": text,
                "normalized_text": normalized,
                "event_id": event_id,
                "capture_seconds": capture_seconds,
                "capture_monotonic": capture_monotonic,
                "recognized_monotonic": received_monotonic,
                "provider_latency_ms": (
                    max(0.0, (received_monotonic - capture_monotonic) * 1000)
                    if capture_monotonic is not None
                    else None
                ),
            }
        )

    def _publish(self, event):
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                pass
            with self._lock:
                self._dropped_events += 1
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                with self._lock:
                    self._dropped_events += 1
