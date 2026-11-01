import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import sys
import hashlib
import json
import html as html_lib
import re
import time
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
MAX_SESSION_EVENT_HISTORY = 500

try:
    from backend.alert_state import acknowledge_alert, active_alerts
    from backend.aura_engine import AuraEngine, GEMINI_MODEL
    from backend.evaluation import evaluate_matcher, load_evaluation_dataset
    from backend.live_pipeline import LiveTranscriptProcessor, monitoring_state
    from backend.live_transcription import LiveTranscriptionService
    from frontend.audio_visualization import render_audio_visualization
    from frontend.live_audio import LiveAudioCapture, consume_receiver_frames
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception as e:
    st.error(f"Backend import failed: {e}")
    st.stop()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AURA — Audio Intelligence",
    layout="wide",
    page_icon="🎙️",
)

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f4f7f5; color: #1c2922; }
    [data-testid="stSidebar"] { background-color: #eaf0ec; }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #52635a;
        margin-bottom: 0.3rem;
    }

    /* Status pill */
    .status-ok {
        display: inline-block;
        background: #e5f4ec;
        color: #17623d;
        border: 1px solid #9bcbae;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-err {
        display: inline-block;
        background: #fff0ef;
        color: #a32626;
        border: 1px solid #e6aaa6;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .priority-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 3px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
    }
    .priority-high { background: #fff0ef; border: 1px solid #df7770; color: #992b26; }
    .priority-medium { background: #fff5df; border: 1px solid #d5a545; color: #805100; }
    .priority-low { background: #e7f4ec; border: 1px solid #78b58c; color: #205c37; }

    .alert-panel {
        border: 1px solid #df8d88; border-left: 5px solid #c3342d; background: #fff2f0;
        padding: 14px 16px; margin: 12px 0; color: #66201d;
    }
    .alert-indicator { display: inline-flex; align-items: center; gap: 9px; font-weight: 800; }
    .alert-indicator::before {
        content: ""; width: 11px; height: 11px; border-radius: 50%; background: #ef4444;
        box-shadow: 0 0 0 3px #f4c4c0; animation: aura-alert-blink 1s steps(2, start) infinite;
    }
    @keyframes aura-alert-blink { to { opacity: 0.25; } }
    @media (prefers-reduced-motion: reduce) { .alert-indicator::before { animation: none; } }

    /* Metric override */
    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Engine (cached — initializes once per session) ────────────────────────────
@st.cache_resource
def load_engine():
    return AuraEngine()

engine = load_engine()


def format_timestamp(seconds):
    """Format transcript seconds without depending on a browser locale."""
    seconds = max(0, round(float(seconds or 0)))
    minutes, second_part = divmod(seconds, 60)
    hours, minute_part = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minute_part:02d}:{second_part:02d}"
    return f"{minute_part:02d}:{second_part:02d}"


def moment_label(segment):
    event = segment.get("event") or "Transcript moment"
    return (
        f"{format_timestamp(segment.get('start'))} – {format_timestamp(segment.get('end'))} "
        f"| {segment.get('speaker', 'Unknown speaker')} | {event}"
    )


def priority_badge(priority):
    priority = str(priority or "LOW").upper()
    css_class = {"HIGH": "priority-high", "MEDIUM": "priority-medium"}.get(priority, "priority-low")
    return f'<span class="priority-badge {css_class}">{priority}</span>'


def render_bullet_list(items, empty_message):
    if items:
        st.markdown("\n".join(f"- {item}" for item in items))
    else:
        st.caption(empty_message)


def highlighted_evidence(text, detections):
    highlighted = html_lib.escape(str(text or ""))
    for detection in detections:
        phrase = html_lib.escape(str(detection.get("matched_phrase", "")))
        if phrase:
            highlighted = re.sub(
                rf"(?<!\w)({re.escape(phrase)})(?!\w)",
                r"<mark style='background:#b5472f;color:#fff;padding:1px 3px'>\1</mark>",
                highlighted,
                flags=re.IGNORECASE,
            )
    return highlighted


def render_active_alert(button_key):
    now = time.time()
    duration = max(1, int(os.getenv("AURA_ALERT_INTERVAL_SECONDS", "15")))
    active = active_alerts(
        st.session_state.alert_history,
        st.session_state.acknowledged_alerts,
        now,
        duration,
    )
    if not active:
        st.markdown("**Alert indicator:** clear · no unacknowledged detection in the alert interval")
        return

    priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    alert = max(active, key=lambda item: (priority_order.get(item.get("priority"), 0), item["created_at"]))
    evidence_requirements = ", ".join(alert.get("evidence_requirements") or [])
    st.markdown(
        '<div class="alert-panel"><span class="alert-indicator">Configured phrase detected</span><br>'
        f'<strong>{html_lib.escape(alert.get("matched_phrase", ""))}</strong> · '
        f'{html_lib.escape(alert.get("event_name", ""))} · {html_lib.escape(alert.get("category", ""))} · '
        f'{html_lib.escape(alert.get("priority", ""))} priority'
        f'<br><small>{html_lib.escape(format_timestamp(alert.get("start", 0)))} · '
        'Transcript-rule evidence only; this does not verify a real-world event.<br>'
        f'Evidence: {html_lib.escape(alert.get("original_evidence", ""))}<br>'
        f'{html_lib.escape(alert.get("explanation", ""))}<br>Evidence requirements: '
        f'{html_lib.escape(evidence_requirements)}<br>Recommended review action: '
        f'{html_lib.escape(alert.get("recommended_action", ""))}</small></div>',
        unsafe_allow_html=True,
    )
    if st.button("Acknowledge Alert", key=button_key, icon="✅"):
        st.session_state.acknowledged_alerts = acknowledge_alert(
            alert["alert_id"], st.session_state.acknowledged_alerts
        )
        st.rerun()


@st.fragment(run_every="1s")
def render_alert_indicator():
    render_active_alert("acknowledge_active_alert")


def consume_live_transcription_events(transcriber):
    if transcriber is None:
        return
    for event in transcriber.drain_events():
        kind = event.get("kind")
        if kind == "status":
            st.session_state.live_transcription_status = event.get("status", "unknown")
            continue
        if kind == "error":
            st.session_state.live_transcription_status = "error"
            st.session_state.live_transcription_error = event.get("error", "Live transcription failed.")
            continue
        if kind == "interim":
            st.session_state.live_interim_text = event.get("text", "")
            continue
        if kind != "final":
            continue

        processor = st.session_state.live_processor
        result = processor.process(event) if processor else None
        if result is None:
            continue
        st.session_state.live_transcript = processor.snapshot()
        st.session_state.live_interim_text = ""
        latency = {
            "capture_seconds": result["capture_seconds"],
            "capture_to_recognition_ms": result["capture_to_recognition_ms"],
            "recognition_to_match_ms": result["recognition_to_match_ms"],
            "capture_to_match_ms": result["capture_to_match_ms"],
            "event_id": event.get("event_id"),
        }
        ui_update_monotonic = time.monotonic()
        latency["capture_to_ui_update_ms"] = LiveTranscriptProcessor.elapsed_ms(
            event.get("capture_monotonic"), ui_update_monotonic
        )
        latency["match_to_ui_update_ms"] = LiveTranscriptProcessor.elapsed_ms(
            result["matched_monotonic"], ui_update_monotonic
        )
        latency_history = list(st.session_state.live_latency_history)
        latency_history.append(latency)
        st.session_state.live_latency_history = latency_history[-100:]

        detections = list(st.session_state.live_detections)
        seen_ids = set(st.session_state.live_detection_ids)
        alert_history = list(st.session_state.alert_history)
        detection_history = list(st.session_state.detection_history)
        known_history = {item.get("history_id") for item in detection_history}
        known_alerts = {item.get("alert_id") for item in alert_history}
        for detection in result["detections"]:
            detection["ui_update_monotonic"] = ui_update_monotonic
            detection["capture_to_ui_update_ms"] = latency["capture_to_ui_update_ms"]
            detection["match_to_ui_update_ms"] = latency["match_to_ui_update_ms"]
            detection_id = detection["detection_id"]
            if detection_id in seen_ids:
                continue
            seen_ids.add(detection_id)
            detections.append(detection)
            history_id = f"live:{st.session_state.live_stream_generation}:{detection_id}"
            if history_id not in known_history:
                detection_history.append({**detection, "history_id": history_id, "created_at": time.time()})
                known_history.add(history_id)
            if detection.get("is_alert") and history_id not in known_alerts:
                alert_history.append({**detection, "alert_id": history_id, "created_at": time.time()})
                known_alerts.add(history_id)

        st.session_state.live_detections = detections[-100:]
        st.session_state.live_detection_ids = list(seen_ids)[-500:]
        st.session_state.alert_history = alert_history[-MAX_SESSION_EVENT_HISTORY:]
        st.session_state.detection_history = detection_history[-MAX_SESSION_EVENT_HISTORY:]


@st.fragment(run_every="250ms")
def render_live_microphone():
    capture = st.session_state.live_capture
    transcriber = st.session_state.live_transcriber
    context = webrtc_streamer(
        key=f"aura-live-microphone-{st.session_state.live_stream_generation}",
        mode=WebRtcMode.SENDONLY,
        desired_playing_state=(
            st.session_state.monitoring_active and not st.session_state.monitoring_paused
        ),
        media_stream_constraints={"audio": True, "video": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        sendback_audio=False,
        sendback_video=False,
        media_toggle_controls=False,
        audio_receiver_size=64,
    )
    if context.state.playing:
        if transcriber and transcriber.status == "idle":
            transcriber.start()
            st.session_state.live_transcription_status = "connecting"
        try:
            receiver = context.audio_receiver
            if receiver is not None:
                consume_receiver_frames(receiver, capture)
        except Exception as exc:
            detail = str(exc).strip() or repr(exc) or type(exc).__name__
            st.session_state.live_monitor_error = f"Audio stream failed ({type(exc).__name__}): {detail}"
            if transcriber:
                transcriber.request_stop()

    if (
        st.session_state.monitoring_active
        and not st.session_state.monitoring_paused
        and not capture.duration_seconds
        and st.session_state.live_monitor_started_at
        and time.monotonic() - st.session_state.live_monitor_started_at > 15
    ):
        st.session_state.live_monitor_error = (
            "No microphone audio was received. Check browser microphone permission, device availability, "
            "HTTPS/localhost, and WebRTC connectivity."
        )
        st.session_state.monitoring_active = False
        if transcriber:
            transcriber.request_stop()

    consume_live_transcription_events(transcriber)
    if transcriber:
        st.session_state.live_transcription_status = transcriber.status
        if transcriber.error:
            st.session_state.live_transcription_error = transcriber.error
        if transcriber.status in {"stopped", "error"} and st.session_state.monitoring_active:
            st.session_state.monitoring_active = False

    if not st.session_state.monitoring_active and not context.state.playing:
        if not transcriber or not transcriber.is_running:
            capture.pcm_sink = None
            if not st.session_state.live_recording_bytes:
                st.session_state.live_recording_bytes = capture.to_wav_bytes()

    monitor_error = st.session_state.live_monitor_error or (
        transcriber.error if transcriber and transcriber.status == "error" else ""
    )
    monitor_status = monitoring_state(
        st.session_state.monitoring_active,
        context.state.playing,
        capture.duration_seconds > 0,
        transcriber.status if transcriber else "unavailable",
        error=bool(monitor_error),
    )
    if st.session_state.monitoring_paused and monitor_status != "Error":
        monitor_status = "Paused"
    if monitor_status == "Error":
        st.error(f"Monitoring state: Error · {monitor_error}")
    elif monitor_status == "Listening":
        st.success(f"Monitoring state: Listening · {capture.duration_seconds:.1f} seconds captured")
    elif monitor_status == "Processing":
        st.info("Monitoring state: Processing · waiting for microphone audio or transcription connection")
    elif monitor_status == "Paused":
        st.warning("Monitoring state: Paused · microphone capture and audio upload are paused")
    else:
        st.caption("Monitoring state: Idle · microphone stopped")

    if st.session_state.live_transcription_status == "connected":
        st.caption(f"Gemini Live transcription connected · model {transcriber.model}")
    elif st.session_state.live_transcription_status == "connecting":
        st.caption("Connecting to Gemini Live Transcription…")
    elif st.session_state.live_transcription_status == "unavailable":
        st.warning("Live transcription is unavailable; local amplitude capture is still active.")
    if st.session_state.get("live_transcription_error"):
        st.warning(f"Live transcription error: {st.session_state.live_transcription_error}")

    points = capture.amplitude_points_snapshot()
    if points:
        graph = go.Figure()
        graph.add_trace(go.Scatter(
            x=[point[0] for point in points],
            y=[point[1] for point in points],
            mode="lines",
            name="RMS amplitude",
            line={"color": "#16745b", "width": 2},
        ))
        for detection in st.session_state.live_detections[-20:]:
            graph.add_vline(
                x=detection.get("start", 0),
                line_color="#c3342d" if detection.get("is_alert") else "#a66a00",
                line_width=2,
                line_dash="dash",
            )
        graph.update_layout(
            template="plotly_white",
            height=230,
            margin={"l": 8, "r": 8, "t": 16, "b": 8},
            xaxis_title="Capture time (seconds)",
            yaxis_title="Amplitude (RMS)",
            showlegend=False,
        )
        st.plotly_chart(graph, use_container_width=True, key=f"live_rms_{st.session_state.live_stream_generation}")
    else:
        st.info("Amplitude samples will appear here after the microphone stream starts.")

    if st.session_state.live_interim_text:
        st.info(f"Interim transcript: {st.session_state.live_interim_text}")
    if st.session_state.live_transcript:
        st.markdown("#### Live transcript")
        for segment in reversed(st.session_state.live_transcript[-12:]):
            related = [
                item for item in st.session_state.live_detections
                if item.get("live_event_id") == segment.get("live_event_id")
            ]
            st.markdown(
                f"**~{format_timestamp(segment.get('start'))}** · "
                f"{highlighted_evidence(segment.get('text', ''), related)}",
                unsafe_allow_html=True,
            )
            if not related:
                st.caption("No matching dataset entry found.")
            for detection in related:
                st.caption(
                    f"{detection.get('matched_phrase', '')} · {detection.get('event_name', 'Event')} · "
                    f"{detection.get('category', 'Event')} · {detection.get('priority', 'LOW')} priority · "
                    f"{'Exact' if detection.get('match_type') == 'exact' else 'Approximate'} match · "
                    f"{'Rule alert; human verification required' if detection.get('is_alert') else 'Needs verification'}"
                )
                st.caption(
                    f"Evidence: {detection.get('original_evidence', '')} · "
                    f"Review action: {detection.get('recommended_action', '')}"
                )
                st.caption(
                    f"Dataset source: {detection.get('source_reference', 'Not provided')} · "
                    f"Evidence requirements: {', '.join(detection.get('evidence_requirements') or [])}"
                )
                confidence = detection.get("confidence")
                if confidence is not None:
                    st.caption(f"Provider confidence: {confidence}")
            st.caption(segment.get("timestamp_basis", "Approximate capture position; not word-level timing."))

    latest_latency = st.session_state.live_latency_history[-1] if st.session_state.live_latency_history else None
    if latest_latency:
        latency_cols = st.columns(5)
        latency_cols[0].metric(
            "Capture → transcript",
            f"{latest_latency['capture_to_recognition_ms']:.0f} ms"
            if latest_latency["capture_to_recognition_ms"] is not None else "Not available",
        )
        latency_cols[1].metric(
            "Transcript → local match",
            f"{latest_latency['recognition_to_match_ms']:.2f} ms"
            if latest_latency["recognition_to_match_ms"] is not None else "Not available",
        )
        latency_cols[2].metric(
            "Capture → local match",
            f"{latest_latency['capture_to_match_ms']:.0f} ms"
            if latest_latency["capture_to_match_ms"] is not None else "Not available",
        )
        latency_cols[3].metric(
            "Local match → dashboard",
            f"{latest_latency['match_to_ui_update_ms']:.2f} ms"
            if latest_latency["match_to_ui_update_ms"] is not None else "Not available",
        )
        latency_cols[4].metric(
            "Capture → dashboard update",
            f"{latest_latency['capture_to_ui_update_ms']:.0f} ms"
            if latest_latency["capture_to_ui_update_ms"] is not None else "Not available",
        )
        st.caption(
            "Capture timing is approximate (the provider supplies no word timestamps). Dashboard latency ends "
            "when the server updates Streamlit session state; it does not measure browser paint or user-perceived latency."
        )

    render_active_alert("live_acknowledge_active_alert")
    live_history = [
        item for item in st.session_state.detection_history
        if str(item.get("history_id", "")).startswith("live:")
    ]
    st.markdown("#### Live detection history")
    if live_history:
        acknowledged = set(st.session_state.acknowledged_alerts)
        for detection in reversed(live_history[-10:]):
            status = (
                "Acknowledged"
                if detection.get("history_id") in acknowledged
                else "Alert" if detection.get("is_alert") else "Needs verification"
            )
            st.caption(
                f"{format_timestamp(detection.get('start', 0))} · {detection.get('matched_phrase', '')} · "
                f"{detection.get('category', '')} · {detection.get('priority', 'LOW')} · {status}"
            )
    else:
        st.caption("No matching dataset entry found.")
    if capture.limit_reached:
        st.warning("The three-minute capture limit was reached. Stop monitoring to analyze the buffered recording.")


# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [("processed", False), ("chat_history", []), ("alert_history", []),
                      ("detection_history", []),
                      ("acknowledged_alerts", []), ("monitoring_active", False),
                      ("monitoring_paused", False),
                      ("live_capture", None), ("live_recording_bytes", b""),
                      ("live_monitor_error", ""), ("live_stream_generation", 0),
                      ("live_monitor_started_at", None),
                      ("live_monitor_started", False),
                      ("live_transcriber", None), ("live_transcription_status", "idle"),
                      ("live_transcription_error", ""),
                      ("live_transcript", []), ("live_interim_text", ""),
                      ("live_detection_ids", []), ("live_processor", None),
                      ("live_detections", []), ("live_latency_history", []),
                      ("upload_widget_generation", 0), ("record_widget_generation", 0),
                      ("transcript", []), ("event", ""), ("emotion", ""),
                      ("summary", ""), ("audio_reply", None), ("analysis", {}),
                      ("selected_moment", None), ("privacy_remove_temp_audio", True),
                      ("privacy_do_not_retain_history", False),
                      ("privacy_anonymize_speakers", True)]:
    if key not in st.session_state:
        st.session_state[key] = default
if st.session_state.live_capture is None:
    st.session_state.live_capture = LiveAudioCapture()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎙️ AURA")
    st.markdown("*Audio Understanding & Reasoning Agent*")
    st.markdown("---")

    # Engine status
    st.markdown('<p class="section-label">AI Engine</p>', unsafe_allow_html=True)
    st.markdown("**Google Gemini**")

    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    st.markdown(f"`{GEMINI_MODEL}`")

    st.markdown('<p class="section-label">Status</p>', unsafe_allow_html=True)
    if engine.is_ready:
        st.markdown('<span class="status-ok">● Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">● Configuration Error</span>', unsafe_allow_html=True)
        st.caption(engine.status_message)

    st.markdown("---")

    st.markdown('<p class="section-label">Language</p>', unsafe_allow_html=True)
    lang = st.selectbox(
        "Language",
        ["English", "Hindi", "Mandarin", "Urdu", "Tamil", "Spanish", "French"],
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-label">Input Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        "Input Mode",
        ["📁 Upload File", "🎤 Record Clip", "🎙️ Live Monitor"],
        label_visibility="collapsed",
    )

    with st.expander("Privacy mode"):
        st.checkbox(
            "Remove temporary uploaded audio after analysis",
            key="privacy_remove_temp_audio",
            help="Deletes AURA's temporary input file after Gemini has finished reading it.",
        )
        st.checkbox(
            "Do not retain conversation history",
            key="privacy_do_not_retain_history",
            help="Shows answers for the current interaction without keeping them in this browser session.",
        )
        st.checkbox(
            "Anonymize speaker labels",
            key="privacy_anonymize_speakers",
            help="Replaces displayed speaker labels with Speaker A, Speaker B, and so on.",
        )
        st.caption("Privacy mode improves local handling only; it is not a complete security guarantee.")

    st.markdown("---")
    st.caption("AURA · Explainable Audio Intelligence")

if mode != "🎙️ Live Monitor" and st.session_state.monitoring_active:
    transcriber = st.session_state.live_transcriber
    if transcriber:
        transcriber.request_stop()
    st.session_state.live_capture.pcm_sink = None
    st.session_state.monitoring_active = False
    st.session_state.monitoring_paused = False

if st.session_state.privacy_do_not_retain_history:
    st.session_state.chat_history = []

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_title, col_status = st.columns([5, 1])
with col_title:
    st.markdown("# 🎙️ AURA")
    st.markdown("**Audio Understanding & Reasoning Agent** — Explainable AI-powered audio intelligence and forensic analysis.")
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if engine.is_ready:
        st.markdown('<span class="status-ok">● Gemini Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-err">● Config Missing</span>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🎧 Audio Input")

audio_path = None
source_name = None
audio_bytes = None
audio_mime_type = None
audio_suffix = ".wav"
audio_input_error = None
maximum_audio_size = 50 * 1024 * 1024

if mode == "📁 Upload File":
    st.markdown('<p class="section-label">Upload an audio file (WAV, MP3, M4A; up to 50 MB)</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a"],
        label_visibility="collapsed",
        key=f"audio_upload_{st.session_state.upload_widget_generation}",
    )
    if uploaded:
        source_name = uploaded.name
        audio_suffix = os.path.splitext(uploaded.name)[-1].lower()
        audio_mime_type = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4"}.get(audio_suffix)
        if audio_mime_type is None:
            audio_input_error = "Unsupported format. Choose WAV, MP3, or M4A."
        elif uploaded.size > maximum_audio_size:
            audio_input_error = "Audio file exceeds the 50 MB upload limit."
        else:
            audio_bytes = uploaded.getvalue()
            if st.button("Remove file", key=f"remove_upload_{st.session_state.upload_widget_generation}", icon="🗑️"):
                st.session_state.upload_widget_generation += 1
                st.session_state.update({"processed": False, "analysis": {}, "selected_moment": None})
                st.rerun()

elif mode == "🎤 Record Clip":
    st.markdown('<p class="section-label">Record a complete microphone clip for analysis</p>', unsafe_allow_html=True)
    recorded = st.audio_input(
        "Record",
        label_visibility="collapsed",
        key=f"clip_recording_{st.session_state.record_widget_generation}",
    )
    if recorded:
        source_name = "microphone-recording.wav"
        audio_suffix = ".wav"
        audio_mime_type = "audio/wav"
        if recorded.size > maximum_audio_size:
            audio_input_error = "Recorded clip exceeds the 50 MB processing limit."
        else:
            audio_bytes = recorded.getvalue()
    if st.button("Start Again", key=f"record_again_{st.session_state.record_widget_generation}", icon="🔄"):
        st.session_state.record_widget_generation += 1
        st.session_state.update({"processed": False, "analysis": {}, "selected_moment": None})
        st.rerun()

elif mode == "🎙️ Live Monitor":
    st.markdown("<p class='section-label'>Live microphone amplitude monitoring</p>", unsafe_allow_html=True)
    st.warning(
        "Microphone audio is streamed to Google Gemini Live Transcription while monitoring. "
        "The amplitude graph is local; the separate buffered recording stays in memory and is only "
        "sent for full-file analysis if you choose Analyze captured audio after stopping."
    )
    st.caption("Microphone permission is required. Remote deployments need HTTPS and working WebRTC network access.")
    start_col, pause_col, stop_col = st.columns(3)
    with start_col:
        start_monitoring = st.button(
            "Start Again" if st.session_state.live_monitor_started else "Start Monitoring",
            type="primary",
            disabled=(
                st.session_state.monitoring_active
                or bool(st.session_state.live_transcriber and st.session_state.live_transcriber.is_running)
            ),
            use_container_width=True, icon="🎙️",
        )
    with pause_col:
        pause_monitoring = st.button(
            "Resume Monitoring" if st.session_state.monitoring_paused else "Pause Monitoring",
            disabled=not st.session_state.monitoring_active,
            use_container_width=True,
            icon="▶️" if st.session_state.monitoring_paused else "⏸️",
        )
    with stop_col:
        stop_monitoring = st.button(
            "Stop Monitoring", disabled=not st.session_state.monitoring_active,
            use_container_width=True, icon="⏹️",
        )
    if start_monitoring:
        transcriber = None
        if engine.live_client is not None:
            transcriber = LiveTranscriptionService(engine.live_client)
            st.session_state.live_transcription_status = "waiting_for_microphone"
        else:
            st.session_state.live_transcription_status = "unavailable"
        st.session_state.live_transcriber = transcriber
        st.session_state.live_processor = LiveTranscriptProcessor(engine.rule_matcher)
        st.session_state.live_capture = LiveAudioCapture(
            pcm_sink=transcriber.submit_audio if transcriber else None
        )
        st.session_state.live_recording_bytes = b""
        st.session_state.live_monitor_error = ""
        st.session_state.live_transcription_error = ""
        st.session_state.monitoring_paused = False
        st.session_state.live_transcript = []
        st.session_state.live_interim_text = ""
        st.session_state.live_detection_ids = []
        st.session_state.live_detections = []
        st.session_state.live_latency_history = []
        st.session_state.live_stream_generation += 1
        st.session_state.live_monitor_started_at = time.monotonic()
        st.session_state.monitoring_active = True
        st.session_state.live_monitor_started = True
        st.rerun()
    if pause_monitoring and st.session_state.monitoring_active:
        st.session_state.monitoring_paused = not st.session_state.monitoring_paused
        if st.session_state.monitoring_paused:
            st.session_state.live_capture.pause()
            st.session_state.live_capture.pcm_sink = None
        else:
            st.session_state.live_capture.resume()
            transcriber = st.session_state.live_transcriber
            st.session_state.live_capture.pcm_sink = (
                transcriber.submit_audio if transcriber else None
            )
        st.rerun()
    if stop_monitoring:
        st.session_state.monitoring_active = False
        st.session_state.monitoring_paused = False
        st.session_state.live_capture.resume()
        if st.session_state.live_transcriber:
            st.session_state.live_transcriber.request_stop()
        st.rerun()

    render_live_microphone()
    if not st.session_state.monitoring_active and st.session_state.live_recording_bytes:
        audio_bytes = st.session_state.live_recording_bytes
        audio_mime_type = "audio/wav"
        audio_suffix = ".wav"
        source_name = "live-microphone-recording.wav"
        duration = st.session_state.live_capture.duration_seconds
        st.caption(f"Buffered recording · {duration:.1f} seconds · kept in memory until this session ends")
        if st.button("Analyze captured audio", type="primary", icon="🔎"):
            st.session_state["analyze_live_capture"] = True

if audio_input_error:
    st.error(audio_input_error)
elif audio_bytes:
    current_audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    previous_record = (st.session_state.get("analysis") or {}).get("analysis_record", {})
    if st.session_state.processed and previous_record.get("audio_sha256") != current_audio_hash:
        st.session_state.update({"processed": False, "analysis": {}, "selected_moment": None})

    current_detections = []
    if st.session_state.processed:
        record = (st.session_state.get("analysis") or {}).get("analysis_record", {})
        if record.get("audio_sha256") == current_audio_hash:
            current_detections = (st.session_state.get("analysis") or {}).get("rule_detections", [])
    st.markdown("#### Recording waveform and event markers")
    render_audio_visualization(audio_bytes, audio_mime_type, current_detections)

# ── Analyze button ────────────────────────────────────────────────────────────
analyze_disabled = not audio_bytes or not engine.is_ready or bool(audio_input_error)
if mode == "🎙️ Live Monitor":
    analyze_clicked = st.session_state.pop("analyze_live_capture", False)
    if not engine.is_ready:
        st.warning("Gemini is not configured. Add GEMINI_API_KEY to the local .env file and restart.")
    elif not audio_bytes and not st.session_state.monitoring_active:
        st.info("Start monitoring, then stop it to prepare a recording for analysis.")
else:
    btn_col, hint_col = st.columns([2, 5])
    with btn_col:
        analyze_clicked = st.button(
            "🚀 Analyze Audio",
            type="primary",
            disabled=analyze_disabled,
            use_container_width=True,
        )
    with hint_col:
        if not engine.is_ready:
            st.warning("⚠️ Gemini not configured. Add `GEMINI_API_KEY` to your `.env` file and restart.")
        elif not audio_bytes:
            st.info("Upload or record audio to begin analysis.")

if analyze_clicked and audio_bytes and engine.is_ready:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name
        with st.spinner("🚀 Sending the complete recording to Google AI Studio for analysis…"):
            analysis = engine.process_audio(
                audio_path,
                language=lang,
                source_name=source_name,
                anonymize_speakers=st.session_state.privacy_anonymize_speakers,
                return_details=True,
            )
            record = analysis.get("analysis_record", {})
            history = list(st.session_state.alert_history)
            detection_history = list(st.session_state.detection_history)
            known_alert_ids = {item["alert_id"] for item in history}
            known_detection_ids = {item["history_id"] for item in detection_history}
            for detection in analysis.get("rule_detections", []):
                alert_id = f"{record.get('audio_sha256', '')}:{detection['detection_id']}"
                if alert_id not in known_detection_ids:
                    detection_history.append({**detection, "history_id": alert_id, "created_at": time.time()})
                    known_detection_ids.add(alert_id)
                if detection.get("is_alert") and alert_id not in known_alert_ids:
                    history.append({**detection, "alert_id": alert_id, "created_at": time.time()})
                    known_alert_ids.add(alert_id)
            st.session_state.update({
                "transcript": analysis["transcript"],
                "event": analysis["main_event"],
                "emotion": analysis["global_emotion"],
                "summary": analysis["summary"],
                "audio_reply": analysis["audio_summary_path"],
                "analysis": analysis,
                "alert_history": history[-MAX_SESSION_EVENT_HISTORY:],
                "detection_history": detection_history[-MAX_SESSION_EVENT_HISTORY:],
                "processed": True,
                "chat_history": [],
                "selected_moment": None,
            })
            st.rerun()
    except Exception as e:
        st.error(f"Analysis failed: {e}")
    finally:
        if st.session_state.privacy_remove_temp_audio and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.processed:
    st.success("✅ Analysis complete")

    analysis = st.session_state.analysis or {}
    t_data = analysis.get("transcript", st.session_state.transcript)
    brief = analysis.get("incident_brief", {})
    important_events = analysis.get("important_events", [])
    priority = analysis.get("attention_priority", "LOW")
    priority_reasons = analysis.get("priority_reasons", [])
    detected_languages = analysis.get("detected_languages", [])
    rule_detections = analysis.get("rule_detections", [])

    render_alert_indicator()
    dashboard_cols = st.columns(4)
    dashboard_cols[0].metric("Monitoring status", "Analysis complete")
    dashboard_cols[1].metric("Transcript segments", len(t_data))
    dashboard_cols[2].metric("AI-reported events", len(important_events))
    dashboard_cols[3].metric("Configured phrase matches", len(rule_detections))

    # ── Incident brief ────────────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("## AURA Incident Brief")
        st.caption("Structured AI assessment of the analyzed audio. Human review remains essential.")

        event_col, time_col, priority_col, confidence_col = st.columns(4)
        event_col.metric("Event", brief.get("event") or st.session_state.event)
        time_col.metric("Time range", brief.get("time_range") or "Not specified")
        with priority_col:
            st.markdown("**Attention Priority**")
            st.markdown(priority_badge(priority), unsafe_allow_html=True)
        confidence = brief.get("confidence")
        confidence_col.metric("AI confidence", f"{confidence}%" if confidence is not None else "Not provided")

        detail_col, context_col = st.columns(2)
        with detail_col:
            st.markdown("**Speakers involved**")
            render_bullet_list(brief.get("speakers", []), "No speaker labels were provided by the analysis.")
            st.markdown("**Important sounds**")
            render_bullet_list(brief.get("important_sounds", []), "No distinct acoustic event was reported.")
            st.caption("Gemini-reported audio observations only; no dedicated sound-event classifier is installed.")
        with context_col:
            st.markdown("**Overall context**")
            st.write(brief.get("overall_context") or "Not specified by the analysis.")
            st.markdown("**Detected languages**")
            render_bullet_list(detected_languages, "Language detection was not provided for this analysis.")

        st.markdown("**Attention-priority reasons**")
        render_bullet_list(priority_reasons, "No separate priority reasons were supplied by the analysis.")
        st.markdown("**Key observations**")
        render_bullet_list(brief.get("key_observations", []), "No additional observations were supplied.")
        st.caption(brief.get("human_review") or "AI-generated assessment. Human review recommended.")

    st.markdown("### Configured Phrase Matches")
    st.caption("Deterministic knowledge-rule matches are separate from Gemini recognition and do not verify real-world events.")
    if engine.rule_matcher is None:
        st.warning(f"Knowledge dataset unavailable: {engine.dataset_error or 'not loaded'}")
    elif rule_detections:
        for index, detection in enumerate(rule_detections):
            with st.container(border=True):
                event_status = "Needs verification" if detection["status"] == "needs_verification" else "Phrase matched"
                st.markdown(f"**{detection['event_name']}** · {event_status}")
                st.caption(
                    f"{format_timestamp(detection['start'])} · {detection['category']} · "
                    f"{detection['priority']} priority · {detection['match_type']} match · "
                    f"Dataset record: {detection['event_id']}"
                )
                if detection["match_type"] == "fuzzy":
                    st.caption(f"Text similarity: {detection['match_score']:.1%} (not event confidence)")
                elif detection["match_type"] == "exact":
                    st.caption("Exact configured phrase match; no probabilistic confidence is inferred.")
                st.write("**Original transcript evidence**")
                st.code(detection["original_evidence"], language=None)
                st.write(f"**Why it matched:** {detection['explanation']}")
                st.write(f"**Recommended review action:** {detection['recommended_action']}")
                st.caption(
                    f"Evidence requirements: {', '.join(detection['evidence_requirements'])}. "
                    f"Source: {detection['source_reference']}"
                )
                if detection["status"] == "needs_verification":
                    st.warning("Needs verification. This context is not treated as a positive alert.")
                segment_index = detection.get("segment_index")
                if st.button("Review transcript moment", key=f"review_rule_match_{index}", icon="🔎"):
                    if isinstance(segment_index, int) and 0 <= segment_index < len(t_data):
                        st.session_state.selected_moment = t_data[segment_index]
                        st.rerun()
    else:
        st.info("No configured phrase rules matched this transcript.")

    # ── Executive summary and audio response ──────────────────────────────────
    with st.container(border=True):
        st.markdown("### 📋 AI Executive Summary")
        sum_col, play_col = st.columns([4, 1])
        with sum_col:
            st.info(brief.get("summary") or st.session_state.summary or "No summary was returned.")
            if analysis.get("original_language_summary"):
                st.markdown("**Original-language summary**")
                st.write(analysis["original_language_summary"])
        with play_col:
            if st.session_state.audio_reply:
                st.markdown("**🔊 Listen**")
                st.audio(st.session_state.audio_reply)
            else:
                st.caption("Audio summary unavailable.")

    # ── Important findings ────────────────────────────────────────────────────
    st.markdown("### Important Findings")
    if important_events:
        for index, event in enumerate(important_events):
            event_title = event.get("event") or "Important transcript moment"
            event_time = f"{format_timestamp(event.get('start'))} – {format_timestamp(event.get('end'))}"
            finding_col, action_col = st.columns([5, 1])
            with finding_col:
                st.markdown(f"**{event_time} · {event_title}**")
                st.caption(
                    f"{event.get('speaker') or 'Speaker not specified'} · "
                    f"{event.get('tone') or 'Tone not specified'} · "
                    f"{event.get('importance', 'HIGH')} priority"
                )
            with action_col:
                if st.button("Select", key=f"important_event_{index}", use_container_width=True):
                    segment_index = event.get("segment_index")
                    if isinstance(segment_index, int) and 0 <= segment_index < len(t_data):
                        st.session_state.selected_moment = t_data[segment_index]
                    else:
                        st.session_state.selected_moment = event
    else:
        st.info("No important moments were marked by the analysis.")

    # ── Timeline, transcript, and QnA ─────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Smart Timeline", "📝 Transcript & Matches", "🤖 Ask AURA",
        "Knowledge Dataset", "Event History", "Evaluation",
    ])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎭 Global Emotion", st.session_state.emotion)
        m2.metric("🔊 Key Event", st.session_state.event)
        m3.metric("📌 Segments", len(t_data))
        m4.metric("Phrase-rule hits", len(rule_detections))

        if t_data:
            df = pd.DataFrame(t_data)
            for column, default in {
                "speaker": "Unknown speaker",
                "text": "",
                "tone": "Not specified",
                "importance": "LOW",
                "event": "",
                "confidence": None,
                "is_urgent": False,
                "flag_reasons": [],
            }.items():
                if column not in df:
                    df[column] = default
            df["start"] = pd.to_numeric(df["start"], errors="coerce").fillna(0)
            df["end"] = pd.to_numeric(df["end"], errors="coerce").fillna(df["start"])
            df["importance"] = df["importance"].fillna("LOW").astype(str).str.upper()
            df["flag_reasons_display"] = df["flag_reasons"].apply(
                lambda reasons: "; ".join(reasons) if isinstance(reasons, list) else str(reasons or "")
            )

            base = pd.Timestamp("2000-01-01")
            df["start_dt"] = df["start"].apply(lambda value: base + pd.Timedelta(seconds=value))
            df["end_dt"] = df["end"].apply(lambda value: base + pd.Timedelta(seconds=value))
            fig = px.timeline(
                df,
                x_start="start_dt",
                x_end="end_dt",
                y="speaker",
                color="importance",
                text="text",
                color_discrete_map={"HIGH": "#c3342d", "MEDIUM": "#a66a00", "LOW": "#28759b"},
                hover_data=["event", "tone", "confidence", "is_urgent", "flag_reasons_display"],
                template="plotly_white",
                title="Smart Event Timeline",
            )
            fig.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                font_color="#1c2922",
                title_font_size=14,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Time (mm:ss)",
                yaxis_title="Speaker",
                legend_title="Importance",
                height=350,
            )
            fig.update_xaxes(tickformat="%M:%S")
            fig.update_traces(textposition="inside", insidetextanchor="middle")
            st.plotly_chart(fig, use_container_width=True)

            moment_options = {"No moment selected": None}
            for index, segment in enumerate(t_data):
                moment_options[f"{index + 1}. {moment_label(segment)}"] = segment
            selected_option = st.selectbox(
                "Select a timeline moment for explainability and Ask AURA",
                list(moment_options),
                key="timeline_moment_selector",
            )
            if moment_options[selected_option] is not None:
                st.session_state.selected_moment = moment_options[selected_option]
        else:
            st.info("No transcript data is available for the timeline.")

    with tab2:
        st.markdown("##### Transcript Segments")
        st.caption("Gemini-recognized transcript with configured phrase evidence highlighted.")
        if t_data:
            for segment_index, segment in enumerate(t_data):
                matches = [item for item in rule_detections if item.get("segment_index") == segment_index]
                st.markdown(
                    f"**{format_timestamp(segment.get('start'))} · "
                    f"{html_lib.escape(segment.get('speaker') or 'Unknown speaker')}**",
                )
                st.markdown(highlighted_evidence(segment.get("text", ""), matches), unsafe_allow_html=True)
                if matches:
                    labels = ", ".join(
                        "Needs verification" if item["status"] == "needs_verification" else item["event_name"]
                        for item in matches
                    )
                    st.caption(labels)
            grid_data = pd.DataFrame(t_data).copy()
            if "flag_reasons" in grid_data:
                grid_data["flag_reasons"] = grid_data["flag_reasons"].apply(
                    lambda reasons: "; ".join(reasons) if isinstance(reasons, list) else reasons
                )
            st.data_editor(grid_data, use_container_width=True, num_rows="dynamic")
        else:
            st.info("No transcript data available.")

    selected_moment = st.session_state.selected_moment
    if selected_moment:
        with st.container(border=True):
            st.markdown("### Selected Moment")
            st.markdown(f"**{moment_label(selected_moment)}**")
            if selected_moment.get("text"):
                st.write(selected_moment["text"])
            selected_confidence = selected_moment.get("confidence")
            st.caption(
                f"Tone: {selected_moment.get('tone') or 'Not specified'} · "
                f"Importance: {selected_moment.get('importance') or 'Not specified'}"
                + (f" · AI confidence: {selected_confidence}%" if selected_confidence is not None else "")
            )

            is_flagged = (
                selected_moment.get("importance") in {"HIGH", "MEDIUM"}
                or bool(selected_moment.get("is_urgent"))
            )
            if is_flagged:
                st.markdown("#### Why was this flagged?")
                reasons = selected_moment.get("flag_reasons", [])
                if not reasons:
                    reasons = [
                        event_reason
                        for event in important_events
                        if event.get("start") == selected_moment.get("start")
                        and event.get("end") == selected_moment.get("end")
                        for event_reason in event.get("flag_reasons", [])
                    ]
                render_bullet_list(
                    reasons,
                    "The model marked this moment as important but did not provide separate signal-level reasons.",
                )
                st.caption("AI-generated assessment. Human review recommended.")
            else:
                st.caption("This moment was not marked as important or urgent by the analysis.")

    with tab3:
        st.markdown("##### 🤖 Ask AURA — Audio Intelligence Assistant")
        if selected_moment:
            st.info(
                "Moment mode is active for "
                f"{format_timestamp(selected_moment.get('start'))} – {format_timestamp(selected_moment.get('end'))}."
            )
            suggestions = [
                "What happened during this moment?",
                "Who was speaking here?",
                "What changed immediately before and after this moment?",
                "Why is this segment important?",
            ]
        else:
            st.caption("Global mode is active. Ask about the complete analyzed audio.")
            suggestions = [
                "What happened in this audio?",
                "When did the important event occur?",
                "What did Speaker A say?",
                "Which segment appears most urgent?",
            ]

        st.markdown("**Suggested questions:**")
        suggestion_cols = st.columns(len(suggestions))

        def ask_aura(question):
            st.chat_message("user").write(question)
            with st.spinner("Thinking…"):
                answer = engine.answer_question(
                    t_data,
                    question,
                    selected_segment=selected_moment,
                    incident_brief=brief,
                    important_events=important_events,
                    detected_languages=detected_languages,
                )
            st.chat_message("assistant").write(answer)
            if not st.session_state.privacy_do_not_retain_history:
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ])

        for index, suggestion in enumerate(suggestions):
            if suggestion_cols[index].button(suggestion, key=f"suggestion_{index}", use_container_width=True):
                ask_aura(suggestion)

        st.markdown("---")
        for message in st.session_state.chat_history:
            st.chat_message(message["role"]).write(message["content"])

        if question := st.chat_input("Ask about specific details in the audio…"):
            ask_aura(question)

    with tab4:
        st.markdown("##### Configured knowledge rules")
        st.caption("Rules are local matching configuration, not a sound-classification model or external safety standard.")
        if engine.rule_matcher is None:
            st.error(engine.dataset_error or "Knowledge dataset unavailable.")
        else:
            rules = engine.rule_matcher.dataset["rules"]
            search_term = st.text_input("Search rule name, phrase, category, or language", key="knowledge_rule_search")
            visible_rules = [
                rule for rule in rules
                if search_term.casefold() in json.dumps(rule, ensure_ascii=False).casefold()
            ]
            st.caption(f"Showing {len(visible_rules)} of {len(rules)} configured rules")
            for rule in visible_rules:
                with st.expander(f"{rule['event_name']} · {rule['priority']} · {rule['category']}"):
                    st.write(f"**Record:** {rule['event_id']} · **Type:** {rule['event_type']}")
                    st.write(f"**Trigger phrases:** {', '.join(rule['trigger_phrases'])}")
                    st.write(f"**Variants:** {', '.join(rule['phrase_variants']) or 'None'}")
                    st.write(f"**Negative examples:** {', '.join(rule['negative_phrases']) or 'None'}")
                    st.write(f"**Context exclusions:** {', '.join(rule['context_exclusions']) or 'None'}")
                    st.write(f"**Ambiguous context:** {', '.join(rule['ambiguity_phrases']) or 'None'}")
                    st.write(f"**Recommended review action:** {rule['recommended_action']}")
            st.caption(
                f"Fuzzy matching: {'enabled' if engine.rule_matcher.fuzzy_enabled else 'disabled'} · "
                f"threshold {engine.rule_matcher.fuzzy_threshold:.0%} · semantic matching unavailable"
            )

    with tab5:
        st.markdown("##### Session event history")
        if st.session_state.detection_history:
            acknowledged_ids = set(st.session_state.acknowledged_alerts)
            history_rows = []
            for item in reversed(st.session_state.detection_history):
                if item.get("is_alert"):
                    alert_status = "Acknowledged" if item["history_id"] in acknowledged_ids else "Alert"
                else:
                    alert_status = "Needs verification"
                history_rows.append({
                    "Timestamp": format_timestamp(item.get("start", 0)),
                    "Category": item.get("category"),
                    "Phrase": item.get("matched_phrase"),
                    "Priority": item.get("priority"),
                    "Status": alert_status,
                    "Explanation": item.get("explanation"),
                })
            st.dataframe(pd.DataFrame(history_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No configured phrase detections have been recorded in this session.")

    with tab6:
        st.markdown("##### Matcher evaluation")
        st.caption("These metrics are computed from the local labeled evaluation set; live detections are not ground truth.")
        try:
            if engine.rule_matcher is None:
                raise ValueError(engine.dataset_error or "Knowledge dataset unavailable.")
            evaluation_data = load_evaluation_dataset(
                os.path.join(parent_dir, "training", "evaluation_cases.json")
            )
            evaluation = evaluate_matcher(engine.rule_matcher, evaluation_data)
            def percentage(value):
                return f"{value:.1%}" if value is not None else "Not measurable"

            metric_cols = st.columns(4)
            metric_cols[0].metric("Precision", percentage(evaluation["precision"]))
            metric_cols[1].metric("Recall", percentage(evaluation["recall"]))
            metric_cols[2].metric("F1", percentage(evaluation["f1"]))
            metric_cols[3].metric("Matcher latency / case", f"{evaluation['matcher_latency_ms_per_case']:.3f} ms")
            rate_cols = st.columns(2)
            rate_cols[0].metric("False-positive rate", percentage(evaluation["false_positive_rate"]))
            rate_cols[1].metric("Missed-event rate", percentage(evaluation["missed_event_rate"]))
            st.write(
                f"**Labeled cases:** {evaluation['case_count']} · **Rule classes:** "
                f"{evaluation['event_class_count']} · **TP:** {evaluation['true_positive']} · "
                f"**FP:** {evaluation['false_positive']} · **FN:** {evaluation['false_negative']} · "
                f"**TN:** {evaluation['true_negative']}"
            )
            st.dataframe(pd.DataFrame(evaluation["case_results"]), use_container_width=True, hide_index=True)
            st.caption(
                "Matcher latency covers only local phrase-rule evaluation on this machine and labeled set. "
                "It excludes microphone capture, Gemini network/API time, and end-to-end transcript latency. "
                "This small example set is not a production performance guarantee."
            )
        except (OSError, ValueError) as exc:
            st.error(f"Evaluation could not run: {exc}")

    # ── Analysis record ───────────────────────────────────────────────────────
    record = analysis.get("analysis_record", {})
    with st.expander("AURA Analysis Record"):
        st.caption("Non-sensitive analysis metadata and an audio file integrity reference.")
        record_left, record_right = st.columns(2)
        with record_left:
            st.write(f"**Analysis ID:** {record.get('analysis_id', 'Not available')}")
            st.write(f"**File:** {record.get('filename', 'Not available')}")
            duration = record.get("duration_seconds")
            st.write(
                "**Duration:** "
                + (format_timestamp(duration) if duration is not None else "Not available")
            )
            st.caption(f"Duration source: {record.get('duration_source', 'Not available')}")
            st.write(f"**Analysis time:** {record.get('analysis_timestamp', 'Not available')}")
        with record_right:
            st.write(f"**AI model:** {record.get('ai_model', GEMINI_MODEL)}")
            st.write(f"**Status:** {record.get('status', 'Not available')}")
            st.write(f"**Important event count:** {record.get('important_event_count', 0)}")
            st.write(f"**Overall attention priority:** {record.get('attention_priority', priority)}")

        st.markdown("**Audio file integrity hash (SHA-256)**")
        if record.get("audio_sha256"):
            st.code(record["audio_sha256"], language=None)
        else:
            st.caption("Integrity hash unavailable.")
        st.caption("This hash is an integrity reference, not a statement of evidentiary or legal status.")
