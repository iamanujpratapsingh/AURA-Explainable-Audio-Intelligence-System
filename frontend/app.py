import streamlit as st
import plotly.express as px
import tempfile
import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from backend.aura_engine import AuraEngine, GEMINI_MODEL
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
    .stApp { background-color: #0e1117; }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.3rem;
    }

    /* Status pill */
    .status-ok {
        display: inline-block;
        background: #052e16;
        color: #4ade80;
        border: 1px solid #166534;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-err {
        display: inline-block;
        background: #2d0a0a;
        color: #f87171;
        border: 1px solid #7f1d1d;
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
    .priority-high { background: #451a1a; border: 1px solid #dc2626; color: #fca5a5; }
    .priority-medium { background: #422006; border: 1px solid #d97706; color: #fcd34d; }
    .priority-low { background: #052e16; border: 1px solid #16a34a; color: #86efac; }

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

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in [("processed", False), ("chat_history", []),
                      ("transcript", []), ("event", ""), ("emotion", ""),
                      ("summary", ""), ("audio_reply", None), ("analysis", {}),
                      ("selected_moment", None), ("privacy_remove_temp_audio", True),
                      ("privacy_do_not_retain_history", False),
                      ("privacy_anonymize_speakers", True)]:
    if key not in st.session_state:
        st.session_state[key] = default

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
        ["📁 Upload File", "🎤 Record Voice"],
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

if mode == "📁 Upload File":
    st.markdown('<p class="section-label">Upload an audio file (WAV, MP3, M4A)</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a"],
        label_visibility="collapsed",
    )
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[-1]) as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name
        source_name = uploaded.name
        st.audio(audio_path)

elif mode == "🎤 Record Voice":
    st.markdown('<p class="section-label">Record audio using your microphone</p>', unsafe_allow_html=True)
    recorded = st.audio_input("Record", label_visibility="collapsed")
    if recorded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(recorded.getvalue())
            audio_path = tmp.name
        source_name = "microphone-recording.wav"
        st.audio(audio_path)

# ── Analyze button ────────────────────────────────────────────────────────────
analyze_disabled = not audio_path or not engine.is_ready
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
    elif not audio_path:
        st.info("Upload or record audio to begin analysis.")

if analyze_clicked and audio_path and engine.is_ready:
    with st.spinner("🚀 Uploading to Google AI Studio and processing…"):
        try:
            analysis = engine.process_audio(
                audio_path,
                language=lang,
                source_name=source_name,
                anonymize_speakers=st.session_state.privacy_anonymize_speakers,
                return_details=True,
            )
            st.session_state.update({
                "transcript": analysis["transcript"],
                "event": analysis["main_event"],
                "emotion": analysis["global_emotion"],
                "summary": analysis["summary"],
                "audio_reply": analysis["audio_summary_path"],
                "analysis": analysis,
                "processed": True,
                "chat_history": [],
                "selected_moment": None,
            })
            st.rerun()
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            if st.session_state.privacy_remove_temp_audio and os.path.exists(audio_path):
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
    tab1, tab2, tab3 = st.tabs(["📊 Smart Timeline", "📝 Transcript Data", "🤖 Ask AURA"])

    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎭 Global Emotion", st.session_state.emotion)
        m2.metric("🔊 Key Event", st.session_state.event)
        m3.metric("📌 Segments", len(t_data))
        m4.metric("Important moments", len(important_events))

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
                color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#3b82f6"},
                hover_data=["event", "tone", "confidence", "is_urgent", "flag_reasons_display"],
                template="plotly_dark",
                title="Smart Event Timeline",
            )
            fig.update_layout(
                paper_bgcolor="#1a1d27",
                plot_bgcolor="#1a1d27",
                font_color="#e5e7eb",
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
        st.caption("All segments returned by Gemini. You can edit values inline for review.")
        if t_data:
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
