import os
import json
import time
import uuid
import hashlib
import io
import wave
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from gtts import gTTS
from backend.knowledge_matcher import KnowledgeMatcher, load_dataset

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

PRIORITY_LEVELS = {"LOW", "MEDIUM", "HIGH"}
TRANSIENT_PROVIDER_CODES = {"429", "500", "502", "503", "504"}


def _as_string_list(value):
    """Return a compact list of non-empty strings from model output."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_confidence(value):
    """Normalise model confidence to an optional 0-100 percentage."""
    if value is None or value == "":
        return None
    confidence = _as_float(value, None)
    if confidence is None:
        return None
    if 0 <= confidence <= 1:
        confidence *= 100
    return max(0, min(100, round(confidence)))


def _format_timestamp(seconds):
    seconds = max(0, round(_as_float(seconds)))
    minutes, remaining = divmod(seconds, 60)
    hours, seconds = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{seconds:02d}:{remaining:02d}"
    return f"{seconds:02d}:{remaining:02d}"


def _normalise_priority(value):
    priority = str(value or "").strip().upper()
    return priority if priority in PRIORITY_LEVELS else None


class KnowledgeBase:
    def __init__(self, db_path="training/knowledge_base.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def save_entry(self, data, notes):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": str(datetime.now()),
            "data": data,
            "notes": notes,
        }
        try:
            with open(self.db_path, "r+", encoding="utf-8") as f:
                try:
                    db = json.load(f)
                except Exception:
                    db = []
                db.append(entry)
                f.seek(0)
                json.dump(db, f, indent=4)
                f.truncate()
            return "Knowledge Base Updated."
        except Exception as e:
            return f"Error saving to knowledge base: {e}"


class AuraEngine:
    def __init__(self):
        self.model_name = GEMINI_MODEL
        self.memory = KnowledgeBase()
        self._gemini_ready = False
        self._init_error = None
        self._client = None
        self._live_client = None
        dataset_path = Path(__file__).resolve().parents[1] / "training" / "alert_rules.json"
        self.rule_matcher = None
        self.dataset_error = None
        try:
            dataset = load_dataset(dataset_path)
            fuzzy_enabled = os.getenv("AURA_FUZZY_MATCHING", "false").strip().lower() in {"1", "true", "yes"}
            self.rule_matcher = KnowledgeMatcher(dataset, fuzzy_enabled=fuzzy_enabled)
        except (OSError, ValueError) as exc:
            self.dataset_error = str(exc)

        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            self._init_error = "Gemini configuration missing. Please add GEMINI_API_KEY to your .env file."
            return

        try:
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._live_client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(timeout=20_000),
            )
            self._gemini_ready = True
        except Exception as e:
            self._init_error = f"Gemini initialization failed: {e}"

    @property
    def is_ready(self):
        return self._gemini_ready

    @property
    def status_message(self):
        if self._gemini_ready:
            return f"Connected - {self.model_name}"
        return self._init_error or "Configuration error."

    @property
    def live_client(self):
        return self._live_client if self._gemini_ready else None

    def match_transcript(self, transcript):
        if self.rule_matcher is None:
            return []
        return self.rule_matcher.match_segments(transcript)

    @staticmethod
    def _is_transient_provider_error(error):
        code = getattr(error, "code", None)
        if callable(code):
            code = code()
        if str(code) in TRANSIENT_PROVIDER_CODES:
            return True
        message = str(error).upper()
        return any(
            marker in message
            for marker in (
                "429",
                "500 INTERNAL",
                "502 BAD GATEWAY",
                "503 UNAVAILABLE",
                "504 GATEWAY",
                "RESOURCE_EXHAUSTED",
                "HIGH DEMAND",
                "TEMPORAR",
            )
        )

    def _generate_content_with_retry(self, **kwargs):
        try:
            retry_count = max(0, int(os.getenv("AURA_PROVIDER_RETRIES", "2")))
        except ValueError:
            retry_count = 2

        for attempt in range(retry_count + 1):
            try:
                return self._client.models.generate_content(**kwargs)
            except Exception as exc:
                if attempt >= retry_count or not self._is_transient_provider_error(exc):
                    raise
                time.sleep(min(2 ** attempt, 8))

    @staticmethod
    def _read_validated_audio(audio_path):
        extension = os.path.splitext(audio_path)[-1].lower()
        mime_map = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4"}
        if extension not in mime_map:
            raise ValueError("Unsupported audio format. Use WAV, MP3, or M4A.")
        try:
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
        except OSError as exc:
            raise ValueError(f"Could not read audio file: {exc}") from exc
        if not audio_bytes:
            raise ValueError("Audio file is empty.")
        maximum_bytes = int(float(os.getenv("AURA_MAX_AUDIO_MB", "50")) * 1024 * 1024)
        if maximum_bytes <= 0 or len(audio_bytes) > maximum_bytes:
            raise ValueError(f"Audio file exceeds the configured {maximum_bytes // (1024 * 1024)} MB limit.")

        if extension == ".wav":
            try:
                with wave.open(io.BytesIO(audio_bytes), "rb") as wav_file:
                    if wav_file.getnframes() <= 0 or wav_file.getframerate() <= 0:
                        raise ValueError("WAV file contains no audio frames.")
            except (wave.Error, EOFError) as exc:
                raise ValueError("WAV file is malformed or incomplete.") from exc
        elif extension == ".m4a" and b"ftyp" not in audio_bytes[:16]:
            raise ValueError("M4A file does not contain a valid MP4 audio header.")
        elif extension == ".mp3":
            has_id3 = audio_bytes.startswith(b"ID3")
            has_frame_sync = any(
                audio_bytes[index] == 0xFF and audio_bytes[index + 1] & 0xE0 == 0xE0
                for index in range(min(len(audio_bytes) - 1, 4096))
            )
            if not has_id3 and not has_frame_sync:
                raise ValueError("MP3 file does not contain a recognized ID3 tag or MPEG audio frame.")
        return audio_bytes, mime_map[extension]

    def generate_audio_response(self, text):
        if not text:
            return None
        try:
            tts = gTTS(text=text, lang="en")
            filename = f"response_{uuid.uuid4().hex[:6]}.mp3"
            path = os.path.abspath(filename)
            tts.save(path)
            return path
        except Exception:
            return None

    @staticmethod
    def _parse_model_json(raw_text):
        """Parse Gemini JSON while tolerating an accidental markdown fence."""
        if not raw_text:
            raise ValueError("Gemini returned an empty analysis response.")

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        # Some models occasionally add one short sentence before or after JSON.
        # Restrict parsing to the object rather than evaluating arbitrary text.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError("Gemini returned malformed JSON. Please try again.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Gemini returned an analysis in an unexpected format.")
        return parsed

    @staticmethod
    def _duration_from_audio(audio_path, transcript):
        """Use WAV metadata when possible, otherwise the final transcript end time."""
        try:
            with wave.open(audio_path, "rb") as audio_file:
                frame_rate = audio_file.getframerate()
                if frame_rate:
                    return round(audio_file.getnframes() / frame_rate, 2), "audio file metadata"
        except (wave.Error, OSError, EOFError):
            pass

        ends = [_as_float(segment.get("end"), 0) for segment in transcript]
        duration = max(ends, default=0)
        return (round(duration, 2), "final transcript timestamp") if duration else (None, "unavailable")

    @staticmethod
    def _anonymize_speakers(analysis):
        """Replace only displayed speaker values; transcript content is unchanged."""
        speaker_map = {}

        def label_for(speaker):
            original = str(speaker or "Unknown speaker").strip() or "Unknown speaker"
            if original not in speaker_map:
                index = len(speaker_map)
                suffix = chr(ord("A") + index) if index < 26 else str(index + 1)
                speaker_map[original] = f"Speaker {suffix}"
            return speaker_map[original]

        for segment in analysis["transcript"]:
            segment["speaker"] = label_for(segment.get("speaker"))
        for event in analysis["important_events"]:
            if event.get("speaker"):
                event["speaker"] = label_for(event["speaker"])

        brief_speakers = analysis["incident_brief"].get("speakers", [])
        analysis["incident_brief"]["speakers"] = [label_for(speaker) for speaker in brief_speakers]

    @staticmethod
    def _normalise_analysis(data):
        """Apply safe defaults so partial model output remains usable in the UI."""
        raw_transcript = data.get("transcript") if isinstance(data.get("transcript"), list) else []
        transcript = []
        for index, segment in enumerate(raw_transcript):
            if not isinstance(segment, dict):
                continue

            start = max(0, _as_float(segment.get("start"), 0))
            end = max(start, _as_float(segment.get("end"), start))
            urgent = segment.get("is_urgent", False)
            if isinstance(urgent, str):
                urgent = urgent.strip().lower() in {"true", "yes", "1"}
            importance = _normalise_priority(segment.get("importance"))
            if importance is None:
                importance = "HIGH" if urgent else "LOW"

            transcript.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "speaker": str(segment.get("speaker") or "Unknown speaker").strip(),
                    "text": str(segment.get("text") or "").strip(),
                    "tone": str(segment.get("tone") or "Not specified").strip(),
                    "is_urgent": bool(urgent),
                    "importance": importance,
                    "confidence": _as_confidence(segment.get("confidence")),
                    "event": str(segment.get("event") or segment.get("event_type") or "").strip(),
                    "flag_reasons": _as_string_list(segment.get("flag_reasons")),
                    "segment_index": index,
                }
            )

        priority = _normalise_priority(data.get("attention_priority"))
        priority_reasons = _as_string_list(data.get("priority_reasons"))
        if priority is None:
            high_segments = [
                segment for segment in transcript
                if segment["importance"] == "HIGH" or segment["is_urgent"]
            ]
            medium_segments = [segment for segment in transcript if segment["importance"] == "MEDIUM"]
            if high_segments:
                priority = "HIGH"
                if not priority_reasons:
                    priority_reasons = [
                        "One or more transcript segments were marked urgent or high importance by the AI analysis."
                    ]
            elif medium_segments:
                priority = "MEDIUM"
                if not priority_reasons:
                    priority_reasons = [
                        "One or more transcript segments were marked medium importance by the AI analysis."
                    ]
            else:
                priority = "LOW"

        raw_events = data.get("important_events") if isinstance(data.get("important_events"), list) else []
        important_events = []
        for event in raw_events:
            if not isinstance(event, dict):
                continue
            start = max(0, _as_float(event.get("start"), 0))
            end = max(start, _as_float(event.get("end"), start))
            importance = _normalise_priority(event.get("importance")) or "HIGH"
            urgent = event.get("is_urgent", importance == "HIGH")
            if isinstance(urgent, str):
                urgent = urgent.strip().lower() in {"true", "yes", "1"}
            important_events.append(
                {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "event": str(event.get("event") or event.get("event_type") or "").strip(),
                    "speaker": str(event.get("speaker") or "").strip(),
                    "tone": str(event.get("tone") or "").strip(),
                    "is_urgent": bool(urgent),
                    "importance": importance,
                    "confidence": _as_confidence(event.get("confidence")),
                    "flag_reasons": _as_string_list(event.get("flag_reasons")),
                    "segment_index": event.get("segment_index"),
                }
            )

        # Older responses have no important_events list. Expose model-marked
        # transcript segments without inventing a separate sound event.
        if not important_events:
            for segment in transcript:
                if segment["importance"] == "HIGH" or segment["is_urgent"]:
                    important_events.append(
                        {
                            key: segment[key]
                            for key in (
                                "start", "end", "event", "speaker", "tone", "is_urgent",
                                "importance", "confidence", "flag_reasons", "segment_index",
                            )
                        }
                    )

        raw_brief = data.get("incident_brief") if isinstance(data.get("incident_brief"), dict) else {}
        brief_speakers = _as_string_list(raw_brief.get("speakers"))
        if not brief_speakers:
            brief_speakers = list(dict.fromkeys(
                segment["speaker"] for segment in transcript
                if segment["speaker"] and segment["speaker"] != "Unknown speaker"
            ))

        important_starts = [event["start"] for event in important_events]
        important_ends = [event["end"] for event in important_events]
        inferred_time_range = ""
        if important_starts and important_ends:
            inferred_time_range = f"{_format_timestamp(min(important_starts))} – {_format_timestamp(max(important_ends))}"

        incident_brief = {
            "event": str(raw_brief.get("event") or data.get("main_event") or "Not specified").strip(),
            "time_range": str(raw_brief.get("time_range") or inferred_time_range).strip(),
            "speakers": brief_speakers,
            "important_sounds": _as_string_list(raw_brief.get("important_sounds")),
            "overall_context": str(raw_brief.get("overall_context") or data.get("overall_context") or "").strip(),
            "key_observations": _as_string_list(raw_brief.get("key_observations")),
            "confidence": _as_confidence(raw_brief.get("confidence") or data.get("confidence")),
            "summary": str(raw_brief.get("summary") or data.get("summary") or "").strip(),
            "human_review": str(
                raw_brief.get("human_review") or "AI-generated assessment. Human review recommended."
            ).strip(),
        }

        return {
            "global_emotion": str(data.get("global_emotion") or "Not specified").strip(),
            "main_event": str(data.get("main_event") or "Not specified").strip(),
            "attention_priority": priority,
            "priority_reasons": priority_reasons,
            "detected_languages": _as_string_list(data.get("detected_languages")),
            "transcript": transcript,
            "important_events": important_events,
            "summary": str(data.get("summary") or "").strip(),
            "original_language_summary": str(data.get("original_language_summary") or "").strip(),
            "incident_brief": incident_brief,
        }

    def _build_analysis_record(self, audio_path, audio_bytes, transcript, source_name, priority, event_count):
        duration_seconds, duration_source = self._duration_from_audio(audio_path, transcript)
        return {
            "analysis_id": f"AURA-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:8].upper()}",
            "filename": os.path.basename(source_name or audio_path),
            "duration_seconds": duration_seconds,
            "duration_source": duration_source,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "ai_model": self.model_name,
            "status": "Analysis completed",
            "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
            "important_event_count": event_count,
            "attention_priority": priority,
        }

    def process_audio(
        self,
        audio_path,
        language="English",
        source_name=None,
        anonymize_speakers=False,
        return_details=False,
    ):
        """Analyze once and optionally return the dashboard's extended schema.

        The default tuple preserves the pre-existing public return contract.
        """
        audio_bytes, mime_type = self._read_validated_audio(audio_path)
        if not self._gemini_ready:
            raise ValueError(self._init_error or "Gemini is not configured.")

        prompt = f"""You are AURA, an AI-assisted audio intelligence system. Analyze this audio file.
The user-selected spoken language is: {language}. Identify languages from the audio where the evidence supports it.

Return ONLY a valid JSON object. Do NOT include markdown, code fences, or any text outside the JSON.
The JSON must match this exact structure:
{{
    "global_emotion": "Observed overall tone, or Not specified",
    "main_event": "Main observed audio event, or Not specified",
    "attention_priority": "LOW, MEDIUM, or HIGH",
    "priority_reasons": ["Short reasons grounded in observed audio or transcript evidence"],
    "detected_languages": ["Languages actually detected in the audio, not merely the selected language"],
    "original_language_summary": "Optional concise summary in the dominant detected language; use an empty string if not practical",
    "transcript": [
        {{
            "start": 0.0,
            "end": 2.5,
            "speaker": "Speaker A",
            "text": "The spoken text here",
            "tone": "Specific emotion of this segment",
            "is_urgent": false,
            "importance": "LOW, MEDIUM, or HIGH",
            "confidence": 82,
            "event": "Observed event at this moment, if any",
            "flag_reasons": ["Evidence-based reasons only when the segment is important or urgent"]
        }}
    ],
    "important_events": [
        {{
            "start": 12.0,
            "end": 18.0,
            "event": "Observed important event",
            "speaker": "Speaker A",
            "tone": "Observed tone",
            "is_urgent": true,
            "importance": "HIGH",
            "confidence": 82,
            "flag_reasons": ["Evidence-based reasons"]
        }}
    ],
    "incident_brief": {{
        "event": "Observed incident or main event",
        "time_range": "00:12 – 00:18 when supported by timestamps",
        "speakers": ["Speaker A"],
        "important_sounds": ["Only acoustic/sound events actually observed"],
        "overall_context": "Brief, cautious context grounded in the audio",
        "key_observations": ["Observed facts and clearly labelled AI assessments"],
        "confidence": 82,
        "summary": "English executive summary",
        "human_review": "AI-generated assessment. Human review recommended."
    }},
    "summary": "A concise English executive summary of the audio content."
}}

Rules: do not identify people by name; use neutral speaker labels such as Speaker A. Do not invent
acoustic events, urgency, or causes. Keep observations distinct from interpretations. If an item is
not supported by the audio, use an empty list, empty string, or "Not specified" rather than guessing.
Confidence is an AI estimate (0-100), not a guarantee. Do not characterize anyone as criminal,
dangerous, or responsible for a real-world event."""

        try:
            response = self._generate_content_with_retry(
                model=self.model_name,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )

            analysis = self._normalise_analysis(self._parse_model_json(response.text))
            if anonymize_speakers:
                self._anonymize_speakers(analysis)
            analysis["rule_detections"] = self.match_transcript(analysis["transcript"])
            analysis["knowledge_dataset_status"] = (
                "Loaded" if self.rule_matcher is not None else f"Unavailable: {self.dataset_error or 'dataset not loaded'}"
            )

            analysis["analysis_record"] = self._build_analysis_record(
                audio_path=audio_path,
                audio_bytes=audio_bytes,
                transcript=analysis["transcript"],
                source_name=source_name,
                priority=analysis["attention_priority"],
                event_count=len(analysis["important_events"]),
            )
            analysis["audio_summary_path"] = self.generate_audio_response(analysis["summary"])

            if return_details:
                return analysis
            return (
                analysis["transcript"],
                analysis["main_event"],
                analysis["global_emotion"],
                analysis["summary"],
                analysis["audio_summary_path"],
            )

        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Audio processing failed ({self.model_name}): {exc}") from exc

    def answer_question(
        self,
        transcript_data,
        question,
        selected_segment=None,
        incident_brief=None,
        important_events=None,
        detected_languages=None,
    ):
        if not self._gemini_ready:
            return self._init_error or "Gemini is not configured."
        try:
            selected_segment = selected_segment if isinstance(selected_segment, dict) else None
            if selected_segment:
                selected_index = selected_segment.get("segment_index")
                nearby_segments = []
                if isinstance(selected_index, int):
                    nearby_segments = transcript_data[max(0, selected_index - 1): selected_index + 2]
                context = {
                    "selected_moment": selected_segment,
                    "adjacent_segments": nearby_segments,
                    "incident_brief": incident_brief or {},
                    "important_events": important_events or [],
                }
                scope_instruction = (
                    "Answer in MOMENT MODE. Focus on the selected timestamp range and adjacent context only. "
                    "State clearly when the audio does not support an answer."
                )
            else:
                context = {
                    "transcript": transcript_data,
                    "incident_brief": incident_brief or {},
                    "important_events": important_events or [],
                    "detected_languages": detected_languages or [],
                }
                scope_instruction = "Answer in GLOBAL MODE using the full analyzed audio context."

            prompt = f"""You are AURA, an AI-assisted audio intelligence assistant.

{scope_instruction}
Analysis context:
{json.dumps(context, ensure_ascii=False, indent=2)}

User question: {question}

Answer helpfully and precisely. Cite timestamps and neutral speaker labels where relevant. Ground every
claim in the supplied analysis; distinguish observed information from an AI assessment, do not make
crime or safety determinations, and close with a concise reminder that human review is recommended.
Answer in the user's question language when practical."""
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text
        except Exception as exc:
            return f"Could not process your question. Detail: {exc}"
