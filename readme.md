# 🎙️ AURA: Explainable Audio Intelligence System

**AURA** (Audio Understanding & Reasoning Agent) is a cloud-native audio forensics dashboard powered by Google Gemini. It transforms raw audio files into actionable intelligence — generating transcripts, detecting granular emotions, identifying key sound events, and delivering vocal executive summaries in seconds.

**Functional TRL-4 Prototype.**

---

## ⚡ Key Features

- **Multimodal AI Engine:** Powered by **Google Gemini** (configured via `.env`), enabling direct audio-to-text processing.
- **Explicit Model Control:** Model is set via `GEMINI_MODEL` in `.env` — no dynamic model switching.
- **7-Language Support:** English, Hindi, Mandarin, Urdu, Tamil, Spanish, French.
- **Granular Emotion Detection:** Identifies complex emotional states (Panic, Hostile, Joy, Urgent, etc.).
- **Vocal Summaries:** Uses **gTTS** to read the executive summary aloud.
- **Interactive Timeline:** Color-coded Plotly visualization of the conversation flow.
- **Forensic QnA:** "Ask AURA" — chat with your audio using Gemini, with timestamp-aware answers.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI Core | Google Generative AI (Gemini) |
| Voice | gTTS (Google Text-to-Speech) |
| Visualization | Plotly (Timeline) |
| Data | Pandas |
| Config | python-dotenv |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- A **Google Gemini API Key** — get one free from [Google AI Studio](https://aistudio.google.com/)

### Step 1: Clone or Download

```bash
git clone <repo-url>
cd AURA-Explainable-Audio-Intelligence-System
```

### Step 2: Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

> ⚠️ **Never commit `.env` to GitHub.** It is already listed in `.gitignore`.
> A safe template is provided in `.env.example`.

---

## ▶️ How to Run

```bash
python -m streamlit run frontend/app.py
```

The application will automatically:
1. Load `GEMINI_API_KEY` and `GEMINI_MODEL` from `.env`
2. Initialize the Gemini engine
3. Show **● Gemini Connected** in the sidebar and header

No API key entry is required in the UI.

---

## 🔍 Verify Gemini Connection

To test your API key and model before running the full app:

```bash
python check_models.py
```

Expected output:
```
--- 🔍 Checking Gemini Configuration ---
Configured model: gemini-2.0-flash
✅ Model 'gemini-2.0-flash' is reachable. Response: OK
```

---

## 📂 Project Structure

```
AURA-Explainable-Audio-Intelligence-System/
├── backend/
│   └── aura_engine.py       # Core logic (Gemini, gTTS, KnowledgeBase)
├── frontend/
│   └── app.py               # Streamlit UI
├── training/
│   └── knowledge_base.json  # Feedback/memory store
├── .env                     # ⚠️ Local only — not committed
├── .env.example             # Safe template to commit
├── .gitignore
├── check_models.py          # Gemini connection test utility
├── requirements.txt
└── README.md
```

---

## 🔄 Application Workflow

```
.env (GEMINI_API_KEY + GEMINI_MODEL)
    ↓
AuraEngine initializes Gemini
    ↓
User uploads or records audio
    ↓
Click "Analyze Audio"
    ↓
Audio uploaded to Google AI Studio
    ↓
Gemini returns: Transcript + Emotion + Event + Summary
    ↓
Live Timeline  |  Transcript Data Grid  |  Ask AURA (QnA)
    ↓
gTTS reads the executive summary aloud
```

---

## 🔮 Future Roadmap

- **Real-time WebSocket Streaming** — live call center analysis
- **Speaker Identification** — voice fingerprinting
- **PDF Reports** — auto-generated forensic evidence reports

---

**Built with ❤️ — AURA Explainable Audio Intelligence**

---

## AURA audio intelligence workflow

AURA is an AI-assisted audio intelligence system that transforms unstructured audio into structured, explainable, time-aware insights for faster human review. Gemini is the analysis engine; AURA supplies the input workflow, structured incident view, explainability, reviewer controls, and an auditable analysis record.

```text
Audio input
  -> Gemini audio analysis (one structured request)
  -> Transcript + tone + event context
  -> Incident brief + attention priority
  -> Smart timeline + selected-moment explanation
  -> Ask AURA (global or moment mode)
  -> Analysis record + optional audio summary
```

## Structured analysis features

- **AURA Incident Brief** — event, timestamp range, speakers, model-reported sound events, context, observations, confidence, summary, and a human-review reminder.
- **Attention Priority** — LOW, MEDIUM, or HIGH based on model-provided priority and reasons. This is a review aid, not a safety, crime, or risk determination.
- **Smart Event Timeline** — transcript segments are coloured by importance and retain timestamps, speaker labels, tone, urgency, confidence, event context, and model-provided reasons when available.
- **Why was this flagged?** — selecting an important/urgent segment displays only the reasons supplied in the analysis. AURA never invents acoustic or behavioural signals that were not returned by the model.
- **Ask About This Moment** — select a timeline segment to switch Ask AURA into Moment Mode; the question receives the selected segment and only its immediate context. Without a selection, Ask AURA remains in Global Mode for the complete audio.
- **Multilingual understanding** — the selected language guides Gemini, detected languages are displayed only when Gemini returns them, transcripts retain their original language, and AURA provides an English summary plus an optional original-language summary.
- **AURA Analysis Record** — includes an analysis ID, input filename, duration (using WAV metadata when available or the final transcript timestamp otherwise), UTC analysis time, configured model, status, event count, overall priority, and SHA-256 integrity reference.

## Privacy controls

Privacy Mode is available in the sidebar:

- **Remove temporary uploaded audio after analysis** is enabled by default. It deletes AURA's local temporary input file once processing finishes; it does not retract a file already sent to the configured AI provider.
- **Do not retain conversation history** displays the current answer but does not keep chat messages in the Streamlit session.
- **Anonymize speaker labels** replaces displayed speaker labels with `Speaker A`, `Speaker B`, and so on. It does not alter the transcript words.

These controls improve predictable local handling but are not a complete privacy or security guarantee. Do not upload audio you are not authorized to process, and review the applicable provider and organizational policies.

## Human-in-the-loop and limitations

AURA reports model-derived observations and AI assessments to support a reviewer. It does not prove events, identify people, determine criminality, make safety predictions, or establish legal admissibility. Confidence values are model estimates, and the SHA-256 value is an integrity reference only. Audio quality, overlapping voices, unsupported languages, and model limitations can affect transcript and timeline accuracy. Human review is recommended for every important finding.

## Example review flow

1. Choose the likely spoken language and upload an MP3, WAV, or M4A file, or record with the microphone.
2. Optionally configure Privacy Mode, then select **Analyze Audio**.
3. Review the Incident Brief, priority reasons, and important findings.
4. Use the Smart Timeline to select a moment and review its model-provided explanation.
5. Ask a global question or a moment-specific question, then inspect the Analysis Record and integrity hash.
