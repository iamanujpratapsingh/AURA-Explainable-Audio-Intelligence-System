
🎙️ NovaTech AURA 

**NovaTech AURA** (Audio Understanding & Reasoning Agent) is a cloud-native audio forensics dashboard.. It replaces traditional, heavy local machine learning models with the speed and intelligence. **Functional TRL-4 Prototype**.

This system transforms raw audio files into actionable intelligence by generating transcripts, detecting granular emotions, identifying key sound events, and providing vocal executive summaries in seconds.

---

## ⚡ Key Features

* **Multimodal AI Engine:** Powered by **Google Gemini 1.5 Flash**, enabling direct audio-to-text processing without downloading massive models.
* **Auto-Discovery Model Loading:** Automatically finds the best available Gemini model for your API key (Flash, Pro, or Latest).
* **7-Language Support:** Native analysis for English, Hindi, Mandarin, Urdu, Tamil, Spanish, and French.
* **Granular Emotion Detection:** Identifies complex emotional states (e.g., *Panic, Hostile, Joy, Urgent*) rather than just positive/negative.
* **Vocal Summaries:** Uses **Google Text-to-Speech (gTTS)** to read the executive summary out loud to the user.
* **Interactive Timeline:** A color-coded visualization of the conversation flow, highlighting urgent segments.
* **Forensic QnA:** A "Chat with your Audio" feature that allows users to ask specific questions about the conversation (e.g., *"What time did the speaker mention the fire?"*).

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Python-based interactive UI)
* **AI Core:** [Google Generative AI](https://ai.google.dev/) (Gemini 1.5 Flash)
* **Voice Generation:** [gTTS](https://gtts.readthedocs.io/) (Google Text-to-Speech)
* **Visualization:** [Plotly](https://plotly.com/) (Timeline Charts) & [PyVis](https://pyvis.readthedocs.io/) (Network Graphs)
* **Data Handling:** Pandas & NumPy

---

## 🚀 Installation & Setup

### Prerequisites
1.  **Python 3.8** or higher installed.
2.  A **Google API Key** (Get it for free from [Google AI Studio](https://aistudio.google.com/)).

### Step 1: Clone or Download
Download this project folder to your local machine.

### Step 2: Create a Virtual Environment
Open your terminal in the project folder and run:

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

---

## ▶️ How to Run

### 1. Launch the Application

Run the following command in your terminal. (Using `python -m` ensures it runs correctly on Windows).

```bash
python -m streamlit run frontend/app.py

```

### 2. Configure the AI

* Once the app opens in your browser, look at the **Sidebar** on the left.
* Paste your **Google Gemini API Key** into the designated field.
* Select your target **Language** (e.g., Hindi, English).
* Choose your **Input Mode** (Upload File or Record Voice).

### 3. Analyze Audio

* Upload an audio file (WAV, MP3, M4A).
* Click **"🚀 Analyze with Gemini Cloud"**.
* The system will upload the audio to Google's secure AI studio, process it, and return the results in seconds.

---

## 📂 Project Structure

```text
NovaTech_AURA/
├── backend/
│   └── aura_engine.py       # Core logic (Gemini integration, gTTS, Auto-discovery)
├── frontend/
│   └── app.py               # UI code (Streamlit, Sidebar, Charts)
├── training/
│   └── knowledge_base.json  # Stores user feedback (simulated memory)
├── requirements.txt         # List of Python libraries
└── README.md                # Project documentation

```

---

## 🔮 Future Roadmap

* **Real-time WebSocket Streaming:** For live call center analysis.
* **Speaker Identification:** Integration with distinct voice fingerprinting.
* **PDF Reports:** Auto-generating forensic evidence reports.

---

**Built with ❤️**

```


```
