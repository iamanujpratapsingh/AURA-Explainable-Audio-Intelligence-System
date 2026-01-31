import os
import json
import time
import google.generativeai as genai
from gtts import gTTS
import uuid
from datetime import datetime

class KnowledgeBase:
    def __init__(self, db_path="training/knowledge_base.json"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            with open(db_path, "w", encoding='utf-8') as f: json.dump([], f)

    def save_entry(self, data, notes):
        entry = {
            "id": str(uuid.uuid4()),
            "timestamp": str(datetime.now()),
            "data": data,
            "notes": notes
        }
        try:
            with open(self.db_path, "r+", encoding='utf-8') as f:
                try: db = json.load(f)
                except: db = []
                db.append(entry)
                f.seek(0); json.dump(db, f, indent=4); f.truncate()
            return "✅ Google Knowledge Graph Updated."
        except Exception as e: return f"Error: {e}"

class AuraEngine:
    def __init__(self):
        print("--- 🧠 INITIALIZING GOOGLE GEMINI ENGINE (DYNAMIC MODE) ---")
        self.api_key = None
        self.active_model_name = None # Will be found dynamically
        self.memory = KnowledgeBase()
        
    def set_api_key(self, key):
        self.api_key = key
        genai.configure(api_key=key)
        # Find a working model immediately
        self.find_best_model()

    def find_best_model(self):
        """Dynamically asks Google which models are available for this Key"""
        print("🔍 Searching for available Gemini models...")
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    # Prefer Flash for speed, but accept Pro or others
                    if 'flash' in m.name:
                        self.active_model_name = m.name
                        print(f"✅ Auto-Selected High Speed Model: {m.name}")
                        return
                    elif 'pro' in m.name and not self.active_model_name:
                        self.active_model_name = m.name
            
            # If we didn't find a specific preference, take the first valid one
            if not self.active_model_name:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        self.active_model_name = m.name
                        print(f"✅ Auto-Selected Available Model: {m.name}")
                        return
                        
            if not self.active_model_name:
                raise ValueError("No models found. Check API Key permissions.")
                
        except Exception as e:
            print(f"⚠️ Model Discovery Failed: {e}")
            # Absolute last resort fallback
            self.active_model_name = "models/gemini-1.5-flash"

    def generate_audio_response(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            filename = f"response_{uuid.uuid4().hex[:6]}.mp3"
            path = os.path.abspath(filename)
            tts.save(path)
            return path
        except: return None

    def process_audio(self, audio_path, language="English"):
        if not self.api_key:
            raise ValueError("⚠️ Google API Key Missing. Please add it in the Sidebar.")

        if not self.active_model_name:
            self.find_best_model()

        print(f"🚀 Uploading to Google AI Studio using {self.active_model_name}...")
        
        # Upload
        myfile = genai.upload_file(audio_path)
        while myfile.state.name == "PROCESSING":
            time.sleep(1)
            myfile = genai.get_file(myfile.name)

        prompt = f"""
        Analyze this audio file (Language: {language}).
        Return a strict JSON object with NO markdown formatting. 
        The JSON must match this structure exactly:
        {{
            "global_emotion": "Overall tone (e.g. Hostile, Panic, Joy)",
            "main_event": "Key sound event (e.g. Siren, Applause, Silence)",
            "transcript": [
                {{
                    "start": 0.0,
                    "end": 2.5,
                    "speaker": "Speaker A",
                    "text": "The spoken text",
                    "tone": "Specific emotion of this sentence",
                    "is_urgent": true/false
                }}
            ],
            "summary": "A 2-sentence executive summary of the situation."
        }}
        """

        try:
            model = genai.GenerativeModel(self.active_model_name)
            result = model.generate_content([myfile, prompt])
            
            raw_text = result.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw_text)
            audio_summary_path = self.generate_audio_response(data['summary'])
            
            return data['transcript'], data['main_event'], data['global_emotion'], data['summary'], audio_summary_path
            
        except Exception as e:
            raise ValueError(f"Gemini Processing Failed ({self.active_model_name}): {e}")

    def answer_question(self, transcript_data, question):
        try:
            model = genai.GenerativeModel(self.active_model_name)
            prompt = f"""
            Context: {json.dumps(transcript_data)}
            User Question: {question}
            Answer as a helpful forensic AI assistant. Cite specific timestamps.
            """
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {e}"