import os
import sys
from dotenv import load_dotenv
from google import genai

# Fix Windows terminal Unicode
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

print("--- Checking Gemini Configuration ---")
print(f"Configured model: {MODEL}")

if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
    print("ERROR: GEMINI_API_KEY not set. Add it to your .env file.")
    print("Get a free key at: https://aistudio.google.com/")
    sys.exit(1)

try:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model=MODEL, contents="Reply with exactly: OK")
    print(f"SUCCESS: Model '{MODEL}' is reachable. Response: {response.text.strip()}")
except Exception as e:
    print(f"FAILED: {e}")
    if "API_KEY_INVALID" in str(e) or "invalid" in str(e).lower():
        print("Your API key appears to be invalid.")
        print("A valid Gemini API key starts with 'AIza...'")
        print("Get one at: https://aistudio.google.com/")

print("-------------------------------------")
