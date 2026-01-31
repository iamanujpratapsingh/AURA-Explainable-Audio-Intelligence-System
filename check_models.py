import google.generativeai as genai
import os

# ---------------------------------------------------------
# PASTE YOUR API KEY BELOW TO TEST
# ---------------------------------------------------------
API_KEY = "AIzaSyDAWuB6s6OVUc-BdOWWdSYW9dx9RyREw1k"

genai.configure(api_key=API_KEY)

print("--- 🔍 Checking Available Google Models ---")
try:
    count = 0
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Found: {m.name}")
            count += 1
    if count == 0:
        print("❌ No models found. Check your API Key permissions.")
except Exception as e:
    print(f"❌ Error: {e}")
print("-------------------------------------------")