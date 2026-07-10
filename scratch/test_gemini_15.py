import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key starting with: {api_key[:5]}")
genai.configure(api_key=api_key)

try:
    for m in genai.list_models():
        print(m.name)
except Exception as e:
    print(f"FAILED: {e}")
