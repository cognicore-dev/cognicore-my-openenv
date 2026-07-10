from google import genai
import os
import time

t0 = time.time()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
try:
    print("Sending request...")
    res = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Hello"
    )
    print("Response:", res.text)
except Exception as e:
    print(f"Error: {e}")

print(f"Time taken: {time.time() - t0}s")
