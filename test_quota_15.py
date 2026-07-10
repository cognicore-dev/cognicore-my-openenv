from google import genai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for i in range(5):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Hello {i}"
        )
        print(f"Success {i}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        break
