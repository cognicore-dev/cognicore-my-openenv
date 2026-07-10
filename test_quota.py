from google import genai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello"
    )
    print("Success:", response.text)
except Exception as e:
    import traceback
    traceback.print_exc()
