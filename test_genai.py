from google import genai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Hello",
    config={"temperature": 0}
)
print("Response text:", response.text)
if hasattr(response, 'usage_metadata'):
    print("Tokens:", response.usage_metadata.prompt_token_count, response.usage_metadata.candidates_token_count)
