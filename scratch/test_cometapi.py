import os
from openai import OpenAI

# 1. Provide your CometAPI Key
COMET_API_KEY = "sk-BJLekCeqn4utK5njNoykhIktZgwjvvmWTKNy2yrRQ50SoMSA"
# 2. Point the base URL to CometAPI
COMET_BASE_URL = "https://api.cometapi.com/v1"

print("Initializing OpenAI client with CometAPI...")
client = OpenAI(api_key=COMET_API_KEY, base_url=COMET_BASE_URL)

print("Sending a basic chat completion request...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello! Can you hear me?"}
        ]
    )
    
    print("\n--- SUCCESS ---")
    print("Response:", response.choices[0].message.content)

except Exception as e:
    print("\n--- API ERROR ---")
    print(repr(e))
