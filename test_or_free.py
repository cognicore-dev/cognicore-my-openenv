import openai
import os
from dotenv import load_dotenv

load_dotenv(override=True)
# The user provided this key earlier
or_key = "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"

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=or_key,
)

models_to_test = [
    "google/gemini-2.0-pro-exp-02-05:free",
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3-8b-instruct:free",
    "nousresearch/hermes-3-llama-3.1-405b:free"
]

for model in models_to_test:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=10
        )
        print(f"Success with {model}: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed with {model}: {e}")
