import os
from cognicore_benchmarks.common.llm_client import LLMClient

models = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-nemo:free",
    "google/gemini-2.0-pro-exp-02-05:free",
    "google/gemini-2.5-pro:free"
]

for m in models:
    try:
        client = LLMClient(model_name=m)
        res = client.generate("Is the sky blue? Answer yes or no only.", system_prompt=None)
        print(f"{m}: {res.get('content')}")
    except Exception as e:
        print(f"{m}: FAILED - {e}")
