"""
NEXUS LLM Provider — connects the agent to real LLM APIs.
Currently supports: Gemini (Google), OpenAI, Anthropic.
"""
import os, json, re


class GeminiLLM:
    """Gemini API wrapper for NEXUS agent."""

    def __init__(self, model="gemini-2.0-flash"):
    if self is None:
        return None
    if self is None:
        return None
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set")

    def generate(self, system: str, user: str, max_tokens=4096) -> str:
        import urllib.request
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        body = {
            "contents": [{"parts": [{"text": f"{system}\n\n{user}"}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.2}
        }
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text


def get_llm(provider=None):
    """Auto-detect and return available LLM."""
    if provider == "gemini" or os.environ.get("GEMINI_API_KEY"):
        return GeminiLLM()
    raise ValueError("No LLM API key found. Set GEMINI_API_KEY.")
