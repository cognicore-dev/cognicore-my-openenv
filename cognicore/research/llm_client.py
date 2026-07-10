"""
LLM Client — multi-provider LLM integration for CogniCore Research.
Supports: Gemini, OpenAI, Claude, local models.
Uses environment variables for API keys (never hardcoded).
"""
import os, time
from typing import Optional


class LLMClient:
    """Unified LLM client with multi-provider support."""

    def __init__(self, provider: str = "auto", model: str = None):
        self.provider = provider
        self.model = model
        self.call_count = 0
        self.errors = 0
        self.available = False
        self._client = None
        self._init_provider(provider)

    def _init_provider(self, provider: str):
        if provider == "auto":
            # Try providers in order
            for p in ["gemini", "openai", "claude"]:
                try:
                    self._init_provider(p)
                    if self.available:
                        return
                except Exception:
                    continue
            return

        if provider == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                return
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
                self.model = self.model or "gemini-2.0-flash"
                self.provider = "gemini"
                self.available = True
            except ImportError:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=key)
                    self._client = genai.GenerativeModel(self.model or "gemini-2.0-flash-lite")
                    self.model = self.model or "gemini-2.0-flash-lite"
                    self.provider = "gemini-legacy"
                    self.available = True
                except Exception:
                    pass

        elif provider == "openai":
            key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            if not key:
                return
            try:
                from openai import OpenAI
                kwargs = {"api_key": key}
                if base_url:
                    kwargs["base_url"] = base_url
                self._client = OpenAI(**kwargs)
                self.model = self.model or "gpt-4.1-mini"
                self.provider = "openai"
                self.available = True
            except ImportError:
                pass

        elif provider == "claude":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                return
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
                self.model = self.model or "claude-3-haiku-20240307"
                self.provider = "claude"
                self.available = True
            except ImportError:
                pass

    def generate(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        if not self.available:
            return None
        self.call_count += 1

        for retry in range(3):
            try:
                if self.provider == "gemini":
                    from google.genai import types
                    r = self._client.models.generate_content(
                        model=self.model, contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=temperature, max_output_tokens=600))
                    return r.text

                elif self.provider == "gemini-legacy":
                    r = self._client.generate_content(
                        prompt,
                        generation_config={"temperature": temperature,
                                           "max_output_tokens": 600})
                    return r.text

                elif self.provider == "openai":
                    r = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=600)
                    return r.choices[0].message.content

                elif self.provider == "claude":
                    r = self._client.messages.create(
                        model=self.model, max_tokens=600,
                        messages=[{"role": "user", "content": prompt}])
                    return r.content[0].text

            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    wait = 10 * (retry + 1)
                    time.sleep(wait)
                    continue
                self.errors += 1
                return None

        self.available = False
        return None

    def test_connection(self) -> bool:
        if not self.available:
            return False
        result = self.generate("Reply with only: ok", temperature=0.1)
        return result is not None and len(result) > 0

    def __repr__(self):
        return f"LLMClient(provider={self.provider}, model={self.model}, available={self.available})"
