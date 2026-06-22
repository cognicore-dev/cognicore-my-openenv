import os
import time
from typing import Dict, Any, Optional
import openai
from dotenv import load_dotenv
load_dotenv(override=True)

class LLMClient:
    """Centralized LLM client for benchmark inference, with token/cost tracking."""
    
    # Simple cost mapping (per 1k tokens) as of mid-2024
    COST_MAP = {
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    }

    def __init__(self, model_name: str = "openai/gpt-4o-mini"):
        self.model_name = model_name
        self.is_mock = False
        self.client_type = "openai"
        
        groq_key = os.getenv("GROQ_API_KEY")
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        try:
            if groq_key:
                self.client = openai.OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_key,
                    timeout=30.0
                )
            elif openrouter_key:
                self.client = openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key,
                    default_headers={
                        "HTTP-Referer": "https://cognicore.dev",
                        "X-Title": "CogniCore Benchmark"
                    }
                )
            elif gemini_key:
                self.client_type = "gemini"
                if "gpt" in self.model_name.lower():
                    self.model_name = "gemini-1.5-flash-latest"
                elif self.model_name.startswith("google/"):
                    self.model_name = self.model_name.split("/", 1)[1]
            else:
                self.client = openai.OpenAI()
        except Exception:
            print("No valid API Key found. Using MOCK LLM Client.")
            self.is_mock = True
            self.client = None
        
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500) -> Dict[str, Any]:
        """Generate a response from the LLM and track usage."""
        if self.is_mock:
            # Mock behavior for testing pipelines without an API key
            time.sleep(0.5)
            self._track_usage(100, 20)
            return {
                "content": "MOCK ANSWER",
                "latency_s": 0.5,
                "prompt_tokens": 100,
                "completion_tokens": 20,
            }
            
        t0 = time.time()
        max_retries = 5
        base_delay = 5.0
        
        for attempt in range(max_retries):
            try:
                if self.client_type == "gemini":
                    from google import genai
                    from google.genai import types
                    
                    gemini_key = os.getenv("GEMINI_API_KEY")
                    client = genai.Client(api_key=gemini_key)
                    
                    config_kwargs = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                    if system_prompt:
                        config_kwargs["system_instruction"] = system_prompt
                    
                    response = client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(**config_kwargs)
                    )
                    
                    content = response.text
                    
                    try:
                        usage_metadata = response.usage_metadata
                        prompt_tokens = usage_metadata.prompt_token_count
                        completion_tokens = usage_metadata.candidates_token_count
                    except Exception:
                        prompt_tokens = len(prompt) // 4
                        completion_tokens = len(content) // 4
                        
                    self._track_usage(prompt_tokens, completion_tokens)
                    
                    return {
                        "content": content,
                        "latency_s": time.time() - t0,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
                else:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})
                    
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    
                    content = response.choices[0].message.content
                    usage = response.usage
                    
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    
                    self._track_usage(prompt_tokens, completion_tokens)
                    
                    return {
                        "content": content,
                        "latency_s": time.time() - t0,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    }
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str or "too many requests" in err_str or "503" in err_str or "500" in err_str:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"Rate limit hit. Retrying in {delay} seconds...")
                        time.sleep(delay)
                        continue
                print(f"LLM API Error: {e}")
                return {
                    "content": "",
                    "latency_s": time.time() - t0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": str(e)
                }

    def _track_usage(self, prompt_tokens: int, completion_tokens: int):
        """Track cumulative tokens and estimate costs."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        
        rates = self.COST_MAP.get(self.model_name, {"prompt": 0.0, "completion": 0.0})
        cost = (prompt_tokens / 1000.0) * rates["prompt"] + (completion_tokens / 1000.0) * rates["completion"]
        self.total_cost += cost

    def get_stats(self) -> Dict[str, Any]:
        """Return cumulative tracking stats."""
        return {
            "model": self.model_name,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "estimated_cost_usd": self.total_cost
        }
