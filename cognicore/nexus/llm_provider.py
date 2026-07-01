"""
NEXUS Multi-Model LLM Provider.
Diverse model chain: Google, DeepSeek, Qwen, Gemma, Arcee.
"""
import http.client
import os, json, time, random

_NO_RETRY = ["api key expired","api key not valid","invalid_argument",
             "permission_denied","api_key_invalid","exceeded your current quota"]

DEFAULT_CHAIN = [
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.6-flash",
    "google/gemma-4-31b-it:free",
    "arcee-ai/trinity-large-thinking:free",
    "deepseek/deepseek-v4-flash:free",
]

class MultiLLM:
    def __init__(self, api_key=None, model_chain=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model_chain = model_chain or DEFAULT_CHAIN
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_calls = 0
        self._last_call = {}
        self._failed = set()
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

    def generate(self, system, user, max_tokens=4096, max_retries=2, initial_wait=15):
        errors = []
        for model in self.model_chain:
            if model in self._failed:
                continue
            try:
                return self._call(model, system, user, max_tokens, max_retries, initial_wait)
            except Exception as e:
                es = str(e)
                errors.append("{}: {}".format(model, es[:100]))
                print("  [LLM] {} failed: {}".format(model, es[:80]))
                if any(m in es.lower() for m in _NO_RETRY):
                    self._failed.add(model)
        raise RuntimeError("All models failed:\n" + "\n".join("  - " + e for e in errors))

    def _call(self, model, system, user, max_tokens, max_retries, initial_wait):
        url = "/api/v1/chat/completions"
        body = {"model": model, "messages": [{"role":"system","content":system},{"role":"user","content":user}], "max_tokens": max_tokens, "temperature": 0.2}
        payload = json.dumps(body).encode()
        connection = http.client.HTTPSConnection("openrouter.ai", timeout=90)
        for attempt in range(max_retries + 1):
            try:
                t0 = time.time()
                connection.request("POST", url, body=payload, headers={"Content-Type":"application/json","Authorization":"Bearer "+self.api_key,"HTTP-Referer":"https://github.com/cognicore/nexus","X-Title":"NEXUS Runtime"})
                resp = connection.getresponse()
                raw = resp.read().decode()
                if resp.status >= 400:
                    status = resp.status
                    eb = raw
                    if any(m in eb.lower() for m in _NO_RETRY):
                        raise RuntimeError("Auth/quota ({}): {}".format(status, eb[:150]))
                    if status in (429,502,503) and attempt < max_retries:
                        w = initial_wait*(2**attempt)+random.uniform(1,5)
                        print("  [LLM] {} rate limited ({}). Retry {}/{} in {}s...".format(model,status,attempt+1,max_retries,int(w)))
                        time.sleep(w); continue
                    raise RuntimeError("API error {}: {}".format(status, eb[:150]))
                data = json.loads(raw)
                lat = int((time.time()-t0)*1000)
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage",{})
                ti = usage.get("prompt_tokens", len(user)//4)
                to = usage.get("completion_tokens", len(text)//4)
                am = data.get("model", model)
                self.total_tokens_in += ti; self.total_tokens_out += to; self.total_calls += 1
                self._last_call = {"tokens_in":ti,"tokens_out":to,"latency_ms":lat,"model":am,"requested_model":model}
                print("  [LLM] OK {} -- {}->{}t, {}ms".format(am, ti, to, lat))
                return text
            except Exception as e:
                if attempt < max_retries: time.sleep(3+random.uniform(1,3)); continue
                raise

def get_llm(provider=None):
    if provider == "openrouter" or os.environ.get("OPENROUTER_API_KEY"):
        try: return MultiLLM()
        except ValueError: pass
    return None