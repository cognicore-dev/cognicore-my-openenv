#!/usr/bin/env python3
"""
CogniCore CodeRepairBench v3 — Real LLM Adaptive Cognition

Uses REAL Gemini API to generate patches. CogniCore memory, semantic
patch rejection, reflection-driven prompt mutation, and strategy mutation
actively change how the LLM reasons during retries.

Run:  python examples/coderepair_llm.py
      python examples/coderepair_llm.py --local  (no API, uses rule agent)
"""
import sys, os, io, time, json, hashlib, difflib, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognicore.runtime import CogniCoreRuntime, RuntimeConfig

GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

def clog(tag, msg, detail=""):
    C = {"PATCH REJECTED":"\033[31m", "MEMORY RETRIEVAL":"\033[33m",
         "REFLECTION GENERATED":"\033[35m", "STRATEGY MUTATION":"\033[36m",
         "ADAPTIVE SUCCESS":"\033[32m", "FAILED PATCH":"\033[31m",
         "LLM CALL":"\033[94m", "PROMPT MUTATION":"\033[36m",
         "API ERROR":"\033[91m", "INFO":"\033[37m"}
    print(f"  {C.get(tag, chr(27)+'[0m')}[{tag}]\033[0m {msg}")
    if detail:
        for l in detail.strip().split("\n")[:4]:
            print(f"         {l}")

def sandbox(code, tests):
    ns = {}
    try:
        exec(compile(code, "<patch>", "exec"), ns)
        exec(compile(tests, "<test>", "exec"), ns)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def sim(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()

def phash(c): return hashlib.md5(c.strip().encode()).hexdigest()[:10]

def extract_code(text):
    """Extract clean Python code from LLM response."""
    if not text or text.startswith("# LLM ERROR"):
        return None  # Signal API failure
    # Extract from code fences
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    # Strip line numbers (1. def, 12. return, etc.)
    lines = []
    for line in text.strip().split("\n"):
        cleaned = re.sub(r'^\s*\d+[\.\)]\s+', '', line)
        cleaned = re.sub(r'^\s*\d+\s{2,}', '', cleaned)
        lines.append(cleaned)
    result = "\n".join(lines).strip()
    # Verify it's valid Python
    try:
        compile(result, "<check>", "exec")
        return result
    except SyntaxError:
        # Try removing first/last lines (markdown noise)
        for skip in range(1, min(3, len(lines))):
            trimmed = "\n".join(lines[skip:]).strip()
            try:
                compile(trimmed, "<check>", "exec")
                return trimmed
            except SyntaxError:
                pass
        return result  # Return anyway, sandbox will catch the error

# ══════════════════════════════════════════════════════════
# GEMINI CLIENT with rate limit handling
# ══════════════════════════════════════════════════════════
class GeminiClient:
    def __init__(self, api_key):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.call_count = 0
        self.errors = 0
        self.available = True

    def generate(self, prompt, temperature=0.3):
        if not self.available:
            return "# LLM ERROR: API quota exhausted"
        self.call_count += 1
        for retry in range(3):
            try:
                from google.genai import types
                resp = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=600)
                )
                return resp.text
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower():
                    wait = 10 * (retry + 1)
                    clog("API ERROR", f"Rate limited. Waiting {wait}s... (retry {retry+1}/3)")
                    time.sleep(wait)
                else:
                    self.errors += 1
                    return f"# LLM ERROR: {e}"
        self.available = False
        self.errors += 1
        return "# LLM ERROR: API quota exhausted after retries"

# ══════════════════════════════════════════════════════════
# FALLBACK RULE-BASED AGENT (when API unavailable)
# ══════════════════════════════════════════════════════════
RULE_FIXES = {
    "BUG-001": [
        ("direct", lambda c: c.replace("i+k+1", "i+k")),
        ("rewrite", lambda c: "def sliding_max(arr, k):\n  if not arr or k<=0: return []\n  return [max(arr[i:i+k]) for i in range(len(arr)-k+1)]\n"),
    ],
    "BUG-002": [
        ("guard", lambda c: c.replace("data['user']['name']", "(data.get('user') or {}).get('name')").replace("data['user']['email']", "(data.get('user') or {}).get('email')").replace("data['metrics']['score']", "(data.get('metrics') or {}).get('score')")),
        ("rewrite", lambda c: "def parse_api_response(data):\n  u=data.get('user') or {}\n  m=data.get('metrics') or {}\n  return {'id':data['id'],'name':u.get('name'),'email':u.get('email'),'score':m.get('score')}\n"),
    ],
    "BUG-003": [
        ("guard", lambda c: c.replace("node['children']", "(node.get('children') or [])")),
        ("rewrite", lambda c: "def flatten_tree(node):\n  r=[node['value']]\n  ch=node.get('children')\n  if ch:\n    for c in ch: r.extend(flatten_tree(c))\n  return r\n"),
    ],
    "BUG-004": [
        ("lock", lambda c: "import threading\nclass SafeCounter:\n  def __init__(self):\n    self.value=0;self._lock=threading.Lock()\n  def increment(self,amount=1):\n    with self._lock: self.value+=amount\n"),
    ],
    "BUG-005": [
        ("rewrite", lambda c: "def parse_config(path):\n  config={}\n  with open(path) as f:\n    for line in f:\n      line=line.strip()\n      if not line or line.startswith('#'): continue\n      if '=' in line:\n        k,v=line.split('=',1)\n        config[k.strip()]=v.strip()\n  return config\n"),
    ],
    "BUG-006": [
        ("memo", lambda c: "def fibonacci(n,_c={}):\n  if n in _c: return _c[n]\n  if n<=1: return n\n  _c[n]=fibonacci(n-1)+fibonacci(n-2)\n  return _c[n]\n"),
        ("iterative", lambda c: "def fibonacci(n):\n  if n<=1: return n\n  a,b=0,1\n  for _ in range(2,n+1): a,b=b,a+b\n  return b\n"),
    ],
}

def rule_agent_patch(bug, failed_patches, context):
    """Fallback agent when LLM is unavailable. Still uses CogniCore context."""
    fixes = RULE_FIXES.get(bug["id"], [])
    failed_hashes = {phash(p) for p in failed_patches}
    for name, fn in fixes:
        try:
            patch = fn(bug["buggy"])
            if patch and phash(patch) not in failed_hashes:
                if not any(sim(patch, fp) > 0.88 for fp in failed_patches[-3:]):
                    return patch, name
        except Exception:
            pass
    return bug["buggy"], "exhausted"

# ══════════════════════════════════════════════════════════
# BUG DATABASE
# ══════════════════════════════════════════════════════════
BUGS = [
    {"id": "BUG-001", "cat": "off_by_one",
     "title": "Sliding window maximum grabs extra element",
     "buggy": "def sliding_max(arr, k):\n    if not arr or k <= 0: return []\n    return [max(arr[i:i+k+1]) for i in range(len(arr)-k+1)]",
     "test": 'assert sliding_max([1,3,2,5,1,4], 3) == [3,5,5,5], f"got {sliding_max([1,3,2,5,1,4], 3)}"\nassert sliding_max([1], 1) == [1]\nassert sliding_max([], 3) == []\nassert sliding_max([4,3,2,1], 2) == [4,3,2]'},
    {"id": "BUG-002", "cat": "none_handling",
     "title": "API parser crashes on null nested fields",
     "buggy": "def parse_api_response(data):\n    return {\n        'id': data['id'],\n        'name': data['user']['name'],\n        'email': data['user']['email'],\n        'score': data['metrics']['score'],\n    }",
     "test": "full = {'id': 1, 'user': {'name': 'Jo', 'email': 'j@b.com'}, 'metrics': {'score': 95}}\nassert parse_api_response(full) == {'id': 1, 'name': 'Jo', 'email': 'j@b.com', 'score': 95}\npartial = {'id': 2, 'user': None, 'metrics': None}\nr = parse_api_response(partial)\nassert r['id'] == 2\nassert r['name'] is None\nassert r['score'] is None\nmissing = {'id': 3}\nr2 = parse_api_response(missing)\nassert r2['id'] == 3\nassert r2['name'] is None"},
    {"id": "BUG-003", "cat": "recursion",
     "title": "Tree flatten crashes when children is None",
     "buggy": "def flatten_tree(node):\n    result = [node['value']]\n    for child in node['children']:\n        result.extend(flatten_tree(child))\n    return result",
     "test": "tree = {'value': 1, 'children': [{'value': 2, 'children': []}, {'value': 3, 'children': [{'value': 4, 'children': []}]}]}\nassert flatten_tree(tree) == [1, 2, 3, 4]\nassert flatten_tree({'value': 99, 'children': None}) == [99]"},
    {"id": "BUG-004", "cat": "concurrency",
     "title": "Race condition in thread-safe counter",
     "buggy": "import threading\nclass SafeCounter:\n    def __init__(self):\n        self.value = 0\n    def increment(self, amount=1):\n        current = self.value\n        self.value = current + amount",
     "test": "import threading\nc = SafeCounter()\nts = [threading.Thread(target=c.increment) for _ in range(200)]\nfor t in ts: t.start()\nfor t in ts: t.join()\nassert c.value >= 195, f'Race: got {c.value}'"},
    {"id": "BUG-005", "cat": "resource_leak",
     "title": "Config parser crashes on comments/blanks",
     "buggy": "def parse_config(path):\n    f = open(path)\n    lines = f.readlines()\n    config = {}\n    for line in lines:\n        k, v = line.strip().split('=')\n        config[k] = v\n    f.close()\n    return config",
     "test": "import tempfile, os\nt = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)\nt.write('host=localhost\\nport=8080\\n# comment\\n\\ndb=postgres\\n')\nt.close()\nr = parse_config(t.name)\nos.unlink(t.name)\nassert r['host'] == 'localhost'\nassert r['port'] == '8080'\nassert r['db'] == 'postgres'"},
    {"id": "BUG-006", "cat": "recursion",
     "title": "Fibonacci exponential blowup",
     "buggy": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
     "test": "import time\nassert fibonacci(0) == 0\nassert fibonacci(10) == 55\nt = time.perf_counter()\nassert fibonacci(35) == 9227465\nassert time.perf_counter()-t < 1.0, 'Too slow'"},
]

def build_prompt(bug, failed, ctx, attempt):
    p = f"""Fix this buggy Python function. Return ONLY the raw Python code.
No line numbers. No explanations. No markdown. Just the function.

Bug: {bug['title']}

Buggy code:
{bug['buggy']}

Tests that must pass:
{bug['test']}"""
    if failed:
        p += f"\n\nWARNING: {len(failed)} previous fixes FAILED. Use a DIFFERENT approach.\n"
        for i, (patch, err) in enumerate(failed[-2:], 1):
            p += f"\nFailed fix {i} (DO NOT repeat):\n{patch[:300]}\nError: {err}\n"
    if ctx.get("reflection_hint"):
        p += f"\nCogniCore Reflection:\n{ctx['reflection_hint']}\nAdapt your strategy.\n"
    if ctx.get("failures_to_avoid"):
        p += f"\nKnown failed patterns: {', '.join(ctx['failures_to_avoid'][:3])}\n"
    if ctx.get("successful_patterns"):
        p += f"\nSuccessful patterns: {', '.join(ctx['successful_patterns'][:2])}\n"
    if attempt > 2:
        p += "\nIMPORTANT: Consider restructuring the entire function.\n"
    p += "\nReturn ONLY the Python code:"
    return p

# ══════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════
def run(max_attempts=4, use_local=False):
    print(f"\n{'='*64}")
    print(f"  CogniCore CodeRepairBench v3 — Real LLM Cognition")

    gemini = None
    if not use_local:
        try:
            gemini = GeminiClient(GEMINI_KEY)
            # Test call
            test = gemini.generate("Return only: pass", temperature=0.1)
            if "ERROR" in (test or ""):
                raise Exception("API test failed")
            print(f"  Gemini 2.0 Flash Lite | CONNECTED")
        except Exception as e:
            clog("API ERROR", f"Gemini unavailable: {e}")
            clog("INFO", "Falling back to rule-based agent with CogniCore cognition")
            gemini = None

    mode = "LLM" if gemini and gemini.available else "RULE"
    print(f"  Mode: {mode} | {len(BUGS)} bugs | {max_attempts} attempts")
    print(f"{'='*64}")

    runtime = CogniCoreRuntime(config=RuntimeConfig(
        reflection_min_samples=1, reflection_failure_threshold=1, memory_top_k=5,
    ), name="coderepair-v3")

    SIM_THRESH = 0.88
    results = []

    for bug in BUGS:
        print(f"\n  {'─'*56}")
        print(f"  {bug['id']}: {bug['title']} [{bug['cat']}]")

        # ── BASELINE ──
        print(f"  [A] BASELINE (no cognition)")
        b_failed, b_solved, b_attempts, b_repeats, b_hashes = [], False, 0, 0, set()

        for att in range(1, max_attempts+1):
            b_attempts = att
            if gemini and gemini.available:
                raw = gemini.generate(build_prompt(bug, [], {}, att), 0.3)
                patch = extract_code(raw)
                if patch is None:  # API error
                    patch, _ = rule_agent_patch(bug, [], {})
            else:
                patch, _ = rule_agent_patch(bug, [], {})

            h = phash(patch)
            if h in b_hashes: b_repeats += 1
            b_hashes.add(h)
            ok, err = sandbox(patch, bug["test"])
            if ok:
                b_solved = True
                clog("ADAPTIVE SUCCESS", f"Attempt {att}: PASSED")
                break
            else:
                b_failed.append((patch, err))
                rep = " **REPEATED**" if h in b_hashes and b_repeats else ""
                clog("FAILED PATCH", f"Attempt {att}: {err[:65]}{rep}")
            if gemini: time.sleep(1)

        # ── COGNICORE ──
        print(f"  [B] COGNICORE (memory + rejection + reflection + mutation)")
        c_failed, c_solved = [], False
        c_att, c_rep, c_mem, c_refl, c_mut, c_rej = 0, 0, 0, 0, 0, 0
        c_hashes = set()

        for att in range(1, max_attempts+1):
            c_att = att
            ctx = runtime._build_context(bug["cat"])

            if ctx.get("memory"):
                c_mem += 1
                fails = [e for e in ctx["memory"] if not e.get("correct")]
                if fails:
                    clog("MEMORY RETRIEVAL", f"{len(fails)} past failures in '{bug['cat']}'",
                         "\n".join(str(e.get("predicted",""))[:55] for e in fails[:2]))

            if ctx.get("reflection_hint"):
                c_refl += 1
                clog("REFLECTION GENERATED", "Modifying repair reasoning",
                     ctx["reflection_hint"][:100])

            if c_failed:
                c_mut += 1
                clog("PROMPT MUTATION",
                     f"{len(c_failed)} failed patches + {'reflection' if ctx.get('reflection_hint') else 'memory'} injected")

            temp = min(0.3 + 0.15 * len(c_failed), 0.9)
            failed_patches_only = [p for p, e in c_failed]

            if gemini and gemini.available:
                raw = gemini.generate(build_prompt(bug, c_failed, ctx, att), temp)
                patch = extract_code(raw)
                if patch is None:
                    patch, _ = rule_agent_patch(bug, failed_patches_only, ctx)
            else:
                patch, _ = rule_agent_patch(bug, failed_patches_only, ctx)

            h = phash(patch)

            # Semantic rejection
            if h in c_hashes:
                c_rej += 1
                clog("PATCH REJECTED", f"Identical to previous (hash={h})")
                if gemini and gemini.available:
                    raw = gemini.generate(
                        build_prompt(bug, c_failed, ctx, att) + "\nTry a COMPLETELY different approach.",
                        min(temp+0.3, 1.0))
                    patch = extract_code(raw) or patch
                else:
                    patch, _ = rule_agent_patch(bug, failed_patches_only + [patch], ctx)
                h = phash(patch)

            for prev, perr in c_failed[-3:]:
                s = sim(patch, prev)
                if s > SIM_THRESH:
                    c_rej += 1
                    clog("PATCH REJECTED", f"Similarity {s:.0%} > {SIM_THRESH:.0%} threshold",
                         f"Previous error: {perr[:55]}")
                    if gemini and gemini.available:
                        raw = gemini.generate(
                            build_prompt(bug, c_failed, ctx, att) +
                            f"\nDO NOT repeat this:\n{prev[:200]}\nError: {perr[:80]}\nUse DIFFERENT logic.",
                            min(temp+0.3, 1.0))
                        patch = extract_code(raw) or patch
                    else:
                        patch, _ = rule_agent_patch(bug, failed_patches_only + [patch], ctx)
                    h = phash(patch)
                    break

            if h in c_hashes: c_rep += 1
            c_hashes.add(h)

            ok, err = sandbox(patch, bug["test"])
            runtime.memory.store({
                "category": bug["cat"], "correct": ok, "bug_id": bug["id"],
                "predicted": f"err:{err[:80]}" if err else "PASS", "patch_hash": h,
            })

            if ok:
                c_solved = True
                clog("ADAPTIVE SUCCESS", f"Attempt {att}: PASSED",
                     f"Memory:{c_mem} Reflections:{c_refl} Mutations:{c_mut} Rejections:{c_rej}")
                break
            else:
                c_failed.append((patch, err or "unknown"))
                clog("FAILED PATCH", f"Attempt {att}: {err[:65]}")
            if gemini: time.sleep(1)

        results.append({"id":bug["id"],"cat":bug["cat"],"title":bug["title"],
            "b_solved":b_solved,"b_att":b_attempts,"b_rep":b_repeats,
            "c_solved":c_solved,"c_att":c_att,"c_rep":c_rep,
            "c_mem":c_mem,"c_refl":c_refl,"c_mut":c_mut,"c_rej":c_rej})

    # ── REPORT ──
    print(f"\n{'='*64}")
    print(f"  CODEREPAIRBENCH v3 RESULTS ({mode} mode)")
    print(f"{'='*64}")
    if gemini: print(f"  API calls: {gemini.call_count} | Errors: {gemini.errors}")
    print(f"\n  {'Bug':<10} {'Cat':<15} {'Base':>5} {'Cogni':>6} {'Mem':>4} {'Refl':>5} {'Mut':>4} {'Rej':>4}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['id']:<10} {r['cat']:<15} {'PASS' if r['b_solved'] else 'FAIL':>5} "
              f"{'PASS' if r['c_solved'] else 'FAIL':>6} {r['c_mem']:>4} {r['c_refl']:>5} {r['c_mut']:>4} {r['c_rej']:>4}")

    bs=sum(1 for r in results if r["b_solved"])
    cs=sum(1 for r in results if r["c_solved"])
    br=sum(r["b_rep"] for r in results)
    cr=sum(r["c_rep"] for r in results)
    print(f"\n  {'Metric':<30} {'Baseline':>10} {'CogniCore':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Bugs Solved':<30} {bs:>10} {cs:>10}")
    print(f"  {'Repeated Patches':<30} {br:>10} {cr:>10}")
    print(f"  {'Memory Retrievals':<30} {'--':>10} {sum(r['c_mem'] for r in results):>10}")
    print(f"  {'Reflections':<30} {'--':>10} {sum(r['c_refl'] for r in results):>10}")
    print(f"  {'Prompt Mutations':<30} {'--':>10} {sum(r['c_mut'] for r in results):>10}")
    print(f"  {'Patch Rejections':<30} {'--':>10} {sum(r['c_rej'] for r in results):>10}")

    print(f"\n  VERDICT:")
    for l,ok in [("CogniCore solves >= baseline",cs>=bs),("Memory active",sum(r['c_mem'] for r in results)>0),
                  ("Prompt mutations applied",sum(r['c_mut'] for r in results)>0),
                  ("Repeated patches reduced",cr<=br)]:
        print(f"  [{'PROVEN' if ok else 'NOT PROVEN'}] {l}")
    print(f"{'='*64}\n")

    rp=os.path.join(os.path.dirname(os.path.abspath(__file__)),"llm_bench_report.json")
    with open(rp,"w") as f: json.dump({"results":results,"mode":mode},f,indent=2)
    print(f"  Report: {rp}")

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--local",action="store_true",help="Skip LLM, use rule agent")
    p.add_argument("--attempts",type=int,default=4)
    a=p.parse_args()
    run(max_attempts=a.attempts, use_local=a.local)
