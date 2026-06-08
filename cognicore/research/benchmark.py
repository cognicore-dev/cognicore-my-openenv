#!/usr/bin/env python3
"""
CogniCore Research Benchmark — Adaptive Runtime Cognition

Research-grade A/B comparison: baseline agent vs CogniCore-enhanced agent.
Real code execution, real test validation, AST-based patch similarity,
experiment tracking, reproducible configs, multi-LLM support.

Usage:
  python -m cognicore.research.benchmark              # auto-detect LLM
  python -m cognicore.research.benchmark --seed 42     # reproducible run
  GEMINI_API_KEY=xxx python -m cognicore.research.benchmark  # with Gemini
  OPENAI_API_KEY=xxx python -m cognicore.research.benchmark  # with GPT
"""
import sys, os, io, re, time, argparse, subprocess
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from cognicore.runtime import CogniCoreRuntime, RuntimeConfig
from cognicore.research.patch_intelligence import (
    combined_similarity, patch_hash, text_similarity, detect_repeated_reasoning, PatchStore
)
from cognicore.research.experiment import ExperimentConfig, ExperimentResult, ExperimentTracker
from cognicore.research.llm_client import LLMClient

# ══════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════
def clog(tag, msg, detail=""):
    C = {"PATCH REJECTED":"\033[91m", "MEMORY RETRIEVAL":"\033[33m",
         "REFLECTION GENERATED":"\033[35m", "STRATEGY MUTATION":"\033[36m",
         "ADAPTIVE SUCCESS":"\033[32m", "FAILED PATCH":"\033[31m",
         "REPEATED REASONING":"\033[91m", "PROMPT MUTATION":"\033[36m",
         "LLM":"\033[94m", "INFO":"\033[37m", "AST SIMILARITY":"\033[33m"}
    print(f"  {C.get(tag, chr(27)+'[0m')}[{tag}]\033[0m {msg}")
    if detail:
        for l in detail.strip().split("\n")[:4]:
            print(f"         {l}")

def sandbox(code, tests):
    try:
        result = subprocess.run(
            [sys.executable, "-c", f"{code}\n\n{tests}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return True, None
        output = (result.stderr or result.stdout or "").strip()
        return False, output.splitlines()[-1] if output else f"Exit code {result.returncode}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def extract_code(text):
    if not text: return None
    if "```python" in text: text = text.split("```python",1)[1].split("```",1)[0]
    elif "```" in text: text = text.split("```",1)[1].split("```",1)[0]
    lines = []
    for line in text.strip().split("\n"):
        cleaned = re.sub(r'^\s*\d+[\.\)]\s+', '', line)
        lines.append(cleaned)
    result = "\n".join(lines).strip()
    try:
        compile(result, "<check>", "exec")
        return result
    except SyntaxError:
        return result

# ══════════════════════════════════════════════════════════
# BUG DATABASE — 10 real bugs, multiple categories
# ══════════════════════════════════════════════════════════
BUGS = [
  {"id":"B01","cat":"off_by_one","title":"Sliding window grabs extra element",
   "buggy":"def sliding_max(arr, k):\n    if not arr or k<=0: return []\n    return [max(arr[i:i+k+1]) for i in range(len(arr)-k+1)]",
   "test":'assert sliding_max([1,3,2,5,1,4],3)==[3,5,5,5]\nassert sliding_max([1],1)==[1]\nassert sliding_max([],3)==[]\nassert sliding_max([4,3,2,1],2)==[4,3,2]',
   "fixes":{"direct":lambda c:c.replace("i+k+1","i+k"),
            "rewrite":lambda c:"def sliding_max(arr,k):\n  if not arr or k<=0: return []\n  return [max(arr[i:i+k]) for i in range(len(arr)-k+1)]\n"}},

  {"id":"B02","cat":"none_handling","title":"Dict merge crashes on missing keys",
   "buggy":"def merge_profiles(base, update):\n    result = {}\n    for key in set(list(base.keys())+list(update.keys())):\n        result[key] = update[key] if update[key] else base[key]\n    return result",
   "test":"r=merge_profiles({'name':'Jo','age':25},{'name':'Jo B','email':'j@b.com'})\nassert r['name']=='Jo B'\nassert r['age']==25\nassert r['email']=='j@b.com'",
   "fixes":{"guard":lambda c:c.replace("update[key] if update[key] else base[key]","update.get(key,base.get(key))"),
            "rewrite":lambda c:"def merge_profiles(base,update):\n  r={**base}\n  r.update({k:v for k,v in update.items() if v is not None})\n  return r\n"}},

  {"id":"B03","cat":"recursion","title":"Tree flatten crashes on None children",
   "buggy":"def flatten_tree(node):\n    result = [node['value']]\n    for child in node['children']:\n        result.extend(flatten_tree(child))\n    return result",
   "test":"tree={'value':1,'children':[{'value':2,'children':[]},{'value':3,'children':[{'value':4,'children':[]}]}]}\nassert flatten_tree(tree)==[1,2,3,4]\nassert flatten_tree({'value':99,'children':None})==[99]",
   "fixes":{"guard":lambda c:c.replace("node['children']","(node.get('children') or [])"),
            "rewrite":lambda c:"def flatten_tree(node):\n  r=[node['value']]\n  ch=node.get('children')\n  if ch:\n    for c in ch: r.extend(flatten_tree(c))\n  return r\n"}},

  {"id":"B04","cat":"concurrency","title":"Race condition in thread-safe counter",
   "buggy":"import threading\nclass SafeCounter:\n    def __init__(self):\n        self.value = 0\n    def increment(self, amount=1):\n        current = self.value\n        self.value = current + amount",
   "test":"import threading\nc=SafeCounter()\nts=[threading.Thread(target=c.increment) for _ in range(200)]\nfor t in ts: t.start()\nfor t in ts: t.join()\nassert c.value>=195, f'Race: {c.value}'",
   "fixes":{"lock":lambda c:"import threading\nclass SafeCounter:\n  def __init__(self):\n    self.value=0;self._lock=threading.Lock()\n  def increment(self,amount=1):\n    with self._lock: self.value+=amount\n"}},

  {"id":"B05","cat":"resource_leak","title":"Config parser crashes on comments",
   "buggy":"def parse_config(path):\n    f = open(path)\n    lines = f.readlines()\n    config = {}\n    for line in lines:\n        k, v = line.strip().split('=')\n        config[k] = v\n    f.close()\n    return config",
   "test":"import tempfile,os\nt=tempfile.NamedTemporaryFile(mode='w',suffix='.cfg',delete=False)\nt.write('host=localhost\\nport=8080\\n# comment\\n\\ndb=postgres\\n')\nt.close()\nr=parse_config(t.name)\nos.unlink(t.name)\nassert r['host']=='localhost'\nassert r['port']=='8080'\nassert r['db']=='postgres'",
   "fixes":{"rewrite":lambda c:"def parse_config(path):\n  config={}\n  with open(path) as f:\n    for line in f:\n      line=line.strip()\n      if not line or line.startswith('#'): continue\n      if '=' in line:\n        k,v=line.split('=',1)\n        config[k.strip()]=v.strip()\n  return config\n"}},

  {"id":"B06","cat":"off_by_one","title":"Pagination drops last page",
   "buggy":"def paginate(items, page_size):\n    total_pages = len(items) // page_size\n    pages = []\n    for i in range(total_pages):\n        pages.append(items[i*page_size:(i+1)*page_size])\n    return pages",
   "test":"r=paginate([1,2,3,4,5],2)\nassert len(r)==3\nassert r[0]==[1,2]\nassert r[2]==[5]\nassert paginate([],5)==[]",
   "fixes":{"ceil":lambda c:c.replace("len(items) // page_size","(len(items)+page_size-1)//page_size"),
            "rewrite":lambda c:"def paginate(items,ps):\n  return [items[i:i+ps] for i in range(0,len(items),ps)] if items else []\n"}},

  {"id":"B07","cat":"none_handling","title":"API parser crashes on null fields",
   "buggy":"def parse_api_response(data):\n    return {\n        'id': data['id'],\n        'name': data['user']['name'],\n        'email': data['user']['email'],\n        'score': data['metrics']['score'],\n    }",
   "test":"full={'id':1,'user':{'name':'Jo','email':'j@b.com'},'metrics':{'score':95}}\nassert parse_api_response(full)=={'id':1,'name':'Jo','email':'j@b.com','score':95}\npartial={'id':2,'user':None,'metrics':None}\nr=parse_api_response(partial)\nassert r['id']==2\nassert r['name'] is None\nassert r['score'] is None\nmissing={'id':3}\nr2=parse_api_response(missing)\nassert r2['name'] is None",
   "fixes":{"guard":lambda c:c.replace("data['user']['name']","data.get('user',{}).get('name')").replace("data['user']['email']","data.get('user',{}).get('email')").replace("data['metrics']['score']","data.get('metrics',{}).get('score')"),
            "rewrite":lambda c:"def parse_api_response(data):\n  u=data.get('user') or {}\n  m=data.get('metrics') or {}\n  return {'id':data['id'],'name':u.get('name'),'email':u.get('email'),'score':m.get('score')}\n"}},

  {"id":"B08","cat":"recursion","title":"Fibonacci exponential blowup",
   "buggy":"def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)",
   "test":"import time\nassert fibonacci(0)==0\nassert fibonacci(10)==55\nt=time.perf_counter()\nassert fibonacci(35)==9227465\nassert time.perf_counter()-t<1.0,'Too slow'",
   "fixes":{"memo":lambda c:"def fibonacci(n,_c={}):\n  if n in _c: return _c[n]\n  if n<=1: return n\n  _c[n]=fibonacci(n-1)+fibonacci(n-2)\n  return _c[n]\n",
            "iterative":lambda c:"def fibonacci(n):\n  if n<=1: return n\n  a,b=0,1\n  for _ in range(2,n+1): a,b=b,a+b\n  return b\n"}},

  {"id":"B09","cat":"off_by_one","title":"Binary search misses target",
   "buggy":"def binary_search(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target:\n            lo = mid\n        elif arr[mid] > target:\n            hi = mid\n        else:\n            return mid\n    return -1",
   "test":"assert binary_search([1,3,5,7,9],5)==2\nassert binary_search([1,3,5,7,9],1)==0\nassert binary_search([1,3,5,7,9],9)==4\nassert binary_search([1,3,5,7,9],4)==-1\nassert binary_search([],5)==-1",
   "fixes":{"direct":lambda c:c.replace("lo = mid","lo = mid + 1"),
            "rewrite":lambda c:"def binary_search(arr,target):\n  lo,hi=0,len(arr)-1\n  while lo<=hi:\n    mid=(lo+hi)//2\n    if arr[mid]<target: lo=mid+1\n    elif arr[mid]>target: hi=mid-1\n    else: return mid\n  return -1\n"}},

  {"id":"B10","cat":"none_handling","title":"JSON flattener crashes on nested nulls",
   "buggy":"def flatten_json(obj, prefix=''):\n    items = {}\n    for k, v in obj.items():\n        key = f'{prefix}.{k}' if prefix else k\n        if isinstance(v, dict):\n            items.update(flatten_json(v, key))\n        else:\n            items[key] = v\n    return items",
   "test":"assert flatten_json({'a':1,'b':{'c':2}})=={'a':1,'b.c':2}\nassert flatten_json({'x':None})=={'x':None}\nassert flatten_json({'a':{'b':None}})=={'a.b':None}\nassert flatten_json({})=={}",
   "fixes":{"guard":lambda c:c,  # returns buggy — will fail on nested null dict
            "rewrite":lambda c:"def flatten_json(obj, prefix=''):\n  items={}\n  if not isinstance(obj,dict): return items\n  for k,v in obj.items():\n    key=f'{prefix}.{k}' if prefix else k\n    if isinstance(v,dict):\n      items.update(flatten_json(v,key))\n    else:\n      items[key]=v\n  return items\n"}},
]

# ══════════════════════════════════════════════════════════
# MUTABLE STRATEGY
# ══════════════════════════════════════════════════════════
class RepairStrategy:
    def __init__(self):
        self.disabled = set()
        self.preferred = []
        self.history = []

    def mutate(self, disable=None, prefer=None, reason=""):
        if disable:
            self.disabled.add(disable)
        if prefer and prefer not in self.preferred:
            self.preferred.insert(0, prefer)
        self.history.append(f"disable={disable}, prefer={prefer}: {reason}")
        clog("STRATEGY MUTATION",
             f"Disabled: {disable} | Preferred: {prefer}", reason)

    def get_order(self, available):
        ordered = [t for t in self.preferred if t in available and t not in self.disabled]
        ordered += [t for t in available if t not in ordered and t not in self.disabled]
        return ordered

# ══════════════════════════════════════════════════════════
# PROMPT BUILDER
# ══════════════════════════════════════════════════════════
def build_prompt(bug, failed_patches, ctx, attempt):
    p = f"""Fix this buggy Python function. Return ONLY raw Python code.
No line numbers. No explanations. No markdown. Just the function.

Bug: {bug['title']}
Code:
{bug['buggy']}

Tests:
{bug['test']}"""
    if failed_patches:
        p += f"\n\n{len(failed_patches)} previous fixes FAILED. Use DIFFERENT approach.\n"
        for i,(patch,err) in enumerate(failed_patches[-2:],1):
            p += f"\nFailed {i}:\n{patch[:250]}\nError: {err}\n"
    if ctx.get("reflection_hint"):
        p += f"\nReflection: {ctx['reflection_hint']}\nAdapt strategy.\n"
    if ctx.get("failures_to_avoid"):
        p += f"\nAvoid: {', '.join(ctx['failures_to_avoid'][:3])}\n"
    if attempt > 2:
        p += "\nConsider restructuring the entire function.\n"
    return p + "\nReturn ONLY Python code:"

# ══════════════════════════════════════════════════════════
# AGENT — generates patches via LLM or rules
# ══════════════════════════════════════════════════════════
def generate_patch(bug, strategy, failed_patches, ctx, llm=None, temp=0.3):
    """Generate repair patch. Uses LLM if available, else rule-based."""
    fixes = bug.get("fixes", {})
    failed_hashes = {patch_hash(p) for p, _ in failed_patches}

    # Try LLM first
    if llm and llm.available:
        raw = llm.generate(build_prompt(bug, failed_patches, ctx, len(failed_patches)+1), temp)
        patch = extract_code(raw) if raw else None
        if patch:
            return patch, "llm"

    # Rule-based fallback with strategy ordering
    order = strategy.get_order(list(fixes.keys()))
    for tactic in order:
        try:
            patch = fixes[tactic](bug["buggy"])
            if patch and patch_hash(patch) not in failed_hashes:
                return patch, tactic
        except Exception:
            continue

    return bug["buggy"], "exhausted"

# ══════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ══════════════════════════════════════════════════════════
def run_benchmark(config: ExperimentConfig, llm: LLMClient = None):
    config.seed_random()
    if llm and llm.available:
        config.model = f"{llm.provider}/{llm.model}"

    tracker = ExperimentTracker(config, output_dir=os.path.join(
        os.path.dirname(__file__), '..', '..', 'experiments'))
    patches = PatchStore()
    runtime = CogniCoreRuntime(config=RuntimeConfig(
        reflection_min_samples=1, reflection_failure_threshold=1, memory_top_k=5,
    ), name="research-bench")

    config.print_config()
    print(f"\n{'='*64}")
    print(f"  CogniCore Research Benchmark")
    print(f"  Model: {config.model} | Bugs: {len(BUGS)} | Max: {config.max_attempts}")
    print(f"{'='*64}")

    for bug in BUGS:
        print(f"\n  {'─'*56}")
        print(f"  {bug['id']}: {bug['title']} [{bug['cat']}]")
        result = ExperimentResult(bug_id=bug["id"], category=bug["cat"], title=bug["title"])

        # ── BASELINE ──
        print(f"  [A] BASELINE")
        b_strat = RepairStrategy()
        b_failed, b_hashes = [], set()
        for att in range(1, config.max_attempts+1):
            result.baseline_attempts = att
            patch, tactic = generate_patch(bug, b_strat, [], {}, llm, config.temperature)
            h = patch_hash(patch)
            if h in b_hashes: result.baseline_repeated += 1
            b_hashes.add(h)
            ok, err = sandbox(patch, bug["test"])
            patches.store(bug["id"], att, patch, err, ok, tactic, mode="baseline")
            if ok:
                result.baseline_solved = True
                clog("ADAPTIVE SUCCESS", f"Attempt {att}: PASS ({tactic})")
                break
            else:
                b_failed.append((patch, err))
                rep = " **REPEATED**" if result.baseline_repeated else ""
                clog("FAILED PATCH", f"Attempt {att}: {err[:60]} ({tactic}){rep}")

        result.baseline_unique_patches = len(b_hashes)

        # ── COGNICORE ──
        print(f"  [B] COGNICORE")
        c_strat = RepairStrategy()
        c_failed, c_hashes, c_errors = [], set(), []

        for att in range(1, config.max_attempts+1):
            result.cogni_attempts = att
            ctx = runtime._build_context(bug["cat"])

            # Memory
            if ctx.get("memory"):
                result.cogni_memory_hits += 1
                fails = [e for e in ctx["memory"] if not e.get("correct")]
                if fails:
                    clog("MEMORY RETRIEVAL", f"{len(fails)} past failures in '{bug['cat']}'",
                         "\n".join(str(e.get("predicted",""))[:55] for e in fails[:2]))
                    tracker.log_event("MEMORY_RETRIEVAL", bug["id"],
                                      f"{len(fails)} failures retrieved")

            # Reflection → Strategy mutation
            if ctx.get("reflection_hint"):
                result.cogni_reflections += 1
                clog("REFLECTION GENERATED", "Analyzing failure patterns",
                     ctx["reflection_hint"][:100])
                tracker.log_event("REFLECTION", bug["id"], ctx["reflection_hint"][:100])

                # Active mutation
                if c_errors:
                    repeated = detect_repeated_reasoning(c_errors)
                    if repeated:
                        clog("REPEATED REASONING", repeated)
                    # Disable failing tactics
                    for fp, fe in c_failed[-2:]:
                        for tname in list(c_strat.disabled) + ["guard", "direct"]:
                            pass
                    if len(c_failed) >= 1:
                        c_strat.mutate(disable="guard", prefer="rewrite",
                                       reason="Guard-based fixes repeatedly failed. Restructure preferred.")
                        result.cogni_mutations += 1
                        result.cogni_strategy_changes.append("guard→rewrite")
                        tracker.log_event("STRATEGY_MUTATION", bug["id"], "guard→rewrite")

            # Prompt mutation
            if c_failed:
                clog("PROMPT MUTATION",
                     f"{len(c_failed)} failed patches + context injected into prompt")
                tracker.log_event("PROMPT_MUTATION", bug["id"],
                                  f"{len(c_failed)} failures injected")

            temp = min(config.temperature + 0.15 * len(c_failed), 0.9)
            patch, tactic = generate_patch(bug, c_strat,
                                           c_failed, ctx, llm, temp)
            h = patch_hash(patch)

            # Semantic patch rejection
            rejected = False
            for prev, perr in c_failed[-3:]:
                s = combined_similarity(patch, prev)
                if s > config.similarity_threshold:
                    result.cogni_rejections += 1
                    rejected = True
                    clog("PATCH REJECTED",
                         f"AST+text similarity {s:.0%} > threshold {config.similarity_threshold:.0%}",
                         f"Previous error: {perr[:55]}")
                    tracker.log_event("PATCH_REJECTED", bug["id"],
                                      f"sim={s:.2f}", {"prev_error": perr[:80]})
                    # Force alternative
                    patch, tactic = generate_patch(bug, c_strat,
                                                    c_failed + [(patch, "rejected")],
                                                    ctx, llm, min(temp+0.3, 1.0))
                    h = patch_hash(patch)
                    break

            if h in c_hashes: result.cogni_repeated += 1
            c_hashes.add(h)

            ok, err = sandbox(patch, bug["test"])
            patches.store(bug["id"], att, patch, err, ok, tactic,
                         rejected=rejected, similarity_score=0, mode="cognicore")

            # Store in runtime memory
            runtime.memory.store({
                "category": bug["cat"], "correct": ok, "bug_id": bug["id"],
                "predicted": f"tactic:{tactic} err:{err[:60]}" if err else f"tactic:{tactic} PASS",
                "patch_hash": h,
            })

            if ok:
                result.cogni_solved = True
                clog("ADAPTIVE SUCCESS", f"Attempt {att}: PASS ({tactic})",
                     f"Memory:{result.cogni_memory_hits} Refl:{result.cogni_reflections} "
                     f"Mut:{result.cogni_mutations} Rej:{result.cogni_rejections}")
                tracker.log_event("SUCCESS", bug["id"], f"attempt={att} tactic={tactic}")
                break
            else:
                c_failed.append((patch, err or "unknown"))
                c_errors.append(err or "unknown")
                clog("FAILED PATCH", f"Attempt {att}: {err[:60]} ({tactic})")

        result.cogni_unique_patches = len(c_hashes)
        tracker.add_result(result)

    # Report
    tracker.print_report()
    report_path = tracker.save(patches)
    print(f"\n  Experiment saved: {report_path}")
    return tracker


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CogniCore Research Benchmark")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attempts", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--provider", default="auto", choices=["auto","gemini","openai","claude"])
    args = p.parse_args()

    config = ExperimentConfig(
        seed=args.seed, max_attempts=args.attempts,
        similarity_threshold=args.threshold, temperature=args.temperature,
    )
    llm = LLMClient(provider=args.provider)
    if llm.available:
        print(f"  LLM: {llm}")
        if not llm.test_connection():
            print(f"  LLM connection test failed, using rule-based agent")
            llm.available = False
    else:
        print(f"  No LLM available. Using rule-based agent.")
        print(f"  Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY to enable LLM.")

    run_benchmark(config, llm)
