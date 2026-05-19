"""
Expanded SWE-bench tasks — 30 additional tasks (50 total).
Covers: concurrency, caching, serialization, boundary, state_mgmt,
        string_ops, collection_ops, io_handling, config, security.
"""

SWEBENCH_TASKS_V2 = [
  # ── CONCURRENCY (3 tasks) ──
  {"id":"SWE-v2-conc-001","repo":"python/cpython","category":"concurrency",
   "issue":"Thread-safe counter has race condition",
   "description":"Counter increment is not atomic, causes lost updates.",
   "fix_hint":"Use threading.Lock",
   "buggy_code":"import threading\nclass Counter:\n    def __init__(s):s.val=0\n    def inc(s):s.val+=1\n    def get(s):return s.val",
   "test_code":"import threading\nc=Counter()\nts=[threading.Thread(target=c.inc) for _ in range(100)]\nfor t in ts:t.start()\nfor t in ts:t.join()\nassert c.get()==100"},

  {"id":"SWE-v2-conc-002","repo":"python/cpython","category":"concurrency",
   "issue":"Bounded queue blocks forever on full",
   "description":"Put doesn't check capacity, get doesn't check empty.",
   "fix_hint":"Add capacity check and raise on full/empty",
   "buggy_code":"class BoundedQueue:\n    def __init__(s,cap):s.q=[];s.cap=cap\n    def put(s,v):s.q.append(v)\n    def get(s):return s.q.pop(0)\n    def size(s):return len(s.q)",
   "test_code":"q=BoundedQueue(2)\nq.put(1);q.put(2)\ntry:\n    q.put(3);assert False,'should raise'\nexcept (OverflowError,ValueError,Exception):pass\nassert q.get()==1\nassert q.size()==1"},

  {"id":"SWE-v2-conc-003","repo":"python/cpython","category":"concurrency",
   "issue":"Timeout decorator doesn't restore signal",
   "description":"Signal handler not reset after function completes.",
   "fix_hint":"Use try/finally to restore handler",
   "buggy_code":"import time\ndef with_timeout(fn, timeout):\n    start=time.time()\n    result=fn()\n    elapsed=time.time()-start\n    if elapsed>timeout:raise TimeoutError\n    return result",
   "test_code":"assert with_timeout(lambda:42,1)==42\nassert with_timeout(lambda:'ok',5)=='ok'\ntry:\n    with_timeout(lambda:time.sleep(0.1) or 1,0.001)\nexcept TimeoutError:pass"},

  # ── CACHING (3 tasks) ──
  {"id":"SWE-v2-cache-001","repo":"django/django","category":"caching",
   "issue":"LRU cache doesn't evict oldest",
   "description":"Cache grows unbounded, no eviction on capacity.",
   "fix_hint":"Evict oldest when full",
   "buggy_code":"class LRUCache:\n    def __init__(s,cap):s.cap=cap;s.d={}\n    def get(s,k):return s.d.get(k)\n    def put(s,k,v):s.d[k]=v",
   "test_code":"c=LRUCache(2)\nc.put('a',1);c.put('b',2);c.put('c',3)\nassert c.get('c')==3\nassert c.get('a') is None\nassert c.get('b')==2"},

  {"id":"SWE-v2-cache-002","repo":"django/django","category":"caching",
   "issue":"TTL cache doesn't expire entries",
   "description":"Cache entries never expire regardless of TTL.",
   "fix_hint":"Check timestamp on get",
   "buggy_code":"import time\nclass TTLCache:\n    def __init__(s,ttl):s.ttl=ttl;s.d={}\n    def set(s,k,v):s.d[k]=(v,time.time())\n    def get(s,k):\n        if k in s.d:return s.d[k][0]\n        return None",
   "test_code":"c=TTLCache(0.05)\nc.set('x',1)\nassert c.get('x')==1\nimport time;time.sleep(0.1)\nassert c.get('x') is None"},

  {"id":"SWE-v2-cache-003","repo":"django/django","category":"caching",
   "issue":"Memoize decorator ignores kwargs",
   "description":"Cache key only uses positional args.",
   "fix_hint":"Include kwargs in cache key",
   "buggy_code":"def memoize(fn):\n    cache={}\n    def wrapper(*args):\n        if args not in cache:cache[args]=fn(*args)\n        return cache[args]\n    return wrapper",
   "test_code":"@memoize\ndef add(a,b=0):return a+b\nassert add(1)==1\nassert add(1,b=2)==3\nassert add(1,b=3)==4"},

  # ── SERIALIZATION (3 tasks) ──
  {"id":"SWE-v2-ser-001","repo":"psf/requests","category":"serialization",
   "issue":"JSON serializer crashes on datetime",
   "description":"Default JSON encoder can't handle datetime objects.",
   "fix_hint":"Add datetime handler",
   "buggy_code":"import json\ndef to_json(obj):\n    return json.dumps(obj)",
   "test_code":"import datetime\nassert to_json({'a':1})=='{\"a\": 1}'\nr=to_json({'t':datetime.datetime(2024,1,1)})\nassert '2024' in r"},

  {"id":"SWE-v2-ser-002","repo":"psf/requests","category":"serialization",
   "issue":"CSV writer doesn't escape commas in values",
   "description":"Values containing commas break CSV format.",
   "fix_hint":"Quote values containing commas",
   "buggy_code":"def to_csv_row(values):\n    return ','.join(str(v) for v in values)",
   "test_code":"assert to_csv_row([1,2,3])=='1,2,3'\nr=to_csv_row(['hello','world, here',42])\nassert r.count(',')==2 or '\"world, here\"' in r"},

  {"id":"SWE-v2-ser-003","repo":"psf/requests","category":"serialization",
   "issue":"Config parser loses types on round-trip",
   "description":"All values become strings after parse/unparse.",
   "fix_hint":"Infer types on parse",
   "buggy_code":"def parse_config(text):\n    d={}\n    for line in text.strip().split('\\n'):\n        if '=' in line:\n            k,v=line.split('=',1)\n            d[k.strip()]=v.strip()\n    return d",
   "test_code":"c=parse_config('port=8080\\ndebug=true\\nname=app')\nassert c['port']==8080 or c['port']=='8080'\nassert c['debug']==True or c['debug']=='true'\nassert c['name']=='app'"},

  # ── BOUNDARY (3 tasks) ──
  {"id":"SWE-v2-bound-001","repo":"sympy/sympy","category":"boundary",
   "issue":"Range function doesn't handle negative step",
   "description":"Custom range ignores negative step direction.",
   "fix_hint":"Check step direction",
   "buggy_code":"def custom_range(start,stop,step=1):\n    result=[]\n    i=start\n    while i<stop:\n        result.append(i)\n        i+=step\n    return result",
   "test_code":"assert custom_range(0,5)==[0,1,2,3,4]\nassert custom_range(5,0,-1)==[5,4,3,2,1]\nassert custom_range(0,0)==[]"},

  {"id":"SWE-v2-bound-002","repo":"sympy/sympy","category":"boundary",
   "issue":"Clamp doesn't handle inverted bounds",
   "description":"min_val > max_val not handled.",
   "fix_hint":"Swap bounds if inverted",
   "buggy_code":"def clamp(val,min_val,max_val):\n    if val<min_val:return min_val\n    if val>max_val:return max_val\n    return val",
   "test_code":"assert clamp(5,0,10)==5\nassert clamp(-1,0,10)==0\nassert clamp(15,0,10)==10\nassert clamp(5,10,0)==5"},

  {"id":"SWE-v2-bound-003","repo":"sympy/sympy","category":"boundary",
   "issue":"Pagination calculates wrong total pages",
   "description":"Division doesn't round up for partial pages.",
   "fix_hint":"Use ceiling division",
   "buggy_code":"def paginate(total,per_page):\n    pages=total//per_page\n    return {'total':total,'pages':pages,'per_page':per_page}",
   "test_code":"assert paginate(10,5)['pages']==2\nassert paginate(11,5)['pages']==3\nassert paginate(0,5)['pages']==0\nassert paginate(1,10)['pages']==1"},

  # ── STATE MANAGEMENT (3 tasks) ──
  {"id":"SWE-v2-state-001","repo":"pallets/flask","category":"state_mgmt",
   "issue":"Undo stack doesn't limit history",
   "description":"Undo history grows unbounded.",
   "fix_hint":"Trim to max_size",
   "buggy_code":"class UndoStack:\n    def __init__(s,max_size=10):s.stack=[];s.max=max_size\n    def push(s,state):s.stack.append(state)\n    def undo(s):return s.stack.pop() if s.stack else None\n    def size(s):return len(s.stack)",
   "test_code":"u=UndoStack(3)\nfor i in range(10):u.push(i)\nassert u.size()<=3\nassert u.undo()==9"},

  {"id":"SWE-v2-state-002","repo":"pallets/flask","category":"state_mgmt",
   "issue":"State machine allows invalid transitions",
   "description":"No transition validation.",
   "fix_hint":"Check valid transitions map",
   "buggy_code":"class StateMachine:\n    def __init__(s,initial,transitions):\n        s.state=initial;s.trans=transitions\n    def transition(s,event):\n        s.state=event\n        return s.state",
   "test_code":"sm=StateMachine('idle',{'idle':['running'],'running':['idle','error']})\nassert sm.transition('running')=='running'\ntry:\n    sm.transition('idle');sm.transition('error')\n    assert False,'invalid'\nexcept (ValueError,Exception):pass"},

  {"id":"SWE-v2-state-003","repo":"pallets/flask","category":"state_mgmt",
   "issue":"Event emitter doesn't handle unsubscribe",
   "description":"Can't remove event listeners.",
   "fix_hint":"Add off/remove method",
   "buggy_code":"class EventEmitter:\n    def __init__(s):s.listeners={}\n    def on(s,evt,fn):\n        s.listeners.setdefault(evt,[]).append(fn)\n    def emit(s,evt,*args):\n        for fn in s.listeners.get(evt,[]):fn(*args)",
   "test_code":"e=EventEmitter()\nresults=[]\ne.on('click',lambda:results.append('a'))\ncb=lambda:results.append('b')\ne.on('click',cb)\ne.emit('click')\nassert len(results)==2\ne.off('click',cb)\ne.emit('click')\nassert len(results)==3"},

  # ── STRING OPS (3 tasks) ──
  {"id":"SWE-v2-str-001","repo":"python/cpython","category":"string_ops",
   "issue":"Slug generator doesn't handle unicode",
   "description":"Non-ASCII chars not transliterated.",
   "fix_hint":"Use unicodedata.normalize",
   "buggy_code":"import re\ndef slugify(text):\n    text=text.lower().strip()\n    return re.sub(r'[^a-z0-9]+','-',text).strip('-')",
   "test_code":"assert slugify('Hello World')=='hello-world'\nassert slugify('  spaces  ')=='spaces'\nassert slugify('a--b')=='a-b'\nassert len(slugify(''))== 0"},

  {"id":"SWE-v2-str-002","repo":"python/cpython","category":"string_ops",
   "issue":"Word wrap breaks mid-word",
   "description":"Wrapping doesn't respect word boundaries.",
   "fix_hint":"Break at last space before width",
   "buggy_code":"def word_wrap(text,width):\n    lines=[]\n    for i in range(0,len(text),width):\n        lines.append(text[i:i+width])\n    return '\\n'.join(lines)",
   "test_code":"r=word_wrap('hello world foo bar',10)\nlines=r.split('\\n')\nassert all(len(l)<=10 for l in lines)\nassert 'hello' in lines[0]"},

  {"id":"SWE-v2-str-003","repo":"python/cpython","category":"string_ops",
   "issue":"Template substitution doesn't escape",
   "description":"Template vars not escaped for HTML.",
   "fix_hint":"Escape < > & in values",
   "buggy_code":"def render_template(tmpl,**kwargs):\n    result=tmpl\n    for k,v in kwargs.items():\n        result=result.replace('{'+k+'}',str(v))\n    return result",
   "test_code":"assert render_template('{name} is {age}',name='Bob',age=30)=='Bob is 30'\nr=render_template('{x}',x='<script>alert(1)</script>')\nassert '<script>' not in r"},

  # ── COLLECTION OPS (3 tasks) ──
  {"id":"SWE-v2-coll-001","repo":"python/cpython","category":"collection_ops",
   "issue":"Flatten doesn't handle nested iterables",
   "description":"Only flattens one level deep.",
   "fix_hint":"Recursive flatten",
   "buggy_code":"def flatten(lst):\n    result=[]\n    for item in lst:\n        if isinstance(item,list):result.extend(item)\n        else:result.append(item)\n    return result",
   "test_code":"assert flatten([1,[2,3],4])==[1,2,3,4]\nassert flatten([1,[2,[3,4]],5])==[1,2,3,4,5]\nassert flatten([])==[]"},

  {"id":"SWE-v2-coll-002","repo":"python/cpython","category":"collection_ops",
   "issue":"Group-by doesn't preserve order",
   "description":"Groups returned in arbitrary order.",
   "fix_hint":"Use OrderedDict or maintain insertion order",
   "buggy_code":"def group_by(items,key_fn):\n    groups={}\n    for item in items:\n        k=key_fn(item)\n        groups.setdefault(k,[]).append(item)\n    return groups",
   "test_code":"r=group_by([1,2,3,4,5,6],lambda x:x%2)\nassert r[0]==[2,4,6]\nassert r[1]==[1,3,5]\nassert list(r.keys())==[1,0]"},

  {"id":"SWE-v2-coll-003","repo":"python/cpython","category":"collection_ops",
   "issue":"Merge dicts doesn't handle conflicts",
   "description":"Later values silently overwrite without merge strategy.",
   "fix_hint":"Add conflict resolution callback",
   "buggy_code":"def deep_merge(d1,d2):\n    result=d1.copy()\n    result.update(d2)\n    return result",
   "test_code":"assert deep_merge({'a':1},{'b':2})=={'a':1,'b':2}\nr=deep_merge({'a':{'x':1}},{'a':{'y':2}})\nassert r['a']=={'x':1,'y':2}"},

  # ── IO HANDLING (3 tasks) ──
  {"id":"SWE-v2-io-001","repo":"astropy/astropy","category":"io_handling",
   "issue":"File reader doesn't close on error",
   "description":"Exception during read leaves file handle open.",
   "fix_hint":"Use context manager",
   "buggy_code":"def read_lines(path):\n    f=open(path)\n    lines=f.readlines()\n    f.close()\n    return [l.strip() for l in lines]",
   "test_code":"import tempfile,os\nwith tempfile.NamedTemporaryFile(mode='w',suffix='.txt',delete=False) as f:\n    f.write('a\\nb\\nc');p=f.name\nassert read_lines(p)==['a','b','c']\nos.unlink(p)\ntry:read_lines('/nonexistent/file')\nexcept:pass"},

  {"id":"SWE-v2-io-002","repo":"astropy/astropy","category":"io_handling",
   "issue":"Path builder doesn't handle Windows paths",
   "description":"Forward slashes only, no OS awareness.",
   "fix_hint":"Use os.path.join",
   "buggy_code":"def build_path(*parts):\n    return '/'.join(parts)",
   "test_code":"import os\nr=build_path('home','user','file.txt')\nassert 'file.txt' in r\nassert os.sep in r or '/' in r"},

  {"id":"SWE-v2-io-003","repo":"astropy/astropy","category":"io_handling",
   "issue":"Temp file not cleaned up",
   "description":"Temporary files left on disk after use.",
   "fix_hint":"Use tempfile context manager or atexit cleanup",
   "buggy_code":"import tempfile\ndef create_temp(content):\n    f=tempfile.NamedTemporaryFile(mode='w',delete=False)\n    f.write(content);f.close()\n    return f.name",
   "test_code":"import os\np=create_temp('test data')\nassert os.path.exists(p)\ndata=open(p).read()\nassert data=='test data'\nos.unlink(p)"},

  # ── CONFIG (2 tasks) ──
  {"id":"SWE-v2-cfg-001","repo":"pytest-dev/pytest","category":"config",
   "issue":"Env var expansion doesn't handle defaults",
   "description":"Missing env vars raise KeyError instead of using default.",
   "fix_hint":"Support ${VAR:-default} syntax",
   "buggy_code":"import os,re\ndef expand_env(text):\n    def repl(m):return os.environ[m.group(1)]\n    return re.sub(r'\\$\\{(\\w+)\\}',repl,text)",
   "test_code":"import os\nos.environ['TEST_X']='hello'\nassert expand_env('${TEST_X}')=='hello'\nassert expand_env('no vars')=='no vars'\ntry:\n    r=expand_env('${NONEXISTENT_VAR_XYZ}')\n    assert r=='${NONEXISTENT_VAR_XYZ}' or r==''\nexcept KeyError:assert False,'should not raise'"},

  {"id":"SWE-v2-cfg-002","repo":"pytest-dev/pytest","category":"config",
   "issue":"Dotted key access crashes on missing intermediate",
   "description":"Nested dict access doesn't handle missing keys.",
   "fix_hint":"Return None for missing intermediates",
   "buggy_code":"def get_nested(d,key_path):\n    keys=key_path.split('.')\n    val=d\n    for k in keys:\n        val=val[k]\n    return val",
   "test_code":"d={'a':{'b':{'c':1}}}\nassert get_nested(d,'a.b.c')==1\nassert get_nested(d,'a.b.missing') is None\nassert get_nested(d,'x.y') is None"},

  # ── SECURITY (2 tasks) ──
  {"id":"SWE-v2-sec-001","repo":"django/django","category":"security",
   "issue":"SQL query builder doesn't escape inputs",
   "description":"String interpolation allows SQL injection.",
   "fix_hint":"Use parameterized queries",
   "buggy_code":"def build_query(table,filters):\n    where=' AND '.join(f\"{k}='{v}'\" for k,v in filters.items())\n    return f'SELECT * FROM {table} WHERE {where}'",
   "test_code":"q=build_query('users',{'name':'bob'})\nassert 'bob' in q\nq2=build_query('users',{'name':\"'; DROP TABLE users; --\"})\nassert 'DROP' not in q2 or '?' in q2 or '$' in q2"},

  {"id":"SWE-v2-sec-002","repo":"django/django","category":"security",
   "issue":"Password hasher uses plain comparison",
   "description":"String comparison is timing-attack vulnerable.",
   "fix_hint":"Use hmac.compare_digest",
   "buggy_code":"import hashlib\ndef hash_password(pw):\n    return hashlib.sha256(pw.encode()).hexdigest()\ndef verify_password(pw,hashed):\n    return hash_password(pw)==hashed",
   "test_code":"h=hash_password('secret123')\nassert verify_password('secret123',h)==True\nassert verify_password('wrong',h)==False\nassert len(h)==64"},
]
