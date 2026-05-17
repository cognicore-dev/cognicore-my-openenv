"""
Real-world code bugs — actual Python bugs that appear in production.

Each case has buggy code, the fix, the bug category, and explanation.
"""

from __future__ import annotations
from typing import Any, Dict, List

REAL_CODE_CASES: List[Dict[str, Any]] = [
    # ── Off-by-one errors ───────────────────────────────────────────
    {
        "code": "def get_last_n(items, n):\n    return items[len(items)-n:len(items)-1]",
        "bug_type": "off_by_one",
        "fix": "def get_last_n(items, n):\n    return items[len(items)-n:]",
        "explanation": "Off-by-one: `len(items)-1` excludes the last element. Should be `len(items)` or just omit the end index.",
        "language": "python",
    },
    {
        "code": "for i in range(1, len(users)):\n    process(users[i])",
        "bug_type": "off_by_one",
        "fix": "for i in range(len(users)):\n    process(users[i])",
        "explanation": "Starts at index 1, skipping the first user. Range should start at 0.",
        "language": "python",
    },

    # ── Mutable default arguments ───────────────────────────────────
    {
        "code": "def add_item(item, items=[]):\n    items.append(item)\n    return items",
        "bug_type": "mutable_default",
        "fix": "def add_item(item, items=None):\n    if items is None:\n        items = []\n    items.append(item)\n    return items",
        "explanation": "Mutable default argument `[]` is shared across all calls. Use `None` and create a new list inside.",
        "language": "python",
    },
    {
        "code": "def create_user(name, metadata={}):\n    metadata['name'] = name\n    return metadata",
        "bug_type": "mutable_default",
        "fix": "def create_user(name, metadata=None):\n    if metadata is None:\n        metadata = {}\n    metadata['name'] = name\n    return metadata",
        "explanation": "Mutable default dict is shared between calls. Second call overwrites first user's data.",
        "language": "python",
    },

    # ── Variable scope / closure bugs ───────────────────────────────
    {
        "code": "functions = []\nfor i in range(5):\n    functions.append(lambda: i)\n# All return 4",
        "bug_type": "closure_bug",
        "fix": "functions = []\nfor i in range(5):\n    functions.append(lambda i=i: i)\n# Returns 0, 1, 2, 3, 4",
        "explanation": "Late binding closure: all lambdas capture the same variable `i`, which ends at 4. Fix with default argument.",
        "language": "python",
    },

    # ── Exception handling ──────────────────────────────────────────
    {
        "code": "try:\n    result = int(user_input)\nexcept:\n    pass",
        "bug_type": "bare_except",
        "fix": "try:\n    result = int(user_input)\nexcept ValueError:\n    result = 0  # or handle appropriately",
        "explanation": "Bare `except` catches KeyboardInterrupt and SystemExit. Always catch specific exceptions.",
        "language": "python",
    },
    {
        "code": "try:\n    data = json.loads(response)\n    return data['results']\nexcept Exception as e:\n    return None",
        "bug_type": "silent_failure",
        "fix": "try:\n    data = json.loads(response)\n    return data['results']\nexcept (json.JSONDecodeError, KeyError) as e:\n    logger.error(f'Failed to parse response: {e}')\n    return None",
        "explanation": "Catches all exceptions silently. Masks real bugs. Log the error and catch specific types.",
        "language": "python",
    },

    # ── Race conditions / concurrency ───────────────────────────────
    {
        "code": "balance = 100\ndef withdraw(amount):\n    global balance\n    if balance >= amount:\n        balance -= amount\n        return True\n    return False",
        "bug_type": "race_condition",
        "fix": "import threading\nbalance = 100\nlock = threading.Lock()\ndef withdraw(amount):\n    global balance\n    with lock:\n        if balance >= amount:\n            balance -= amount\n            return True\n    return False",
        "explanation": "Race condition: two threads can both check `balance >= amount` before either subtracts. Use a lock.",
        "language": "python",
    },

    # ── SQL injection ───────────────────────────────────────────────
    {
        "code": "def get_user(username):\n    query = f\"SELECT * FROM users WHERE name = '{username}'\"\n    return db.execute(query)",
        "bug_type": "sql_injection",
        "fix": "def get_user(username):\n    query = \"SELECT * FROM users WHERE name = ?\"\n    return db.execute(query, (username,))",
        "explanation": "SQL injection: user input is directly interpolated into SQL. Use parameterized queries.",
        "language": "python",
    },
    {
        "code": "def search(term):\n    cursor.execute(\"SELECT * FROM products WHERE name LIKE '%\" + term + \"%'\")",
        "bug_type": "sql_injection",
        "fix": "def search(term):\n    cursor.execute(\"SELECT * FROM products WHERE name LIKE ?\", (f'%{term}%',))",
        "explanation": "String concatenation in SQL query allows injection. Use parameterized queries with placeholders.",
        "language": "python",
    },

    # ── Type confusion ──────────────────────────────────────────────
    {
        "code": "def calculate_total(prices):\n    return sum(prices) / len(prices) * 100",
        "bug_type": "type_error",
        "fix": "def calculate_total(prices):\n    if not prices:\n        return 0.0\n    return sum(prices) / len(prices) * 100",
        "explanation": "ZeroDivisionError when `prices` is empty. Always check for empty sequences before dividing.",
        "language": "python",
    },
    {
        "code": "user_age = input('Enter age: ')\nif user_age > 18:\n    print('Adult')",
        "bug_type": "type_error",
        "fix": "user_age = int(input('Enter age: '))\nif user_age > 18:\n    print('Adult')",
        "explanation": "`input()` returns a string. Comparing string '5' > 18 raises TypeError in Python 3. Convert to int.",
        "language": "python",
    },

    # ── Resource leaks ──────────────────────────────────────────────
    {
        "code": "def read_config(path):\n    f = open(path)\n    data = json.load(f)\n    return data",
        "bug_type": "resource_leak",
        "fix": "def read_config(path):\n    with open(path) as f:\n        data = json.load(f)\n    return data",
        "explanation": "File handle is never closed. If json.load raises, the file leaks. Use `with` statement.",
        "language": "python",
    },
    {
        "code": "conn = sqlite3.connect('db.sqlite')\ncursor = conn.cursor()\ncursor.execute('SELECT * FROM users')\nresults = cursor.fetchall()\n# conn.close() missing",
        "bug_type": "resource_leak",
        "fix": "with sqlite3.connect('db.sqlite') as conn:\n    cursor = conn.cursor()\n    cursor.execute('SELECT * FROM users')\n    results = cursor.fetchall()",
        "explanation": "Database connection is never closed. Use context manager for automatic cleanup.",
        "language": "python",
    },

    # ── Logic errors ────────────────────────────────────────────────
    {
        "code": "def is_palindrome(s):\n    return s == s.reverse()",
        "bug_type": "logic_error",
        "fix": "def is_palindrome(s):\n    return s == s[::-1]",
        "explanation": "Strings don't have `.reverse()`. Use slicing `[::-1]` for reversal.",
        "language": "python",
    },
    {
        "code": "def flatten(nested):\n    result = []\n    for item in nested:\n        if type(item) == list:\n            result.extend(item)\n        else:\n            result.append(item)\n    return result",
        "bug_type": "logic_error",
        "fix": "def flatten(nested):\n    result = []\n    for item in nested:\n        if isinstance(item, (list, tuple)):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
        "explanation": "1) Uses `type()` instead of `isinstance()` (misses subclasses). 2) Only flattens one level. Needs recursion.",
        "language": "python",
    },

    # ── Security: hardcoded secrets ─────────────────────────────────
    {
        "code": "import os\nAPI_KEY = os.environ.get('API_KEY')\ndef call_api(data):\n    headers = {'Authorization': f'Bearer {API_KEY}'}\n    return requests.post(URL, json=data, headers=headers)",
        "bug_type": "hardcoded_secret",
        "fix": "import os\nAPI_KEY = os.environ.get('API_KEY')\ndef call_api(data):\n    if not API_KEY:\n        raise ValueError('API_KEY environment variable not set')\n    headers = {'Authorization': f'Bearer {API_KEY}'}\n    return requests.post(URL, json=data, headers=headers)",
        "explanation": "API key hardcoded in source. Will be leaked if code is committed to git. Use environment variables.",
        "language": "python",
    },

    # ── Async bugs ──────────────────────────────────────────────────
    {
        "code": "async def fetch_all(urls):\n    results = []\n    for url in urls:\n        result = await fetch(url)\n        results.append(result)\n    return results",
        "bug_type": "async_antipattern",
        "fix": "async def fetch_all(urls):\n    tasks = [fetch(url) for url in urls]\n    return await asyncio.gather(*tasks)",
        "explanation": "Sequential awaits defeat the purpose of async. Use `asyncio.gather()` for concurrent execution.",
        "language": "python",
    },
]
