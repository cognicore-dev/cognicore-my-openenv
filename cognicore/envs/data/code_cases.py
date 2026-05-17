"""
Code Debugging Dataset — 30 buggy code snippets across three difficulty levels.

Easy (10):   Syntax errors, typos, off-by-one — obvious fixes
Medium (10): Logic errors, wrong algorithms, edge cases
Hard (10):   Subtle bugs — race conditions, floating point, security flaws
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CodeCase:
    """A single code debugging problem."""

    id: str
    language: str
    buggy_code: str
    bug_description: str
    bug_line: int  # 1-indexed line number where the bug is
    fix_type: str  # The category of fix needed
    category: str
    difficulty: str
    explanation: str
    correct_fix: str  # Description of the correct fix


# ═══════════════════════════════════════════════════════════════
# EASY — Syntax & Obvious Errors
# ═══════════════════════════════════════════════════════════════

EASY_CASES = [
    CodeCase(
        id="code_easy_01",
        language="python",
        buggy_code="def add(a, b):\n    return a - b",
        bug_description="Function named 'add' but subtracts instead of adding.",
        bug_line=2,
        fix_type="wrong_operator",
        category="operator_error",
        difficulty="easy",
        explanation="The minus operator should be plus.",
        correct_fix="Change `a - b` to `a + b`",
    ),
    CodeCase(
        id="code_easy_02",
        language="python",
        buggy_code="def greet(name):\n    print('Hello, ' + name\n",
        bug_description="Missing closing parenthesis on print statement.",
        bug_line=2,
        fix_type="syntax_error",
        category="syntax",
        difficulty="easy",
        explanation="The print call is missing a closing parenthesis.",
        correct_fix="Add `)` at end of line 2",
    ),
    CodeCase(
        id="code_easy_03",
        language="python",
        buggy_code="numbers = [1, 2, 3, 4, 5]\nfor i in range(1, len(numbers)):\n    print(numbers[i])",
        bug_description="Loop starts at index 1 instead of 0, skipping the first element.",
        bug_line=2,
        fix_type="off_by_one",
        category="off_by_one",
        difficulty="easy",
        explanation="range(1, ...) skips index 0. Should be range(0, ...) or range(len(numbers)).",
        correct_fix="Change `range(1, len(numbers))` to `range(len(numbers))`",
    ),
    CodeCase(
        id="code_easy_04",
        language="python",
        buggy_code="def is_even(n):\n    return n % 2 == 1",
        bug_description="Returns True for odd numbers instead of even.",
        bug_line=2,
        fix_type="wrong_comparison",
        category="logic_error",
        difficulty="easy",
        explanation="n % 2 == 1 checks for odd. Should be n % 2 == 0 for even.",
        correct_fix="Change `n % 2 == 1` to `n % 2 == 0`",
    ),
    CodeCase(
        id="code_easy_05",
        language="python",
        buggy_code="def reverse_string(s):\n    return s[1:]",
        bug_description="Slicing removes first character instead of reversing.",
        bug_line=2,
        fix_type="wrong_slice",
        category="string_manipulation",
        difficulty="easy",
        explanation="s[1:] removes the first char. s[::-1] reverses.",
        correct_fix="Change `s[1:]` to `s[::-1]`",
    ),
    CodeCase(
        id="code_easy_06",
        language="python",
        buggy_code="total = 0\nnums = [10, 20, 30]\nfor n in nums:\n    total = n\nprint(total)",
        bug_description="Assignment instead of accumulation — ends up with only the last value.",
        bug_line=4,
        fix_type="missing_operator",
        category="accumulation",
        difficulty="easy",
        explanation="Should use += to accumulate, not = which overwrites.",
        correct_fix="Change `total = n` to `total += n`",
    ),
    CodeCase(
        id="code_easy_07",
        language="python",
        buggy_code="def factorial(n):\n    result = 0\n    for i in range(1, n + 1):\n        result *= i\n    return result",
        bug_description="Initializes result to 0 — multiplication by 0 always gives 0.",
        bug_line=2,
        fix_type="wrong_initial_value",
        category="initialization",
        difficulty="easy",
        explanation="result starts at 0, so result *= i is always 0. Should start at 1.",
        correct_fix="Change `result = 0` to `result = 1`",
    ),
    CodeCase(
        id="code_easy_08",
        language="python",
        buggy_code='name = "Alice"\nif name = "Alice":\n    print("Found Alice!")',
        bug_description="Single = (assignment) instead of == (comparison) in if statement.",
        bug_line=2,
        fix_type="syntax_error",
        category="syntax",
        difficulty="easy",
        explanation="Python requires == for comparison, = is assignment.",
        correct_fix='Change `name = "Alice"` to `name == "Alice"` in the if statement',
    ),
    CodeCase(
        id="code_easy_09",
        language="python",
        buggy_code="def max_of_two(a, b):\n    if a > b:\n        return b\n    return a",
        bug_description="Returns the wrong value — returns b when a is larger.",
        bug_line=3,
        fix_type="swapped_return",
        category="logic_error",
        difficulty="easy",
        explanation="When a > b, should return a (the larger), not b.",
        correct_fix="Change `return b` to `return a` on line 3",
    ),
    CodeCase(
        id="code_easy_10",
        language="python",
        buggy_code="fruits = ['apple', 'banana', 'cherry']\nprint(fruits[3])",
        bug_description="Index out of range — list has 3 items (indices 0-2), accessing index 3.",
        bug_line=2,
        fix_type="index_error",
        category="index_error",
        difficulty="easy",
        explanation="fruits[3] is out of bounds. Last valid index is 2.",
        correct_fix="Change `fruits[3]` to `fruits[2]` to access last element",
    ),
]


# ═══════════════════════════════════════════════════════════════
# MEDIUM — Logic & Algorithm Errors
# ═══════════════════════════════════════════════════════════════

MEDIUM_CASES = [
    CodeCase(
        id="code_med_01",
        language="python",
        buggy_code="def binary_search(arr, target):\n    low, high = 0, len(arr)\n    while low < high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid\n        else:\n            high = mid\n    return -1",
        bug_description="Infinite loop: when arr[mid] < target, low = mid doesn't advance (should be mid + 1).",
        bug_line=8,
        fix_type="infinite_loop",
        category="algorithm_error",
        difficulty="medium",
        explanation="low = mid can cause infinite loop when high - low = 1. Should be low = mid + 1.",
        correct_fix="Change `low = mid` to `low = mid + 1`",
    ),
    CodeCase(
        id="code_med_02",
        language="python",
        buggy_code="def flatten(nested_list):\n    result = []\n    for item in nested_list:\n        if type(item) == list:\n            result.append(flatten(item))\n        else:\n            result.append(item)\n    return result",
        bug_description="append() instead of extend() for recursive flattening — creates nested lists.",
        bug_line=5,
        fix_type="wrong_method",
        category="recursion",
        difficulty="medium",
        explanation="append adds the sublist as a single element. extend merges it flat.",
        correct_fix="Change `result.append(flatten(item))` to `result.extend(flatten(item))`",
    ),
    CodeCase(
        id="code_med_03",
        language="python",
        buggy_code="def remove_duplicates(lst):\n    seen = []\n    for item in lst:\n        if item in seen:\n            seen.append(item)\n    return seen",
        bug_description="Logic is inverted — adds items that ARE already seen instead of items NOT seen.",
        bug_line=4,
        fix_type="inverted_condition",
        category="logic_error",
        difficulty="medium",
        explanation="Should be `if item not in seen` to add unique items.",
        correct_fix="Change `if item in seen` to `if item not in seen`",
    ),
    CodeCase(
        id="code_med_04",
        language="python",
        buggy_code="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 3)",
        bug_description="Recursive call uses n-3 instead of n-2.",
        bug_line=4,
        fix_type="wrong_parameter",
        category="recursion",
        difficulty="medium",
        explanation="Fibonacci recurrence is f(n-1) + f(n-2), not f(n-1) + f(n-3).",
        correct_fix="Change `fibonacci(n - 3)` to `fibonacci(n - 2)`",
    ),
    CodeCase(
        id="code_med_05",
        language="python",
        buggy_code="def count_words(sentence):\n    words = sentence.split(',')\n    return len(words)",
        bug_description="Splits on comma instead of whitespace — counts comma-separated segments, not words.",
        bug_line=2,
        fix_type="wrong_delimiter",
        category="string_manipulation",
        difficulty="medium",
        explanation="split(',') splits on commas. split() or split(' ') splits on whitespace for word counting.",
        correct_fix="Change `split(',')` to `split()`",
    ),
    CodeCase(
        id="code_med_06",
        language="python",
        buggy_code="def merge_sorted(a, b):\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    return result",
        bug_description="Missing the remaining elements — when one list is exhausted, the rest of the other is lost.",
        bug_line=11,
        fix_type="missing_code",
        category="algorithm_error",
        difficulty="medium",
        explanation="After the while loop, remaining elements from a[i:] or b[j:] must be appended.",
        correct_fix="Add `result.extend(a[i:])` and `result.extend(b[j:])` before return",
    ),
    CodeCase(
        id="code_med_07",
        language="python",
        buggy_code="def is_palindrome(s):\n    s = s.lower()\n    return s == s.reverse()",
        bug_description="Strings don't have a .reverse() method — should use slicing [::-1].",
        bug_line=3,
        fix_type="wrong_method",
        category="api_misuse",
        difficulty="medium",
        explanation="str has no reverse() method. Use s[::-1] or reversed().",
        correct_fix="Change `s.reverse()` to `s[::-1]`",
    ),
    CodeCase(
        id="code_med_08",
        language="python",
        buggy_code="def safe_divide(a, b):\n    try:\n        return a / b\n    except ValueError:\n        return 0",
        bug_description="Catches ValueError but division by zero raises ZeroDivisionError.",
        bug_line=4,
        fix_type="wrong_exception",
        category="error_handling",
        difficulty="medium",
        explanation="Division by zero raises ZeroDivisionError, not ValueError.",
        correct_fix="Change `except ValueError` to `except ZeroDivisionError`",
    ),
    CodeCase(
        id="code_med_09",
        language="python",
        buggy_code="def power(base, exp):\n    if exp == 0:\n        return 1\n    return base * power(base, exp)",
        bug_description="Infinite recursion — exp never decreases in recursive call.",
        bug_line=4,
        fix_type="missing_decrement",
        category="recursion",
        difficulty="medium",
        explanation="Should be power(base, exp - 1) to converge toward base case.",
        correct_fix="Change `power(base, exp)` to `power(base, exp - 1)`",
    ),
    CodeCase(
        id="code_med_10",
        language="python",
        buggy_code="def find_max(lst):\n    if not lst:\n        return None\n    max_val = 0\n    for val in lst:\n        if val > max_val:\n            max_val = val\n    return max_val",
        bug_description="Initializes max to 0 — fails for lists with all negative numbers.",
        bug_line=4,
        fix_type="wrong_initial_value",
        category="edge_case",
        difficulty="medium",
        explanation="max_val = 0 means all negative values are missed. Should be lst[0] or float('-inf').",
        correct_fix="Change `max_val = 0` to `max_val = lst[0]`",
    ),
]


# ═══════════════════════════════════════════════════════════════
# HARD — Subtle & Security Bugs
# ═══════════════════════════════════════════════════════════════

HARD_CASES = [
    CodeCase(
        id="code_hard_01",
        language="python",
        buggy_code="def memoize(func):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = func(*args)\n        return cache[args]\n    return wrapper\n\n@memoize\ndef add_to_list(item, lst=[]):\n    lst.append(item)\n    return lst",
        bug_description="Mutable default argument — the default list is shared across all calls.",
        bug_line=10,
        fix_type="mutable_default",
        category="mutable_default",
        difficulty="hard",
        explanation="lst=[] is created once at function definition, not per-call. Should use None and create inside.",
        correct_fix="Change `lst=[]` to `lst=None` and add `if lst is None: lst = []`",
    ),
    CodeCase(
        id="code_hard_02",
        language="python",
        buggy_code="import os\n\ndef read_user_file(username, filename):\n    path = f'/data/users/{username}/{filename}'\n    with open(path) as f:\n        return f.read()",
        bug_description="Path traversal vulnerability — user can pass '../../../etc/passwd' as filename.",
        bug_line=4,
        fix_type="security_vulnerability",
        category="security",
        difficulty="hard",
        explanation="No sanitization of filename allows directory traversal attacks.",
        correct_fix="Validate filename: use os.path.basename() or check for '..' in the path",
    ),
    CodeCase(
        id="code_hard_03",
        language="python",
        buggy_code="total = 0.0\nfor i in range(10):\n    total += 0.1\nprint(total == 1.0)",
        bug_description="Floating point comparison fails — 0.1 * 10 != 1.0 in IEEE 754.",
        bug_line=4,
        fix_type="floating_point",
        category="floating_point",
        difficulty="hard",
        explanation="0.1 is not exactly representable in binary. Sum is 0.99999... not 1.0. Use math.isclose().",
        correct_fix="Change `total == 1.0` to `math.isclose(total, 1.0)`",
    ),
    CodeCase(
        id="code_hard_04",
        language="python",
        buggy_code="class Counter:\n    count = 0\n    \n    def __init__(self):\n        Counter.count += 1\n    \n    def get_count(self):\n        return self.count",
        bug_description="Class variable vs instance variable confusion — count is shared across all instances.",
        bug_line=2,
        fix_type="class_vs_instance",
        category="oop_error",
        difficulty="hard",
        explanation="count is a class variable, modified via Counter.count, but read via self.count. All instances share it.",
        correct_fix="Use self.count in __init__ or make design intention explicit",
    ),
    CodeCase(
        id="code_hard_05",
        language="python",
        buggy_code="def process_items(items):\n    results = []\n    for item in items:\n        if item.startswith('#'):\n            continue\n        results.append(item.strip())\n        if len(results) > 100:\n            break\n    return results",
        bug_description="Off-by-one in limit check — `> 100` allows 101 items before breaking.",
        bug_line=7,
        fix_type="off_by_one",
        category="boundary_error",
        difficulty="hard",
        explanation="The check > 100 breaks after 101 items added. Should be >= 100 for a 100-item limit.",
        correct_fix="Change `len(results) > 100` to `len(results) >= 100`",
    ),
    CodeCase(
        id="code_hard_06",
        language="python",
        buggy_code="def deep_copy_dict(d):\n    new = {}\n    for key, value in d.items():\n        new[key] = value\n    return new",
        bug_description="Shallow copy — nested objects (lists, dicts) are still shared references.",
        bug_line=4,
        fix_type="shallow_vs_deep",
        category="reference_error",
        difficulty="hard",
        explanation="Simple assignment copies references, not values, for mutable objects. Use copy.deepcopy().",
        correct_fix="Use `import copy; return copy.deepcopy(d)` or recursively copy nested structures",
    ),
    CodeCase(
        id="code_hard_07",
        language="python",
        buggy_code="def thread_safe_increment(lock, counter):\n    value = counter['value']\n    value += 1\n    counter['value'] = value",
        bug_description="Race condition — read-modify-write without acquiring the lock.",
        bug_line=2,
        fix_type="race_condition",
        category="concurrency",
        difficulty="hard",
        explanation="The lock parameter exists but is never acquired. Should use `with lock:` block.",
        correct_fix="Wrap lines 2-4 in `with lock:` block",
    ),
    CodeCase(
        id="code_hard_08",
        language="python",
        buggy_code="def parse_int(s):\n    try:\n        return int(s)\n    except:\n        return 0",
        bug_description="Bare except catches everything including KeyboardInterrupt and SystemExit.",
        bug_line=4,
        fix_type="broad_except",
        category="error_handling",
        difficulty="hard",
        explanation="Bare `except:` catches SystemExit, KeyboardInterrupt, etc. Should be `except ValueError:`.",
        correct_fix="Change `except:` to `except (ValueError, TypeError):`",
    ),
    CodeCase(
        id="code_hard_09",
        language="python",
        buggy_code="def check_password(stored_hash, user_input):\n    return stored_hash == user_input",
        bug_description="Timing attack vulnerability — string comparison short-circuits, leaking info about the hash.",
        bug_line=2,
        fix_type="timing_attack",
        category="security",
        difficulty="hard",
        explanation="== short-circuits on first different byte. Use hmac.compare_digest() for constant-time comparison.",
        correct_fix="Use `hmac.compare_digest(stored_hash, user_input)` instead of `==`",
    ),
    CodeCase(
        id="code_hard_10",
        language="python",
        buggy_code="import json\n\ndef load_config(user_data):\n    config = json.loads(user_data)\n    return int(config.get('threshold', 0))",
        bug_description="Code injection vulnerability — arbitrary code execution from user input via dynamic evaluation.",
        bug_line=5,
        fix_type="code_injection",
        category="security",
        difficulty="hard",
        explanation="Dynamic evaluation executes arbitrary Python code. User-controlled input should never be dynamically evaluated.",
        correct_fix="Use ast.literal_eval for safe evaluation of literals, or avoid dynamic evaluation entirely",
    ),
]


# ═══════════════════════════════════════════════════════════════
# Access helpers
# ═══════════════════════════════════════════════════════════════

ALL_CODE_CASES = EASY_CASES + MEDIUM_CASES + HARD_CASES

CODE_CASES_BY_DIFFICULTY = {
    "easy": EASY_CASES,
    "medium": MEDIUM_CASES,
    "hard": HARD_CASES,
}

CODE_CASES_BY_ID = {case.id: case for case in ALL_CODE_CASES}


def get_code_cases(difficulty: str = None) -> list:
    """Get code cases filtered by difficulty, or all if None."""
    if difficulty is None:
        return ALL_CODE_CASES
    return CODE_CASES_BY_DIFFICULTY.get(difficulty, [])


def grade_code_answer(
    predicted_line: int,
    correct_line: int,
    predicted_fix_type: str,
    correct_fix_type: str,
) -> float:
    """Grade a code debugging answer.

    - Correct line AND fix type: 1.0
    - Correct line, wrong fix type: 0.6
    - Within 1 line of correct: 0.3
    - Wrong: 0.0
    """
    line_correct = predicted_line == correct_line
    type_correct = (
        predicted_fix_type.lower().strip() == correct_fix_type.lower().strip()
    )

    if line_correct and type_correct:
        return 1.0
    if line_correct:
        return 0.6
    if abs(predicted_line - correct_line) <= 1:
        return 0.3
    return 0.0
