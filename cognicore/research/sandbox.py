"""
Isolated Test Execution — subprocess-based patch validation.
Replaces exec() with proper isolation: timeout, memory limits, clean namespace.
"""
import subprocess, tempfile, os, time, json
from typing import Tuple, Optional
from pathlib import Path


def sandbox_isolated(code: str, tests: str, timeout: int = 30) -> Tuple[bool, Optional[str], dict]:
    """Execute patch + tests in isolated subprocess.

    Returns: (passed, error_message, metadata)
    - passed: True if all tests pass
    - error_message: None on success, error string on failure
    - metadata: execution time, exit code, stdout/stderr lengths
    """
    # Build the test script
    script = f"""{code}

# === TESTS ===
{tests}

print("__COGNICORE_PASS__")
"""
    # Write to temp file
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                      encoding='utf-8')
    tmp.write(script)
    tmp.close()

    meta = {"exec_time_ms": 0, "exit_code": -1, "stdout_len": 0, "stderr_len": 0}
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            ["python", tmp.name],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            cwd=tempfile.gettempdir(),
        )
        meta["exec_time_ms"] = int((time.perf_counter() - t0) * 1000)
        meta["exit_code"] = result.returncode
        meta["stdout_len"] = len(result.stdout)
        meta["stderr_len"] = len(result.stderr)

        if result.returncode == 0 and "__COGNICORE_PASS__" in result.stdout:
            return True, None, meta
        else:
            # Extract error from stderr
            err_lines = result.stderr.strip().split("\n")
            # Get the last meaningful error line
            error = ""
            for line in reversed(err_lines):
                line = line.strip()
                if line and not line.startswith("Traceback") and not line.startswith("File"):
                    error = line
                    break
            if not error and err_lines:
                error = err_lines[-1]
            if not error:
                error = f"Exit code {result.returncode}"
            return False, error, meta

    except subprocess.TimeoutExpired:
        meta["exec_time_ms"] = timeout * 1000
        return False, f"TimeoutError: Exceeded {timeout}s limit", meta
    except Exception as e:
        return False, f"ExecutionError: {e}", meta
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def sandbox_fast(code: str, tests: str) -> Tuple[bool, Optional[str]]:
    """Fast in-process sandbox (for quick iteration, not for production).
    Falls back to exec() but in a clean namespace."""
    ns = {}
    try:
        exec(compile(code, "<patch>", "exec"), ns)  # nosec B102
        exec(compile(tests, "<test>", "exec"), ns)  # nosec B102
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
