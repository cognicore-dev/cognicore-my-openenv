import os
import tempfile
import subprocess
import re
from typing import Dict, Any, Tuple
from harness.env import BaseEnvironment

class PythonSandboxEnv(BaseEnvironment):
    """Executes a Python script in a temporary directory to verify correctness."""
    
    def __init__(self, timeout_sec: int = 5):
        self.timeout_sec = timeout_sec
        self.current_task = None
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def reset(self, task: Dict[str, Any]) -> str:
        self.current_task = task
        # The prompt will show the code and the error to fix
        buggy_code = task.get("buggy_code", "")
        error_type = task.get("error_type", "Error")
        description = task.get("description", "Fix the bug in the code.")
        
        prompt = (
            f"Task: {description}\n"
            f"Error Mode: {error_type}\n\n"
            f"Here is the buggy code:\n```python\n{buggy_code}\n```\n\n"
            f"Please output the fully corrected script inside a ```python ... ``` block."
        )
        return prompt

    def step(self, action: str) -> Tuple[str, bool, bool, Dict[str, Any]]:
        # Extract the python code block
        match = re.search(r"```python\s*(.*?)\s*```", action, re.DOTALL)
        if not match:
            # Maybe they just output raw code without markdown
            code_to_run = action.strip()
            # If it starts with a sentence, it's probably malformed. We'll try running it anyway.
            if "def " not in code_to_run and "import " not in code_to_run:
                return "Failed to parse Python code block. Please wrap in ```python ... ```.", False, False, {}
        else:
            code_to_run = match.group(1)

        hidden_assert = self.current_task.get("hidden_assert", "")
        code_to_run += "\n\n" + hidden_assert

        script_path = os.path.join(self.temp_dir.name, "script.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code_to_run)

        # Execute
        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_sec
            )
            if result.returncode == 0:
                return "Execution successful.", True, False, {"output": result.stdout}
            else:
                return f"Execution failed:\n{result.stderr}", False, False, {"output": result.stderr}
        except subprocess.TimeoutExpired:
            return "Execution timed out.", False, False, {}
        except Exception as e:
            return f"System error: {str(e)}", False, True, {}

    def cleanup(self):
        self.temp_dir.cleanup()
