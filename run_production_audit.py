import os
import sys
import subprocess
import json
from pydantic import BaseModel

print("=== STARTING PRODUCTION AUDIT ===")

print("\n--- 1. Testing Imports ---")
try:
    from cognicore.integrations.langchain import cognicore_tools
    from cognicore.integrations.crewai import cognicore_crewai_tools
    print("SUCCESS: Imports work.")
except Exception as e:
    print(f"FAILED: Imports failed with {e}")

print("\n--- 2. Checking CrewAI args_schema ---")
crewai_tools = cognicore_crewai_tools()
for t in crewai_tools:
    print(f"Tool: {t.name}")
    if t.args_schema and issubclass(t.args_schema, BaseModel):
        try:
            schema = t.args_schema.model_json_schema()
        except AttributeError:
            schema = t.args_schema.schema() # Pydantic v1 fallback
        print(f"  Valid BaseModel schema attached: {json.dumps(schema)}")
    else:
        print("  FAILED: Invalid args_schema")

print("\n--- 3. Checking LangChain args_schema ---")
lc_tools = cognicore_tools()
for t in lc_tools:
    print(f"Tool: {t.name}")
    if t.args_schema and issubclass(t.args_schema, BaseModel):
        try:
            schema = t.args_schema.model_json_schema()
        except AttributeError:
            schema = t.args_schema.schema() # Pydantic v1 fallback
        print(f"  Valid BaseModel schema attached: {json.dumps(schema)}")
    else:
        print("  WARNING/FAILED: args_schema might not be Pydantic BaseModel")

print("\n--- 4. Executing examples/memory_behavior_demo.py ---")
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
result1 = subprocess.run(["python", "examples/memory_behavior_demo.py"], capture_output=True, text=True, env=env)
print("Return code:", result1.returncode)
print("STDOUT:\n", result1.stdout)
if result1.stderr:
    print("STDERR:\n", result1.stderr)

print("\n--- 5. Executing examples/crewai_advanced_demo.py ---")
result2 = subprocess.run(["python", "examples/crewai_advanced_demo.py"], capture_output=True, text=True, env=env)
print("Return code:", result2.returncode)
print("STDOUT:\n", result2.stdout)
if result2.stderr:
    print("STDERR:\n", result2.stderr)

print("=== AUDIT COMPLETE ===")
