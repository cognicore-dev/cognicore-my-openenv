from typing import List, Dict, Any
import random

def generate_python_tasks(num_episodes: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    tasks = []
    
    key_desc_templates = [
        "Fix the KeyError by safely accessing the dictionary.",
        "Resolve the missing key exception in the code.",
        "The dictionary lookup fails when the key is absent. Please fix it.",
        "Handle the invalid dictionary access gracefully."
    ]
    
    idx_desc_templates = [
        "Fix the IndexError by safely accessing the list.",
        "Prevent the out-of-bounds array access.",
        "The list index is out of range. Add bounds checking.",
        "Safely handle the index out of bounds error."
    ]
    
    for i in range(num_episodes):
        concept_type = "KeyError" if i % 2 == 0 else "IndexError"
        
        var_name = random.choice(["data_dict", "config", "payload", "user_map", "settings"])
        key_name = random.choice(["'id'", "'status'", "'username'", "'role'", "'token'"])
        
        list_name = random.choice(["items", "results", "records", "nodes", "elements"])
        idx = random.randint(10, 50)
        
        if concept_type == "KeyError":
            buggy_code = f"""
def process_data({var_name}):
    # Bug: accessing missing key directly
    return {var_name}[{key_name}]
"""
            desc = random.choice(key_desc_templates)
            hidden_assert = f"""
try:
    assert process_data({{"other_key": "val"}}) is None or process_data({{"other_key": "val"}}) == ""
except Exception as e:
    import sys
    print("Assertion Failed:", e, file=sys.stderr)
    sys.exit(1)
"""
        else:
            buggy_code = f"""
def get_item({list_name}):
    # Bug: accessing out of bounds index directly
    return {list_name}[{idx}]
"""
            desc = random.choice(idx_desc_templates)
            hidden_assert = f"""
try:
    assert get_item([1, 2, 3]) is None or get_item([1, 2, 3]) == ""
except Exception as e:
    import sys
    print("Assertion Failed:", e, file=sys.stderr)
    sys.exit(1)
"""

        tasks.append({
            "domain": "software_debugging",
            "episode_id": i,
            "concept_type": concept_type,
            "error_type": concept_type,
            "description": desc,
            "buggy_code": buggy_code.strip(),
            "hidden_assert": hidden_assert.strip()
        })
    return tasks

def generate_docker_tasks(num_episodes: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    tasks = []
    
    port_desc_templates = [
        "Fix the port conflict where both services map to the same host port.",
        "Resolve the host port collision between the two services.",
        "The services are trying to bind to the identical host port. Fix the mapping."
    ]
    
    env_desc_templates = [
        "Fix the environment variable. TIMEOUT must be an integer.",
        "The linter expects the TIMEOUT env var to be a numeric integer type.",
        "Correct the TIMEOUT value to be an integer instead of a string."
    ]
    
    for i in range(num_episodes):
        concept_type = "no_port_conflict" if i % 2 == 0 else "valid_env_types"
        
        svc1 = random.choice(["web", "api", "frontend", "backend", "gateway"])
        svc2 = random.choice(["db", "cache", "worker", "queue", "redis"])
        port = random.choice(["8080", "3000", "5000", "9000", "8000"])
        
        if concept_type == "no_port_conflict":
            buggy_yaml = f"""
version: '3.8'
services:
  {svc1}:
    image: nginx:latest
    ports:
      - "{port}:80"
  {svc2}:
    image: redis:alpine
    ports:
      - "{port}:6379"  # Bug: Port conflict on host
"""
            desc = random.choice(port_desc_templates)
            req_state = {"required_services": [svc1, svc2], "must_expose_ports": True}
        else:
            buggy_yaml = f"""
version: '3.8'
services:
  {svc1}:
    image: myapp:latest
    environment:
      TIMEOUT: "30s"  # Bug: Linter expects TIMEOUT to be an integer
"""
            desc = random.choice(env_desc_templates)
            req_state = {"required_services": [svc1], "must_expose_ports": False}

        tasks.append({
            "domain": "deployment_infrastructure",
            "episode_id": i,
            "concept_type": concept_type,
            "error_type": concept_type,
            "description": desc,
            "buggy_yaml": buggy_yaml.strip(),
            "validation_rule": concept_type,
            "required_state": req_state
        })
    return tasks
