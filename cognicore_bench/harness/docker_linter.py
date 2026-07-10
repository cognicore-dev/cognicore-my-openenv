import os
import tempfile
import re
import yaml
from typing import Dict, Any, Tuple
from harness.env import BaseEnvironment

class DockerLinterEnv(BaseEnvironment):
    """Simulates deployment by validating docker-compose.yml syntax and semantics."""
    
    def __init__(self):
        self.current_task = None
        
    def reset(self, task: Dict[str, Any]) -> str:
        self.current_task = task
        buggy_yaml = task.get("buggy_yaml", "")
        error_type = task.get("error_type", "Validation Error")
        description = task.get("description", "Fix the docker-compose deployment.")
        
        prompt = (
            f"Task: {description}\n"
            f"Error Mode: {error_type}\n\n"
            f"Here is the buggy docker-compose.yml:\n```yaml\n{buggy_yaml}\n```\n\n"
            f"Please output the fully corrected yaml inside a ```yaml ... ``` block."
        )
        return prompt

    def step(self, action: str) -> Tuple[str, bool, bool, Dict[str, Any]]:
        # Extract the yaml code block
        match = re.search(r"```yaml\s*(.*?)\s*```", action, re.DOTALL)
        if not match:
            # Fallback
            code_to_run = action.strip()
            if not code_to_run.startswith("version:") and not code_to_run.startswith("services:"):
                return "Failed to parse YAML block. Please wrap in ```yaml ... ```.", False, False, {}
        else:
            code_to_run = match.group(1)

        # Basic parse validation
        try:
            parsed = yaml.safe_load(code_to_run)
        except yaml.YAMLError as e:
            return f"YAML Parsing Error:\n{str(e)}", False, False, {}

        if not isinstance(parsed, dict) or "services" not in parsed:
            return "Validation Error: Missing 'services' top-level key.", False, False, {}

        services = parsed.get("services") or {}
        req_state = self.current_task.get("required_state", {})
        req_services = req_state.get("required_services", [])
        must_expose_ports = req_state.get("must_expose_ports", False)
        
        # Verify required state hasn't been maliciously deleted
        for req_svc in req_services:
            if req_svc not in services:
                return f"Validation Error: Missing required service '{req_svc}'.", False, False, {}
                
        if must_expose_ports:
            has_ports = any(svc and svc.get("ports") for svc in services.values())
            if not has_ports:
                return "Validation Error: Required port mapping was completely deleted.", False, False, {}

        target_rule = self.current_task.get("validation_rule")
        
        if target_rule == "no_port_conflict":
            ports_used = set()
            for s_name, svc in services.items():
                if svc is None: continue
                for p in svc.get("ports", []):
                    host_port = str(p).split(":")[0]
                    if host_port in ports_used:
                        return f"Validation Error: Port conflict on host port {host_port}.", False, False, {}
                    ports_used.add(host_port)
                    
        elif target_rule == "valid_env_types":
            for s_name, svc in services.items():
                if svc is None: continue
                env = svc.get("environment", {})
                if isinstance(env, dict):
                    for k, v in env.items():
                        if k == "TIMEOUT" and not str(v).isdigit():
                            return "Validation Error: Environment variable TIMEOUT must be an integer.", False, False, {}

        return "Deployment successful. Services are healthy.", True, False, {}

    def cleanup(self):
        pass
