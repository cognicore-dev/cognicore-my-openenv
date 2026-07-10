import json
import os
import time

class MetricsTracker:
    def __init__(self, output_file: str):
        self.output_file = output_file
        # Ensure dir exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
    def log_episode(self, run_id: int, arm_name: str, domain: str, episode_id: int, 
                    concept: str, success: bool, retries: int, first_action_acc: bool,
                    rfr: float, tokens_prompt: int, tokens_completion: int, 
                    cost_usd: float, tts_sec: float):
        
        record = {
            "run_id": run_id,
            "arm": arm_name,
            "domain": domain,
            "episode": episode_id,
            "task_concept": concept,
            "solve_status": success,
            "retry_count": retries,
            "first_action_accuracy": first_action_acc,
            "repeated_failure_rate": rfr,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "cost_usd": cost_usd,
            "time_to_solution_sec": tts_sec,
            "timestamp": time.time()
        }
        
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
