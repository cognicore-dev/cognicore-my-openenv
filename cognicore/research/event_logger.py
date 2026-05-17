"""
Deterministic Event Logger — records every decision point for replay.
Produces a JSON event stream that can reproduce any experiment bit-for-bit.
"""
import json, time, hashlib, os
from typing import Dict, List, Optional
from pathlib import Path


class EventType:
    EXPERIMENT_START = "experiment_start"
    TASK_START = "task_start"
    PATCH_GENERATED = "patch_generated"
    PATCH_REJECTED = "patch_rejected"
    TEST_EXECUTED = "test_executed"
    MEMORY_RETRIEVED = "memory_retrieved"
    REFLECTION_GENERATED = "reflection_generated"
    STRATEGY_MUTATED = "strategy_mutated"
    PROMPT_MUTATED = "prompt_mutated"
    TASK_COMPLETE = "task_complete"
    EXPERIMENT_COMPLETE = "experiment_complete"


class EventLogger:
    """Structured event logger for deterministic replay."""

    def __init__(self, experiment_id: str, output_dir: str = "experiments"):
        self.experiment_id = experiment_id
        self.events: List[Dict] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def log(self, event_type: str, bug_id: str = "", attempt: int = 0,
            data: Dict = None):
        self._seq += 1
        event = {
            "seq": self._seq,
            "type": event_type,
            "timestamp": time.time(),
            "experiment_id": self.experiment_id,
            "bug_id": bug_id,
            "attempt": attempt,
            "data": data or {},
        }
        self.events.append(event)
        return event

    def log_patch(self, bug_id: str, attempt: int, patch: str, tactic: str,
                  passed: bool, error: str = "", similarity: float = 0.0,
                  rejected: bool = False, exec_time_ms: int = 0):
        return self.log(
            EventType.PATCH_GENERATED if not rejected else EventType.PATCH_REJECTED,
            bug_id, attempt, {
                "patch_hash": hashlib.md5(patch.encode()).hexdigest()[:12],
                "patch_preview": patch[:200],
                "tactic": tactic,
                "passed": passed,
                "error": error[:150],
                "similarity": round(similarity, 3),
                "rejected": rejected,
                "exec_time_ms": exec_time_ms,
            })

    def log_memory(self, bug_id: str, attempt: int, hits: int,
                   categories: List[str] = None):
        return self.log(EventType.MEMORY_RETRIEVED, bug_id, attempt, {
            "hits": hits, "categories": categories or [],
        })

    def log_reflection(self, bug_id: str, attempt: int, hint: str):
        return self.log(EventType.REFLECTION_GENERATED, bug_id, attempt, {
            "reflection": hint[:200],
        })

    def log_mutation(self, bug_id: str, attempt: int, disabled: str = "",
                     preferred: str = "", reason: str = ""):
        return self.log(EventType.STRATEGY_MUTATED, bug_id, attempt, {
            "disabled": disabled, "preferred": preferred, "reason": reason[:150],
        })

    def save(self) -> str:
        path = self.output_dir / f"events_{self.experiment_id}.jsonl"
        with open(path, "w") as f:
            for event in self.events:
                f.write(json.dumps(event, default=str) + "\n")
        # Also save a summary
        summary = {
            "experiment_id": self.experiment_id,
            "total_events": len(self.events),
            "event_types": {},
            "checksum": hashlib.sha256(
                json.dumps(self.events, default=str).encode()
            ).hexdigest()[:16],
        }
        for e in self.events:
            t = e["type"]
            summary["event_types"][t] = summary["event_types"].get(t, 0) + 1
        with open(self.output_dir / f"summary_{self.experiment_id}.json", "w") as f:
            json.dump(summary, f, indent=2)
        return str(path)

    def get_checksum(self) -> str:
        return hashlib.sha256(
            json.dumps(self.events, default=str).encode()
        ).hexdigest()[:16]
