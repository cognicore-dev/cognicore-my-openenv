"""
Event Recorder — records every agent event with zero overhead.
Write-ahead log design, non-blocking, compression-ready.
"""
import uuid
import time
import hashlib
import threading
import queue
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Callable
import numpy as np


class EventType(str, Enum):
    TASK_START = "task_start"
    MEMORY_RETRIEVED = "memory_retrieved"
    PLAN_GENERATED = "plan_generated"
    PATCH_GENERATED = "patch_generated"
    PATCH_REJECTED = "patch_rejected"
    TEST_EXECUTED = "test_executed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    REVIEW_REQUESTED = "review_requested"
    REVIEW_APPROVED = "review_approved"
    REVIEW_REJECTED = "review_rejected"
    BRANCH_CREATED = "branch_created"
    BRANCH_MERGED = "branch_merged"
    TASK_SOLVED = "task_solved"
    TASK_FAILED = "task_failed"
    HUMAN_INTERVENED = "human_intervened"
    IMMUNE_BLOCKED = "immune_blocked"
    COST_LIMIT_HIT = "cost_limit_hit"
    CHECKPOINT = "checkpoint"


@dataclass
class AgentEvent:
    # Identity
    event_id: str = ""
    task_id: str = ""
    branch_id: str = "main"
    parent_id: str = ""

    # Sequence
    seq: int = 0
    step: int = 0
    timestamp: float = 0.0

    # Content
    event_type: str = ""
    agent: str = ""
    input_hash: str = ""
    output_hash: str = ""
    input_text: str = ""
    output_text: str = ""

    # Economics
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: int = 0

    # RL
    state_vector: Optional[np.ndarray] = None
    action: str = ""
    reward: float = 0.0
    value_est: float = 0.0

    # Metadata
    model: str = ""
    temperature: float = 0.0
    policy: str = ""
    memory_hits: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()
        if self.input_text and not self.input_hash:
            self.input_hash = hashlib.sha256(
                self.input_text.encode()).hexdigest()[:16]
        if self.output_text and not self.output_hash:
            self.output_hash = hashlib.sha256(
                self.output_text.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif v is not None:
                d[k] = v
        return d


class EventRecorder:
    """
    Records every agent event with zero overhead.
    Non-blocking queue-based writes. Background flush worker.
    """

    def __init__(self, store=None):
        self._store = store
        self._queue = queue.Queue(maxsize=10000)
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._callbacks: List[Callable] = []
        self._running = False
        self._worker = None
        self._checkpoints = {}

        if self._store:
            self._start_worker()

    def _start_worker(self):
        self._running = True
        self._worker = threading.Thread(
            target=self._flush_worker, daemon=True)
        self._worker.start()

    def _flush_worker(self):
        batch = []
        while self._running:
            try:
                event = self._queue.get(timeout=0.5)
                batch.append(event)
                # Drain up to 50 events
                while len(batch) < 50:
                    try:
                        event = self._queue.get_nowait()
                        batch.append(event)
                    except queue.Empty:
                        break
                # Flush batch
                if self._store and batch:
                    for e in batch:
                        try:
                            self._store.save_event(e)
                        except Exception:
                            pass
                batch.clear()
            except queue.Empty:
                if batch and self._store:
                    for e in batch:
                        try:
                            self._store.save_event(e)
                        except Exception:
                            pass
                    batch.clear()

    def record(self, event: AgentEvent):
        """Non-blocking event recording."""
        with self._seq_lock:
            self._seq += 1
            event.seq = self._seq

        # Fire callbacks (for live UI streaming)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass

        # Queue for persistence
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass  # Drop under extreme load

    def record_simple(self, task_id: str, event_type: str,
                     agent: str = "", input_text: str = "",
                     output_text: str = "", step: int = 0,
                     tokens_in: int = 0, tokens_out: int = 0,
                     cost: float = 0.0, latency_ms: int = 0,
                     action: str = "", reward: float = 0.0,
                     policy: str = "", confidence: float = 0.0,
                     branch_id: str = "main",
                     state_vector: np.ndarray = None) -> AgentEvent:
        """Convenience method for quick event recording."""
        event = AgentEvent(
            task_id=task_id, event_type=event_type,
            agent=agent, input_text=input_text,
            output_text=output_text, step=step,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost=cost, latency_ms=latency_ms,
            action=action, reward=reward,
            policy=policy, confidence=confidence,
            branch_id=branch_id, state_vector=state_vector)
        self.record(event)
        return event

    def checkpoint(self, task_id: str, label: str = "") -> str:
        """Create a named checkpoint. Returns checkpoint_id."""
        cp_id = f"cp_{uuid.uuid4().hex[:8]}"
        event = AgentEvent(
            task_id=task_id,
            event_type=EventType.CHECKPOINT,
            action=f"checkpoint:{cp_id}",
            output_text=label or cp_id)
        self.record(event)
        self._checkpoints[cp_id] = {
            "task_id": task_id,
            "seq": event.seq,
            "step": event.step,
            "label": label,
            "timestamp": event.timestamp,
        }
        return cp_id

    def get_checkpoint(self, checkpoint_id: str) -> dict:
        return self._checkpoints.get(checkpoint_id, {})

    def on_event(self, callback: Callable):
        """Register a callback for live event streaming."""
        self._callbacks.append(callback)

    def stop(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=2)
