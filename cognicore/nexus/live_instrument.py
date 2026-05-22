"""
NEXUS Full Runtime Instrumentor — hooks into ALL subsystems:
- NexusRunner (autonomous execution)
- NexusShield (immune system)
- EventRecorder + EventStore (replay)
- TaskBrancher (branching)
- Memory (episodic + persistent cognition)
- Multi-agent orchestration

Every event is REAL. Nothing synthetic.
"""
import uuid
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Callable, Optional


@dataclass
class LiveEvent:
    event_id: str = ""
    task_id: str = ""
    timestamp: float = 0.0
    phase: str = ""
    action: str = ""
    detail: str = ""
    status: str = "done"
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_ms: int = 0
    model: str = ""
    branch_id: str = "main"
    attempt: int = 0
    agent: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = uuid.uuid4().hex[:12]
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self):
        return {k: v for k, v in asdict(self).items()
                if v or k in ("tokens_in", "tokens_out", "attempt")}

    def to_json(self):
        return json.dumps(self.to_dict())


class FullInstrumentor:
    """
    Hooks into ALL NEXUS subsystems and streams real events.
    """

    def __init__(self, runner=None):
        if runner is None:
            from cognicore.nexus.autonomous import NexusRunner
            runner = NexusRunner()

        self.runner = runner
        self.runner.on_step(self._on_runner_step)

        # Event persistence
        try:
            from cognicore.replay.store import EventStore
            from cognicore.replay.recorder import EventRecorder
            self.store = EventStore()
            self.recorder = EventRecorder(store=self.store)
        except Exception:
            self.store = None
            self.recorder = None

        # Immune system
        self.shield = None
        try:
            from cognicore.immune.shield import NexusShield
            self.shield = NexusShield()
        except Exception as e:
            print(f"  [INIT] Immune system: {e}")

        # Brancher
        self.brancher = None
        try:
            from cognicore.replay.brancher import TaskBrancher
            self.brancher = TaskBrancher(store=self.store)
        except Exception as e:
            print(f"  [INIT] Brancher: {e}")

        # Memory middleware
        self.memory = None
        try:
            from cognicore.middleware.memory import Memory
            self.memory = Memory()
        except Exception as e:
            print(f"  [INIT] Memory: {e}")

        # Persistent cognition
        self.persistent_store = None
        try:
            from cognicore.research.persistent_store import PersistentCognitionStore
            self.persistent_store = PersistentCognitionStore()
        except Exception as e:
            print(f"  [INIT] PersistentStore: {e}")

        # Safety monitor
        self.safety = None
        try:
            from cognicore.middleware.safety_monitor import SafetyMonitor
            self.safety = SafetyMonitor()
        except Exception as e:
            print(f"  [INIT] SafetyMonitor: {e}")

        # Reflection engine
        self.reflection = None
        try:
            from cognicore.middleware.reflection import ReflectionEngine
            if self.memory:
                self.reflection = ReflectionEngine(memory=self.memory)
            else:
                from cognicore.middleware.memory import Memory
                self.reflection = ReflectionEngine(memory=Memory())
        except Exception as e:
            print(f"  [INIT] ReflectionEngine: {e}")

        # Callbacks
        self._callbacks = []
        self._lock = threading.Lock()
        self.task_id = ""
        self.events = []
        self._step = 0
        self._attempt = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_cost = 0.0
        self._start_time = 0.0

    def on_event(self, cb):
        with self._lock:
            self._callbacks.append(cb)

    def remove_callback(self, cb):
        with self._lock:
            self._callbacks = [c for c in self._callbacks if c != cb]

    def solve(self, prompt, repo="", repo_path="", auto_pr=False):
        """Run real execution with full instrumentation."""
        self.task_id = "task_" + uuid.uuid4().hex[:8]
        self.events = []
        self._step = 0
        self._attempt = 0
        self._total_tokens_in = 0
        self._total_tokens_out = 0
        self._total_cost = 0.0
        self._start_time = time.time()

        # Emit start
        self._emit(LiveEvent(
            task_id=self.task_id, phase="setup",
            action="task_start", detail="Task: " + prompt,
            status="running", agent="nexus_runner"))

        # Run immune check on the prompt
        if self.shield:
            try:
                decision = self.shield(prompt)
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="immune",
                    action="shield_scan",
                    detail=json.dumps({
                        "allowed": decision.allowed,
                        "blocked": decision.blocked,
                        "threat_score": decision.threat_score,
                        "threat_category": decision.threat_category,
                        "confidence": round(decision.confidence, 3),
                        "action": decision.action,
                        "reason": decision.reason,
                        "latency_ms": round(decision.latency_ms, 2),
                    }),
                    status="done" if decision.allowed else "blocked",
                    latency_ms=int(decision.latency_ms),
                    agent="immune_shield"))

                if decision.blocked:
                    # Only hard-block on high-confidence threats
                    if decision.threat_score >= 0.8:
                        self._emit(LiveEvent(
                            task_id=self.task_id, phase="immune",
                            action="HARD_BLOCKED",
                            detail="High-confidence threat: " + decision.reason,
                            status="failed", agent="immune_shield"))
                        return {"task_id": self.task_id, "solved": False,
                                "blocked": True, "reason": decision.reason}
                    else:
                        # Advisory mode: log warning but continue
                        self._emit(LiveEvent(
                            task_id=self.task_id, phase="immune",
                            action="ADVISORY_WARNING",
                            detail="Low-confidence block (continuing): score={:.2f} conf={:.2f} reason={}".format(
                                decision.threat_score, decision.confidence, decision.reason),
                            status="done", agent="immune_shield"))
            except Exception as e:
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="immune",
                    action="shield_error", detail=str(e),
                    status="failed", agent="immune_shield"))

        # Check persistent memory
        if self.persistent_store:
            try:
                cat = self._classify(prompt)
                insights = self.persistent_store.get_cross_session_insights(cat)
                tactics = insights.get("successful_tactics", {})
                failures = insights.get("failed_tactics", {})
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="memory",
                    action="persistent_recall",
                    detail=json.dumps({
                        "category": cat,
                        "successful_tactics": dict(list(tactics.items())[:5]),
                        "failed_tactics": dict(list(failures.items())[:5]),
                        "total_episodes": insights.get("total_episodes", 0),
                    }),
                    agent="persistent_cognition"))
            except Exception as e:
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="memory",
                    action="recall_error", detail=str(e),
                    status="failed", agent="persistent_cognition"))

        # Reflection pre-check
        if self.reflection:
            try:
                hints = self.reflection.reflect_before(prompt)
                if hints:
                    self._emit(LiveEvent(
                        task_id=self.task_id, phase="reflection",
                        action="pre_analysis",
                        detail=str(hints)[:300],
                        agent="reflection_engine"))
            except Exception:
                pass

        # Safety monitor check
        if self.safety:
            try:
                health = self.safety.stats()
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="safety",
                    action="health_check",
                    detail=json.dumps(health),
                    agent="safety_monitor"))
            except Exception:
                pass

        # Record to replay store
        if self.recorder:
            self.recorder.record_simple(
                self.task_id, "task_start",
                agent="nexus_runner", input_text=prompt)

        # REAL EXECUTION
        result = self.runner.solve(
            prompt, repo=repo, repo_path=repo_path, auto_pr=auto_pr)

        duration = round(time.time() - self._start_time, 2)

        # Post-execution memory store
        if self.memory:
            try:
                self.memory.store({
                    "category": self._classify(prompt),
                    "task": prompt[:100],
                    "correct": result.solved,
                    "attempts": result.attempts,
                    "duration": duration,
                })
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="memory",
                    action="episode_stored",
                    detail=json.dumps(self.memory.stats()),
                    agent="episodic_memory"))
            except Exception:
                pass

        # Post-execution reflection
        if self.reflection:
            try:
                analysis = self.reflection.reflect_after(
                    prompt, result.solved, result.attempts)
                if analysis:
                    self._emit(LiveEvent(
                        task_id=self.task_id, phase="reflection",
                        action="post_analysis",
                        detail=str(analysis)[:300],
                        agent="reflection_engine"))
            except Exception:
                pass

        # Post-execution safety update
        if self.safety:
            try:
                self.safety.check(result.solved)
                health = self.safety.stats()
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="safety",
                    action="health_updated",
                    detail=json.dumps(health),
                    agent="safety_monitor"))
            except Exception:
                pass

        # Create branch if failed (for replay comparison)
        if self.brancher and not result.solved and self.store:
            try:
                branch = self.brancher.branch(
                    self.task_id, from_step=1,
                    modifications={"strategy": "alternative"})
                self.brancher.record_branch_outcome(
                    branch, solved=False,
                    tokens=self._total_tokens_in + self._total_tokens_out,
                    cost=self._total_cost, duration=duration)
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="replay",
                    action="branch_created",
                    detail=json.dumps(branch.to_dict()),
                    branch_id=branch.branch_id,
                    agent="brancher"))
            except Exception as e:
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="replay",
                    action="branch_error", detail=str(e)[:200],
                    status="failed", agent="brancher"))

        # Immune stats
        if self.shield:
            try:
                stats = self.shield.get_stats()
                self._emit(LiveEvent(
                    task_id=self.task_id, phase="immune",
                    action="immune_stats",
                    detail=json.dumps(stats),
                    agent="immune_shield"))
            except Exception:
                pass

        # Final event
        final = LiveEvent(
            task_id=self.task_id, phase="done",
            action="SOLVED" if result.solved else "FAILED",
            detail=json.dumps({
                "solved": result.solved,
                "attempts": result.attempts,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "duration": duration,
                "total_tokens_in": self._total_tokens_in,
                "total_tokens_out": self._total_tokens_out,
                "total_cost": round(self._total_cost, 6),
                "files_changed": result.files_changed,
                "error": result.error,
            }),
            status="done" if result.solved else "failed",
            tokens_in=self._total_tokens_in,
            tokens_out=self._total_tokens_out,
            cost=round(self._total_cost, 6))
        self._emit(final)

        # Persist to replay
        if self.recorder:
            self.recorder.record_simple(
                self.task_id,
                "task_solved" if result.solved else "task_failed",
                agent="nexus_runner", output_text=final.detail,
                step=self._step,
                tokens_in=self._total_tokens_in,
                tokens_out=self._total_tokens_out,
                cost=self._total_cost)

        return {
            "task_id": self.task_id,
            "solved": result.solved,
            "attempts": result.attempts,
            "tests_passed": result.tests_passed,
            "tests_failed": result.tests_failed,
            "duration": duration,
            "tokens_in": self._total_tokens_in,
            "tokens_out": self._total_tokens_out,
            "cost": round(self._total_cost, 6),
            "files_changed": result.files_changed,
            "error": result.error,
            "events": len(self.events),
        }

    def _on_runner_step(self, step):
        phase = step.get("phase", "")
        action = step.get("action", "")
        detail = step.get("detail", "")
        status = step.get("status", "done")
        raw_tokens = step.get("tokens", 0)
        self._step += 1

        if phase == "patch" and "Attempt" in action:
            try:
                self._attempt = int(action.split("/")[0].split()[-1])
            except (ValueError, IndexError):
                pass

        tokens_in = tokens_out = latency_ms = 0
        model = ""
        cost = 0.0

        if phase == "llm" and hasattr(self.runner, '_llm') and self.runner._llm:
            llm = self.runner._llm
            if hasattr(llm, '_last_call'):
                lc = llm._last_call
                tokens_in = lc.get("tokens_in", 0)
                tokens_out = lc.get("tokens_out", 0)
                latency_ms = lc.get("latency_ms", 0)
                model = lc.get("model", "")
                cost = (tokens_in * 0.00001 + tokens_out * 0.00004)
                self._total_tokens_in += tokens_in
                self._total_tokens_out += tokens_out
                self._total_cost += cost
        elif raw_tokens > 0:
            tokens_in = raw_tokens
            self._total_tokens_in += tokens_in

        # Map phase to agent name
        agent_map = {
            "setup": "workspace", "search": "localizer",
            "read": "reader", "memory": "memory",
            "plan": "planner", "llm": "coder",
            "patch": "coder", "apply": "patcher",
            "test": "tester", "pr": "github",
        }

        evt = LiveEvent(
            task_id=self.task_id, phase=phase, action=action,
            detail=detail, status=status,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cost=round(cost, 6), latency_ms=latency_ms,
            model=model, attempt=self._attempt,
            agent=agent_map.get(phase, phase))

        if phase in ("test", "search", "setup") and detail:
            pass  # detail already contains subprocess output

        self._emit(evt)

        if self.recorder:
            self.recorder.record_simple(
                self.task_id, phase, agent=agent_map.get(phase, phase),
                action=action, output_text=detail, step=self._step,
                tokens_in=tokens_in, tokens_out=tokens_out,
                cost=cost, latency_ms=latency_ms)

    def _emit(self, event):
        self.events.append(event)
        with self._lock:
            for cb in self._callbacks:
                try:
                    cb(event)
                except Exception:
                    pass

    def _classify(self, prompt):
        p = prompt.lower()
        for kw, cat in [("none", "none_handling"), ("null", "none_handling"),
                        ("crash", "crash"), ("encoding", "encoding"),
                        ("type", "type_error"), ("import", "import_error")]:
            if kw in p:
                return cat
        return "general"

    def get_immune_stats(self):
        if self.shield:
            try:
                return self.shield.get_stats()
            except Exception:
                pass
        return {"total_calls": 0, "total_blocked": 0, "block_rate": 0}

    def get_immune_dashboard(self):
        if self.shield:
            try:
                return self.shield.dashboard_data()
            except Exception:
                pass
        return {}

    def get_memory_stats(self):
        if self.memory:
            try:
                return self.memory.stats()
            except Exception:
                pass
        return {"total_entries": 0}

    def get_replay_events(self, task_id=None):
        tid = task_id or self.task_id
        if self.store and tid:
            events = self.store.load_events(tid)
            return [e.to_dict() for e in events]
        return []

    def get_branches(self, task_id=None):
        tid = task_id or self.task_id
        if self.brancher and tid:
            try:
                branches = self.brancher.list_branches(tid)
                return [b.to_dict() for b in branches]
            except Exception:
                pass
        return []

    def get_subsystem_status(self):
        return {
            "runner": True,
            "llm": bool(self.runner._llm) if self.runner else False,
            "immune": self.shield is not None,
            "replay": self.store is not None,
            "brancher": self.brancher is not None,
            "memory": self.memory is not None,
            "persistent_cognition": self.persistent_store is not None,
            "safety": self.safety is not None,
            "reflection": self.reflection is not None,
        }

    def scan_threat(self, text):
        """Run immune scan on arbitrary text."""
        if not self.shield:
            return {"error": "Immune system not loaded"}
        try:
            d = self.shield(text)
            return {
                "allowed": d.allowed, "blocked": d.blocked,
                "threat_score": d.threat_score,
                "threat_category": d.threat_category,
                "confidence": round(d.confidence, 3),
                "action": d.action, "reason": d.reason,
                "latency_ms": round(d.latency_ms, 2),
            }
        except Exception as e:
            return {"error": str(e)}

    def get_stats(self):
        return {
            "task_id": self.task_id,
            "events": len(self.events),
            "tokens_in": self._total_tokens_in,
            "tokens_out": self._total_tokens_out,
            "cost": round(self._total_cost, 6),
            "duration": round(time.time() - self._start_time, 2) if self._start_time else 0,
            "attempts": self._attempt,
        }

    def stop(self):
        if self.recorder:
            self.recorder.stop()
