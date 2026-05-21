"""Tests for the NEXUS Replay & Time Travel system."""
import sys, os, tempfile, time, pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cognicore.replay.recorder import EventRecorder, AgentEvent, EventType
from cognicore.replay.store import EventStore
from cognicore.replay.replayer import TaskReplayer, ReplaySession
from cognicore.replay.brancher import TaskBrancher, Branch
from cognicore.replay.comparator import BranchComparator, Comparison
from cognicore.replay.rl_navigator import RLNavigator, NavAction, BranchDecision
from cognicore.replay.visualizer import TimelineVisualizer
from cognicore.replay.exporter import TrajectoryExporter


def make_store():
    return EventStore(db_path=tempfile.mktemp(suffix=".db"))


def make_events(task_id="task_001", n=5):
    """Create a sequence of test events."""
    events = []
    types = ["task_start", "plan_generated", "patch_generated",
             "test_executed", "task_solved"]
    for i in range(n):
        e = AgentEvent(
            task_id=task_id, branch_id="main",
            step=i, event_type=types[i % len(types)],
            agent="test_agent", action=f"action_{i}",
            input_text=f"input_{i}", output_text=f"output_{i}",
            tokens_in=100, tokens_out=50, cost=0.001,
            latency_ms=50, confidence=0.9)
        events.append(e)
    return events


# ── Event Store ────────────────────────────────────────────────────
class TestEventStore:
    def test_save_and_load(self):
        store = make_store()
        events = make_events()
        for e in events:
            store.save_event(e)
        loaded = store.load_events("task_001")
        assert len(loaded) == 5

    def test_load_by_step_range(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        loaded = store.load_events("task_001", from_step=1, to_step=3)
        assert all(1 <= e.step <= 3 for e in loaded)

    def test_task_ids(self):
        store = make_store()
        for e in make_events("t1"):
            store.save_event(e)
        for e in make_events("t2"):
            store.save_event(e)
        ids = store.get_task_ids()
        assert "t1" in ids and "t2" in ids

    def test_stats(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        stats = store.get_stats()
        assert stats["total_events"] == 5
        assert stats["total_tasks"] == 1

    def test_save_branch(self):
        store = make_store()
        store.save_branch({
            "branch_id": "br_001", "parent_task_id": "task_001",
            "branch_point": 2, "solved": True,
            "tokens_used": 500, "cost": 0.01, "duration": 1.5})
        branches = store.load_branches("task_001")
        assert len(branches) == 1
        assert branches[0]["branch_id"] == "br_001"

    def test_delete_task(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        store.delete_task("task_001")
        assert len(store.load_events("task_001")) == 0


# ── Event Recorder ─────────────────────────────────────────────────
class TestRecorder:
    def test_record_with_store(self):
        store = make_store()
        rec = EventRecorder(store=store)
        e = AgentEvent(task_id="t1", event_type="task_start", step=0)
        rec.record(e)
        time.sleep(1)  # let worker flush
        rec.stop()
        loaded = store.load_events("t1")
        assert len(loaded) >= 1

    def test_record_simple(self):
        rec = EventRecorder()
        e = rec.record_simple("t1", "task_start", agent="test")
        assert e.event_id
        assert e.seq > 0

    def test_checkpoint(self):
        rec = EventRecorder()
        cp_id = rec.checkpoint("t1", "before_patch")
        assert cp_id.startswith("cp_")
        cp = rec.get_checkpoint(cp_id)
        assert cp["task_id"] == "t1"

    def test_callback(self):
        rec = EventRecorder()
        received = []
        rec.on_event(lambda e: received.append(e))
        rec.record_simple("t1", "test_passed")
        assert len(received) == 1


# ── Replayer ───────────────────────────────────────────────────────
class TestReplayer:
    def test_replay_session(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        replayer = TaskReplayer(store)
        session = replayer.replay("task_001")
        assert session.total_steps == 5
        assert not session.is_done

    def test_step_forward_back(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        replayer = TaskReplayer(store)
        session = replayer.replay("task_001")
        e1 = session.step_forward()
        assert e1.step == 0
        e2 = session.step_forward()
        assert e2.step == 1
        e_back = session.step_back()
        assert e_back.step == 1

    def test_replay_to_step(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        replayer = TaskReplayer(store)
        state = replayer.replay_to_step("task_001", 2)
        assert state["step"] == 2
        assert state["task_id"] == "task_001"

    def test_state_reconstruction(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        replayer = TaskReplayer(store)
        session = replayer.replay("task_001")
        state = session.get_state_at(4)
        assert state["total_tokens_in"] == 500  # 5 events * 100

    def test_task_summary(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        replayer = TaskReplayer(store)
        summary = replayer.get_task_summary("task_001")
        assert summary["total_events"] == 5
        assert summary["solved"]


# ── Brancher ───────────────────────────────────────────────────────
class TestBrancher:
    def test_create_branch(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        brancher = TaskBrancher(store)
        branch = brancher.branch("task_001", from_step=2)
        assert branch.branch_id.startswith("br_")
        assert branch.branch_point == 2

    def test_list_branches(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        brancher = TaskBrancher(store)
        brancher.branch("task_001", from_step=1)
        brancher.branch("task_001", from_step=3)
        branches = brancher.list_branches("task_001")
        assert len(branches) == 2

    def test_record_outcome(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        brancher = TaskBrancher(store)
        branch = brancher.branch("task_001", from_step=2)
        brancher.record_branch_outcome(
            branch, solved=True, tokens=1000, cost=0.05)
        assert branch.solved


# ── Comparator ─────────────────────────────────────────────────────
class TestComparator:
    def test_compare_branches(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        brancher = TaskBrancher(store)
        b1 = brancher.branch("task_001", from_step=1)
        b2 = brancher.branch("task_001", from_step=2)
        brancher.record_branch_outcome(b1, True, 500, 0.01, 1.0)
        brancher.record_branch_outcome(b2, False, 800, 0.02, 2.0)

        comp = BranchComparator(store)
        result = comp.compare("task_001")
        assert result.winner == b1.branch_id
        assert len(result.branches) == 2

    def test_rank_branches(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        brancher = TaskBrancher(store)
        b1 = brancher.branch("task_001", from_step=1)
        brancher.record_branch_outcome(b1, True, 500, 0.01, 1.0)

        comp = BranchComparator(store)
        ranked = comp.rank_branches("task_001")
        assert len(ranked) >= 1
        assert "score" in ranked[0]


# ── RL Navigator ───────────────────────────────────────────────────
class TestNavigator:
    def test_decide(self):
        nav = RLNavigator(model_path=tempfile.mktemp(suffix=".json"))
        features = np.random.randn(128).astype(np.float32)
        dec = nav.should_branch(features, step=3)
        assert isinstance(dec.action, NavAction)
        assert isinstance(dec.should_branch, bool)

    def test_learn(self):
        nav = RLNavigator(model_path=tempfile.mktemp(suffix=".json"))
        f = np.random.randn(128).astype(np.float32)
        for _ in range(50):
            nav.learn_from_branch(f, 1, True, False, step=5)
        assert nav.train_steps > 0

    def test_save_load(self):
        path = tempfile.mktemp(suffix=".json")
        nav = RLNavigator(model_path=path)
        f = np.random.randn(128).astype(np.float32)
        q1 = nav.q_network.predict(f)
        nav.save()
        nav2 = RLNavigator(model_path=path)
        q2 = nav2.q_network.predict(f)
        np.testing.assert_array_almost_equal(q1, q2, decimal=5)

    def test_context_features(self):
        nav = RLNavigator()
        f = np.random.randn(96).astype(np.float32)
        ctx = {"tests_failed": 3, "patches_generated": 2}
        built = nav._build_features(f, step=5, context=ctx)
        assert built.shape == (128,)
        assert built[97] == 0.3  # tests_failed / 10


# ── Visualizer ─────────────────────────────────────────────────────
class TestVisualizer:
    def test_timeline(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        viz = TimelineVisualizer(store)
        tl = viz.generate_timeline("task_001")
        assert tl["total_events"] == 5
        assert len(tl["timeline"]) == 5
        assert "color" in tl["timeline"][0]

    def test_branch_tree(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        brancher = TaskBrancher(store)
        brancher.branch("task_001", from_step=2)
        viz = TimelineVisualizer(store)
        tree = viz.generate_branch_tree("task_001")
        assert tree["total_branches"] >= 1

    def test_step_detail(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        viz = TimelineVisualizer(store)
        detail = viz.generate_step_detail("task_001", 0)
        assert detail["found"]
        assert "icon" in detail


# ── Exporter ───────────────────────────────────────────────────────
class TestExporter:
    def test_export_jsonl(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        exp = TrajectoryExporter(store)
        path = tempfile.mktemp(suffix=".jsonl")
        count = exp.export_jsonl(output_path=path)
        assert count == 5

    def test_export_trajectories(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        exp = TrajectoryExporter(store)
        path = tempfile.mktemp(suffix=".jsonl")
        count = exp.export_trajectories(output_path=path)
        assert count == 1  # 1 task

    def test_export_stats(self):
        store = make_store()
        for e in make_events():
            store.save_event(e)
        exp = TrajectoryExporter(store)
        stats = exp.get_export_stats()
        assert stats["total_tasks"] == 1
        assert stats["total_events"] == 5
