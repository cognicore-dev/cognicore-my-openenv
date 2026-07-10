from concurrent.futures import ThreadPoolExecutor

from cognicore.integrations.task_queue import NexusTask, NexusTaskQueue
from cognicore.research.persistent_store import PersistentCognitionStore


def test_persistent_store_handles_concurrent_writes(tmp_path):
    db_path = tmp_path / "persistent_memory.db"

    def worker(worker_id: int) -> None:
        store = PersistentCognitionStore(str(db_path))
        for i in range(25):
            store.store_episode(
                session_id=f"session-{worker_id}",
                category="timeout",
                bug_id=f"bug-{worker_id}-{i}",
                action="store_episode",
                outcome="ok",
                success=(i % 2 == 0),
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(worker, range(8)))

    store = PersistentCognitionStore(str(db_path))
    stats = store.get_stats()
    assert stats["total_episodes"] == 200


def test_task_queue_handles_concurrent_instances(tmp_path):
    db_path = tmp_path / "task_queue.db"

    def worker(worker_id: int) -> None:
        queue = NexusTaskQueue(db_path=str(db_path))
        for i in range(15):
            queue.submit(
                NexusTask(
                    source="manual",
                    repo="demo/repo",
                    title=f"task-{worker_id}-{i}",
                    description="timeout regression coverage",
                )
            )

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(worker, range(6)))

    queue = NexusTaskQueue(db_path=str(db_path))
    stats = queue.stats()
    assert stats["total"] == 90
