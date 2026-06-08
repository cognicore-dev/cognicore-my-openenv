"""
NEXUS Scheduler — cron-style recurring task execution.
Nightly bug fixes, weekly benchmarks, CI monitoring.
"""
import os, json, time, threading
from pathlib import Path
from datetime import datetime
from cognicore.integrations.task_queue import NexusTask, NexusTaskQueue, TaskSource

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

SCHEDULE_FILE = Path.home() / ".cognicore" / "schedule.yml"


class NexusScheduler:
    """Cron-based NEXUS task scheduler."""

    def __init__(self, config=None, queue=None):
        self.config = config or {}
        self.queue = queue or NexusTaskQueue()
        self.scheduler = BackgroundScheduler() if SCHEDULER_AVAILABLE else None
        self._jobs = {}
        self._history = []

    @property
    def available(self):
        return SCHEDULER_AVAILABLE

    def load_schedules(self, path=None):
        """Load schedule config from YAML file."""
        path = path or SCHEDULE_FILE
        if not path.exists():
            return []

        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return data.get("schedules", [])
        except ImportError:
            # Fallback: simple JSON-like parsing
            return self.config.get("schedules", [])

    def start(self):
        """Start the scheduler with all configured jobs."""
        if not self.scheduler:
            print("  [Scheduler] APScheduler not installed. pip install apscheduler")
            return

        schedules = self.load_schedules()
        for sched in schedules:
            self.add_job(sched)

        self.scheduler.start()
        print(f"  [Scheduler] Started with {len(self._jobs)} jobs")

    def stop(self):
        if self.scheduler:
            self.scheduler.shutdown(wait=False)

    def add_job(self, schedule: dict):
        """Add a scheduled job."""
        name = schedule.get("name", f"job_{len(self._jobs)}")
        cron = schedule.get("cron", "0 2 * * *")
        task_type = schedule.get("task", "fix_labeled_issues")
        repo = schedule.get("repo", "")

        def run_job():
            self._execute_schedule(name, task_type, repo, schedule)

        if self.scheduler:
            trigger = CronTrigger.from_crontab(cron)
            job = self.scheduler.add_job(run_job, trigger, id=name, replace_existing=True)
            self._jobs[name] = {"schedule": schedule, "job": job, "last_run": None}

    def _execute_schedule(self, name, task_type, repo, config):
        """Execute a scheduled task."""
        now = datetime.utcnow().isoformat()
        print(f"  [Scheduler] Running: {name} ({task_type})")

        task = NexusTask(
            source=TaskSource.SCHEDULED.value,
            repo=repo,
            title=f"Scheduled: {name}",
            description=f"Scheduled task: {task_type} on {repo}",
            priority=3,  # low priority for scheduled tasks
            policy=config.get("policy", "test_first"),
            max_attempts=config.get("max_tasks", 10),
            metadata={"schedule_name": name, "task_type": task_type,
                      "triggered_at": now}
        )

        task_id = self.queue.submit(task)
        self._history.append({
            "name": name, "task_id": task_id, "triggered_at": now, "status": "submitted"})

        if name in self._jobs:
            self._jobs[name]["last_run"] = now

    def run_now(self, name: str):
        """Trigger a schedule immediately."""
        if name in self._jobs:
            sched = self._jobs[name]["schedule"]
            self._execute_schedule(name, sched.get("task", ""), sched.get("repo", ""), sched)
            return True
        return False

    def list_jobs(self) -> list:
        """List all scheduled jobs."""
        jobs = []
        for name, info in self._jobs.items():
            s = info["schedule"]
            jobs.append({
                "name": name, "cron": s.get("cron", ""),
                "task": s.get("task", ""), "repo": s.get("repo", ""),
                "last_run": info.get("last_run"), "active": True})
        return jobs

    def history(self, limit=20) -> list:
        return self._history[-limit:]

    def pause_all(self):
        if self.scheduler:
            self.scheduler.pause()

    def resume_all(self):
        if self.scheduler:
            self.scheduler.resume()
