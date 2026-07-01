"""
NEXUS Dashboard Backend — FastAPI server with WebSocket streaming.
Serves both the REST API and the React frontend.

Start with: cognicore ui
"""
import os, json, time, asyncio, sqlite3, webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from cognicore.nexus.trajectory_store import TrajectoryStore
from cognicore.research.persistent_store import PersistentCognitionStore
from cognicore.nexus.token_tracker import TokenTracker

NEXUS_DIR = Path.home() / ".cognicore"
NEXUS_DIR.mkdir(exist_ok=True)
EVENTS_FILE = NEXUS_DIR / "events.jsonl"
DB_PATH = NEXUS_DIR / "nexus.db"
FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"

# ── WebSocket Manager ──
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)

manager = ConnectionManager()

# ── Event file watcher ──
async def watch_events():
    """Watch ~/.cognicore/events.jsonl for new events and broadcast."""
    last_pos = 0
    while True:
        try:
            if EVENTS_FILE.exists():
                with open(EVENTS_FILE, 'r') as f:
                    f.seek(last_pos)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                event = json.loads(line)
                                await manager.broadcast(event)
                            except json.JSONDecodeError:
                                pass
                    last_pos = f.tell()
        except Exception:
            pass
        await asyncio.sleep(0.5)

# ── Database ──
def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY, value TEXT
    )""")
    return db

# ── App ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(watch_events())
    yield
    task.cancel()

app = FastAPI(title="NEXUS Dashboard", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── WebSocket endpoint ──
@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ── API Routes ──
@app.get("/api/stats")
def get_stats():
    ts = TrajectoryStore()
    stats = ts.get_stats()
    ps = PersistentCognitionStore()
    insights = ps.get_cross_session_insights("all")

    total_cost = 0
    total_tokens = 0
    solved = stats.get("solved", 0)
    total = stats.get("total_trajectories", 0)

    # Calculate from trajectories
    try:
        export_path = ts.export_for_training()
        with open(export_path) as f:
            for line in f:
                t = json.loads(line)
                total_tokens += t.get("total_tokens", 0)
                total_cost += t.get("total_tokens", 0) * 0.00000015
    except Exception:
        pass

    return {
        "total_tasks": total,
        "solved": solved,
        "solve_rate": round(solved / max(total, 1) * 100, 1),
        "total_tokens": total_tokens,
        "total_cost": round(total_cost, 4),
        "memory_entries": sum(insights.get("successful_tactics", {}).values()) +
                         sum(insights.get("failed_tactics", {}).values()),
        "policies": stats.get("policies", []),
        "active_connections": len(manager.active),
    }

@app.get("/api/tasks")
def get_tasks(status: Optional[str] = None, policy: Optional[str] = None,
              limit: int = 100):
    ts = TrajectoryStore()
    try:
        export_path = ts.export_for_training()
        tasks = []
        with open(export_path) as f:
            for line in f:
                t = json.loads(line)
                task = {
                    "id": t["task_id"],
                    "repo": t.get("repo", "unknown"),
                    "category": t.get("category", "unknown"),
                    "status": "solved" if t.get("solved") else "failed",
                    "policy": t.get("policy", "unknown"),
                    "tokens": t.get("total_tokens", 0),
                    "cost": round(t.get("total_tokens", 0) * 0.00000015, 6),
                    "attempts": len(t.get("steps", [])) // 2,
                    "reward": round(t.get("total_reward", 0), 3),
                    "timestamp": t.get("timestamp", ""),
                }
                if status and task["status"] != status:
                    continue
                if policy and task["policy"] != policy:
                    continue
                tasks.append(task)
        return tasks[-limit:]
    except Exception as e:
        return []

@app.get("/api/tasks/{task_id}")
def get_task_detail(task_id: str):
    ts = TrajectoryStore()
    try:
        export_path = ts.export_for_training()
        with open(export_path) as f:
            for line in f:
                t = json.loads(line)
                if t["task_id"] == task_id:
                    return t
    except Exception:
        pass
    return {"error": "Task not found"}

@app.get("/api/memory")
def get_memory():
    ps = PersistentCognitionStore()
    insights = ps.get_cross_session_insights("all")
    categories = {}
    for cat in ["arithmetic", "encoding", "off_by_one", "none_handling",
                "parsing", "error_handling", "type_conversion", "validation",
                "recursion", "routing", "translation", "deep_copy"]:
        cat_insights = ps.get_cross_session_insights(cat)
        if cat_insights.get("successful_tactics") or cat_insights.get("failed_tactics"):
            categories[cat] = cat_insights
    return {
        "global": insights,
        "categories": categories,
        "total_entries": sum(insights.get("successful_tactics", {}).values()) +
                        sum(insights.get("failed_tactics", {}).values()),
    }

@app.get("/api/analytics")
def get_analytics():
    ts = TrajectoryStore()
    cp = ts.compare_policies()
    policy_data = []
    for p, d in cp.items():
        policy_data.append({
            "policy": p, "n": d["n"], "solved": d["solved"],
            "mean_reward": round(d["mean_reward"], 3),
            "avg_tokens": round(d["avg_tokens"]),
            "solve_rate": round(d["solved"] / max(d["n"], 1) * 100, 1),
        })

    # Category breakdown
    category_data = {}
    try:
        export_path = ts.export_for_training()
        with open(export_path) as f:
            for line in f:
                t = json.loads(line)
                cat = t.get("category", "unknown")
                if cat not in category_data:
                    category_data[cat] = {"total": 0, "solved": 0, "tokens": 0}
                category_data[cat]["total"] += 1
                category_data[cat]["solved"] += int(t.get("solved", 0))
                category_data[cat]["tokens"] += t.get("total_tokens", 0)
    except Exception:
        pass

    return {
        "policies": policy_data,
        "categories": [{"category": k, **v} for k, v in category_data.items()],
    }

@app.get("/api/settings")
def get_settings():
    db = get_db()
    rows = db.execute("SELECT key, value FROM settings").fetchall()
    settings = {r["key"]: r["value"] for r in rows}
    return {
        "default_policy": settings.get("default_policy", "test_first"),
        "max_tokens": int(settings.get("max_tokens", "1000000")),
        "budget_usd": float(settings.get("budget_usd", "50.0")),
        "port": int(settings.get("port", "7842")),
        "gemini_key": "****" if os.environ.get("GEMINI_API_KEY") else "",
        "openai_key": "****" if os.environ.get("OPENAI_API_KEY") else "",
        "anthropic_key": "****" if os.environ.get("ANTHROPIC_API_KEY") else "",
    }

@app.post("/api/settings")
def update_settings(data: dict):
    db = get_db()
    for k, v in data.items():
        if k in ("default_policy", "max_tokens", "budget_usd", "port"):
            db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                      (k, str(v)))
    db.commit()
    return {"ok": True}

@app.post("/api/events")
async def post_event(event: dict):
    """Receive events from cognicore runtime."""
    event["timestamp"] = datetime.utcnow().isoformat()
    with open(EVENTS_FILE, 'a') as f:
        f.write(json.dumps(event) + "\n")
    await manager.broadcast(event)
    return {"ok": True}

# ── Serve React frontend ──
@app.get("/")
@app.get("/{path:path}")
def serve_frontend(path: str = ""):
    if path.startswith("api/") or path.startswith("ws/"):
        return JSONResponse({"error": "not found"}, 404)
    if FRONTEND_DIR.exists():
        file_path = FRONTEND_DIR / path
        if file_path.is_file():
            return FileResponse(file_path)
        index = FRONTEND_DIR / "index.html"
        if index.exists():
            return FileResponse(index)
    # Fallback: serve inline dashboard
    return FileResponse(Path(__file__).parent / "frontend" / "index.html")


def start_server(port=7842, open_browser=True, host=None):
    """Start the NEXUS dashboard server."""
    import uvicorn
    host = host or os.environ.get("NEXUS_HOST", "127.0.0.1")
    print(f"\n  ======================================")
    print(f"   NEXUS Dashboard -- {host}:{port}")
    print(f"  ======================================\n")
    if open_browser:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
