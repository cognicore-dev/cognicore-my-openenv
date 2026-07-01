"""
NEXUS Live Server — Full runtime observability with ALL subsystems.
FastAPI + WebSocket. Real events only.
"""
import os, sys, json, time, asyncio, threading
from pathlib import Path
from typing import Set
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("pip install fastapi uvicorn"); sys.exit(1)

from cognicore.nexus.live_instrument import FullInstrumentor, LiveEvent

_inst = None
_sockets: Set[WebSocket] = set()
_loop = None
_history = []


def _get():
    global _inst
    if _inst is None:
        from cognicore.nexus.autonomous import NexusRunner
        _inst = FullInstrumentor(NexusRunner())
        _inst.on_event(_broadcast)
    return _inst


def _broadcast(event):
    if not _sockets or not _loop: return
    asyncio.run_coroutine_threadsafe(_send_all(event.to_json()), _loop)


async def _send_all(data):
    dead = set()
    for ws in _sockets.copy():
        try: await ws.send_text(data)
        except: dead.add(ws)
    _sockets.difference_update(dead)


@asynccontextmanager
async def lifespan(app):
    global _loop
    _loop = asyncio.get_event_loop()
    yield

app = FastAPI(title="NEXUS Live Runtime", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _sockets.add(ws)
    try:
        await ws.send_text(json.dumps({
            "event_id": "connected", "phase": "system",
            "action": "connected",
            "detail": json.dumps(_get().get_subsystem_status()),
            "timestamp": time.time(), "status": "done",
        }))
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            if data.get("type") == "solve":
                prompt = data.get("prompt", "")
                repo_path = data.get("repo_path", "")
                if repo_path == ".":
                    repo_path = str(Path(__file__).parent.parent.parent)
                if prompt:
                    threading.Thread(target=_run, args=(prompt, data.get("repo", ""), repo_path), daemon=True).start()

            elif data.get("type") == "scan":
                text = data.get("text", "")
                if text:
                    result = _get().scan_threat(text)
                    await ws.send_text(json.dumps({
                        "phase": "immune", "action": "scan_result",
                        "detail": json.dumps(result),
                        "timestamp": time.time(), "status": "done",
                    }))

            elif data.get("type") == "replay":
                task_id = data.get("task_id", "")
                if task_id:
                    events = _get().get_replay_events(task_id)
                    for e in events:
                        await ws.send_text(json.dumps({
                            "phase": e.get("event_type", ""), "action": e.get("action", ""),
                            "detail": e.get("output_text", ""), "status": "replay",
                            "timestamp": e.get("timestamp", 0),
                            "tokens_in": e.get("tokens_in", 0),
                            "tokens_out": e.get("tokens_out", 0),
                        }))

    except WebSocketDisconnect: pass
    finally: _sockets.discard(ws)


def _run(prompt, repo, repo_path):
    inst = _get()
    try:
        result = inst.solve(prompt, repo=repo, repo_path=repo_path, auto_pr=False)
        _history.append(result)
    except Exception as e:
        _broadcast(LiveEvent(task_id=inst.task_id, phase="error",
                            action="execution_error", detail=str(e), status="failed"))


# REST API endpoints
@app.get("/api/subsystems")
async def subsystems():
    return JSONResponse(_get().get_subsystem_status())

@app.get("/api/immune/stats")
async def immune_stats():
    return JSONResponse(_get().get_immune_stats())

@app.get("/api/immune/dashboard")
async def immune_dashboard():
    return JSONResponse(_get().get_immune_dashboard())

@app.post("/api/immune/scan")
async def immune_scan(body: dict):
    return JSONResponse(_get().scan_threat(body.get("text", "")))

@app.get("/api/memory/stats")
async def memory_stats():
    return JSONResponse(_get().get_memory_stats())

@app.get("/api/replay/{task_id}")
async def replay(task_id: str):
    return JSONResponse({"events": _get().get_replay_events(task_id)})

@app.get("/api/branches/{task_id}")
async def branches(task_id: str):
    return JSONResponse({"branches": _get().get_branches(task_id)})

@app.get("/api/tasks")
async def tasks():
    try:
        from cognicore.replay.store import EventStore
        store = EventStore()
        ids = store.get_task_ids()
        items = []
        for tid in ids[-20:]:
            evts = store.load_events(tid)
            if evts:
                items.append({
                    "task_id": tid, "events": len(evts),
                    "first": evts[0].timestamp, "last": evts[-1].timestamp,
                    "solved": any(e.event_type == "task_solved" for e in evts),
                    "tokens": sum(e.tokens_in + e.tokens_out for e in evts),
                })
        return JSONResponse({"tasks": items})
    except:
        return JSONResponse({"tasks": []})

@app.get("/api/stats")
async def stats():
    return JSONResponse(_get().get_stats())

# SWE-bench endpoints
_swe_results = {}
_swe_running = False

@app.get("/api/swe/tasks")
async def swe_tasks():
    """List available SWE-bench mini tasks."""
    try:
        from cognicore.research.swebench import load_swebench_tasks
        tasks = load_swebench_tasks()
        return JSONResponse({"tasks": [
            {"id": t.id, "repo": t.repo, "category": t.category,
             "issue": t.issue, "description": t.description[:100]}
            for t in tasks
        ]})
    except Exception as e:
        return JSONResponse({"error": str(e), "tasks": []})

@app.get("/api/swe/results")
async def swe_results():
    """Get latest SWE-bench results."""
    return JSONResponse(_swe_results if _swe_results else {"status": "no results yet"})

@app.get("/api/swe/status")
async def swe_status():
    """Check if a benchmark is running."""
    return JSONResponse({"running": _swe_running, "has_results": bool(_swe_results)})

@app.post("/api/swe/run")
async def swe_run(body: dict = {}):
    """Start a SWE-bench run in background."""
    global _swe_running
    if _swe_running:
        return JSONResponse({"error": "Already running"}, status_code=409)
    mode = body.get("mode", "mini")
    attempts = body.get("attempts", 3)
    limit = body.get("limit", 50)
    threading.Thread(target=_run_swe, args=(mode, attempts, limit), daemon=True).start()
    return JSONResponse({"status": "started", "mode": mode})

def _run_swe(mode, attempts, limit):
    global _swe_results, _swe_running
    _swe_running = True
    try:
        from cognicore.nexus.swe_runner import NexusSWERunner
        runner = NexusSWERunner(max_attempts=attempts, on_event=_swe_broadcast)
        if mode == "mini":
            results = runner.run_mini_bench()
        else:
            results = runner.run_swebench_lite(limit=limit)
        results.compute()
        _swe_results = results.to_dict()
        # Export predictions
        out_dir = Path(__file__).parent.parent.parent / "experiments"
        out_dir.mkdir(exist_ok=True)
        results.export_predictions(str(out_dir / f"predictions_{mode}.jsonl"))
    except Exception as e:
        _swe_results = {"error": str(e)}
    finally:
        _swe_running = False

def _swe_broadcast(event):
    """Broadcast SWE-bench events to WebSocket clients."""
    if not _sockets or not _loop: return
    asyncio.run_coroutine_threadsafe(_send_all(json.dumps({
        "phase": "swe_bench", "action": event.get("action", ""),
        "detail": event.get("detail", ""), "status": event.get("status", ""),
        "task_id": event.get("task_id", ""), "timestamp": event.get("timestamp", 0),
    })), _loop)


@app.get("/")
async def ui():
    p = Path(__file__).parent / "live_ui.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>")


if __name__ == "__main__":
    port = int(os.environ.get("NEXUS_PORT", "8420"))
    host = os.environ.get("NEXUS_HOST", "127.0.0.1")
    print(f"\n  NEXUS Live Runtime v1.0")
    print(f"  http://{host}:{port}")
    print(f"  WebSocket: ws://{host}:{port}/ws\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")

