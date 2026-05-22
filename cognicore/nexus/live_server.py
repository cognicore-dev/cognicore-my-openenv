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

@app.get("/")
async def ui():
    p = Path(__file__).parent / "live_ui.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found</h1>")


if __name__ == "__main__":
    port = int(os.environ.get("NEXUS_PORT", "8420"))
    print(f"\n  NEXUS Live Runtime v1.0")
    print(f"  http://localhost:{port}")
    print(f"  WebSocket: ws://localhost:{port}/ws\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
