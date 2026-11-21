"""The web app: static UI + REST + one WebSocket per browser tab."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .backends.base import Backend
from .models.base import Policy
from .session import Session

WEB = Path(__file__).parent / "web"
COMMANDS = {"play", "pause", "step", "reset", "nav", "ask", "realtime", "stop_on_incident", "signal"}


class _Client:
    """Per-browser outbox. Frames are coalesced (only the newest matters); other events queue."""

    def __init__(self):
        self.events: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.frame: dict | None = None
        self.wake = asyncio.Event()

    def push(self, msg: dict) -> None:
        if msg.get("type") == "frame":
            self.frame = msg
        elif not self.events.full():
            self.events.put_nowait(msg)
        self.wake.set()


class _RevalidatedStatic(StaticFiles):
    """Static UI files with `Cache-Control: no-cache`: browsers keep them but re-check every load (a 304
    when unchanged). Without it a browser may pair a fresh index.html with a stale app.js for hours."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(backend_factory: Callable[[], Backend], policy_factory: Callable[[], Policy], scenario: str,
               *, realtime: bool = True, autoplay: bool = False, runtime_label: str = "local") -> FastAPI:
    clients: set[_Client] = set()
    state: dict = {}

    def broadcast(msg: dict) -> None:
        for c in list(clients):
            c.push(msg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()
        session = Session(backend_factory, policy_factory, scenario,
                          lambda m: loop.call_soon_threadsafe(broadcast, m),
                          realtime=realtime, autoplay=autoplay, runtime_label=runtime_label)
        state["session"] = session
        session.start()
        yield
        await asyncio.to_thread(session.shutdown)

    app = FastAPI(title="AlpaSandbox", version=__version__, lifespan=lifespan)
    app.mount("/static", _RevalidatedStatic(directory=WEB), name="static")

    def session() -> Session:
        return state["session"]

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/api/info")
    def info():
        s = session().info()
        s.pop("logs", None)
        return s

    @app.get("/api/frame")
    def frame():
        return session().last_frame or {}

    @app.post("/api/command")
    async def command(request: Request):
        body = await request.json()
        cmd = body.pop("cmd", None)
        if cmd not in COMMANDS:
            return JSONResponse({"error": f"unknown command {cmd!r}", "commands": sorted(COMMANDS)}, 400)
        session().command(cmd, **body)
        return {"ok": True}

    @app.get("/api/trace.jsonl")
    def trace():
        s = session()
        name = f"alpasandbox_{s.scenario_name}_{(s.policy.info.name if s.policy else 'model')}.jsonl"
        return PlainTextResponse(s.trace_jsonl(), media_type="application/x-ndjson",
                                 headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/api/summary")
    def summary():
        return session().summary or {}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        client = _Client()
        clients.add(client)
        s = session()
        await sock.send_json(s.info())
        if s.last_frame:
            await sock.send_json(s.last_frame)

        async def receive():
            while True:
                msg = await sock.receive_json()
                cmd = msg.pop("cmd", None)
                if cmd in COMMANDS:
                    s.command(cmd, **msg)

        async def send():
            while True:
                await client.wake.wait()
                client.wake.clear()
                while not client.events.empty():
                    await sock.send_json(client.events.get_nowait())
                if client.frame is not None:
                    f, client.frame = client.frame, None
                    await sock.send_json(f)

        tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()
            clients.discard(client)

    return app
