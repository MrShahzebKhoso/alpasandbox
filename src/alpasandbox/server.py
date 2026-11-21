"""Model API: serve any model over HTTP so the GPU can live on another machine.

    GET  /healthz            liveness
    GET  /v1/info            ModelInfo (input specs, capabilities)
    POST /v1/reset           new episode
    POST /v1/drive           encoded Observation -> 64 waypoints + reasoning (+ candidates)
    POST /v1/query           {observation, question} -> answer   (models with VQA only)

Set ALPASANDBOX_TOKEN to require `Authorization: Bearer <token>`.
"""
from __future__ import annotations

import os
import threading

from fastapi import Depends, FastAPI, HTTPException, Request

from . import __version__
from .models.base import Policy
from .models.protocol import output_to_dict, query_to_dict
from .wire import decode_observation


def create_app(policy: Policy, token: str | None = None) -> FastAPI:
    app = FastAPI(title="alpasandbox model server", version=__version__)
    lock = threading.Lock()   # one GPU, one request at a time
    token = token or os.environ.get("ALPASANDBOX_TOKEN")

    def auth(request: Request) -> None:
        if token and request.headers.get("authorization") != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong bearer token")

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "model": policy.info.name, "version": __version__}

    @app.get("/v1/info", dependencies=[Depends(auth)])
    def info():
        return policy.info.to_dict()

    @app.post("/v1/reset", dependencies=[Depends(auth)])
    def reset():
        with lock:
            policy.reset()
        return {"ok": True}

    @app.post("/v1/drive", dependencies=[Depends(auth)])
    async def drive(request: Request):
        obs = decode_observation(await request.json())
        with lock:
            out = policy.act(obs)
        return output_to_dict(out)

    @app.post("/v1/query", dependencies=[Depends(auth)])
    async def query(request: Request):
        if not policy.info.supports_vqa:
            raise HTTPException(400, f"{policy.info.display_name} does not answer questions")
        body = await request.json()
        question = str(body.get("question", "")).strip()
        if not question:
            raise HTTPException(422, "question must be non-empty")
        obs = decode_observation(body["observation"])
        with lock:
            res = policy.query(obs, question)
        return query_to_dict(res)

    return app
