"""Stdio model worker: `python -m alpasandbox.worker --model edge`.

Runs a model in its own Python environment (another venv, a conda env, or inside WSL) and
talks JSON lines over stdin/stdout. See docs/protocol.md. Kept dependency-light on purpose:
it only needs numpy, pillow and the model's own packages.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import traceback


def _claim_stdout():
    """Reserve the real stdout file descriptor for the protocol.

    Redirecting `sys.stdout` alone is not enough: CUDA, C++ extensions and some progress bars
    write straight to fd 1 and would corrupt the JSON stream. Duplicate fd 1 for the protocol,
    then point fd 1 (and sys.stdout) at stderr so stray output lands in the log instead.
    """
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(proto_fd, "w", encoding="utf-8", buffering=1)


def main(argv=None) -> None:
    p = argparse.ArgumentParser("alpasandbox.worker")
    p.add_argument("--model", default="edge")
    p.add_argument("--settings-b64", default="", help="base64 JSON of model settings")
    a = p.parse_args(argv)
    proto = _claim_stdout()

    def send(msg: dict) -> None:
        proto.write(json.dumps(msg, separators=(",", ":")) + "\n")
        proto.flush()

    try:
        from .models.protocol import output_to_dict, query_to_dict
        from .models.registry import load_local
        from .wire import decode_observation

        settings = json.loads(base64.b64decode(a.settings_b64).decode()) if a.settings_b64 else {}
        print(f"[worker] loading model '{a.model}' (python {sys.version.split()[0]}, pid {os.getpid()})",
              file=sys.stderr, flush=True)
        policy = load_local(a.model, settings)
        send({"type": "ready", "success": True, "info": policy.info.to_dict()})
        print("[worker] ready", file=sys.stderr, flush=True)
    except BaseException as exc:  # noqa: BLE001 - report anything, including ImportError/SystemExit
        traceback.print_exc(file=sys.stderr)
        send({"type": "ready", "success": False, "error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()[-4000:]})
        return

    for line in sys.stdin:
        if not line.strip():
            continue
        req: dict = {}
        try:
            req = json.loads(line)
            op, rid = req.get("op"), req.get("id")
            if op == "shutdown":
                break
            if op == "ping":
                send({"id": rid, "success": True})
            elif op == "reset":
                policy.reset()
                send({"id": rid, "success": True})
            elif op == "predict":
                out = policy.act(decode_observation(req["observation"]))
                send({"id": rid, "success": True, "output": output_to_dict(out)})
            elif op == "query":
                res = policy.query(decode_observation(req["observation"]), req["question"])
                send({"id": rid, "success": True, "output": query_to_dict(res)})
            else:
                send({"id": rid, "success": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # keep serving after a failed request
            traceback.print_exc(file=sys.stderr)
            send({"id": req.get("id"), "success": False, "error": f"{type(exc).__name__}: {exc}"})
    policy.close()


if __name__ == "__main__":
    main()
