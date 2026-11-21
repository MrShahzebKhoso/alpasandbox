"""Where a model runs, relative to the sandbox.

local       same Python process (Linux, or native Windows with CUDA PyTorch)
subprocess  another Python on the same machine (e.g. the upstream model's own .venv)
wsl         a Python inside WSL2, driven from Windows (CARLA on Windows, model in Linux)
remote      an HTTP model server (`alpasandbox serve-model`) on any machine

`subprocess` and `wsl` both speak the JSON-lines worker protocol (see `alpasandbox.worker`).
"""
from __future__ import annotations

import base64
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .. import __version__
from ..core import ModelInfo, Observation, PolicyOutput, QueryResult
from ..wire import encode_observation
from .base import Policy
from .protocol import output_from_dict, query_from_dict

LogFn = Callable[[str], None]


def _default_log(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------
# stdio worker client
# ------------------------------------------------------------------------------------------
class WorkerError(RuntimeError):
    pass


class StdioWorkerPolicy(Policy):
    """Drives `python -m alpasandbox.worker` in a child process over stdin/stdout."""

    def __init__(self, command: list[str], *, cwd: str | None = None, env: dict | None = None,
                 startup_timeout: float = 1800.0, timeout: float = 300.0, log: LogFn | None = None,
                 label: str = "worker"):
        self.command, self.timeout, self.label = command, timeout, label
        self.log = log or _default_log
        self._id = 0
        self._responses: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.log(f"[{label}] starting: {' '.join(shlex.quote(c) for c in command[:6])}{' ...' if len(command) > 6 else ''}")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=cwd, env=env, text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=flags)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        try:
            ready = self._wait(None, startup_timeout, "startup (model loading)")
        except Exception:
            self.close()
            raise
        if not ready.get("success"):
            self.close()
            raise WorkerError(f"{label} failed to start: {ready.get('error')}\n{ready.get('traceback', '')}")
        self._info = ModelInfo.from_dict(ready["info"])
        self._info.extra = {**self._info.extra, "runtime": label}

    # -- plumbing --------------------------------------------------------------------------
    def _pump_stdout(self) -> None:
        assert self.process.stdout
        for line in self.process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                self.log(f"[{self.label}] (non-protocol stdout) {line[:300]}")
        self._responses.put({"type": "eof"})

    def _pump_stderr(self) -> None:
        assert self.process.stderr
        for line in self.process.stderr:
            line = line.rstrip()
            if line:
                self.log(f"[{self.label}] {line}")

    def _wait(self, rid: int | None, timeout: float, what: str) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{self.label}: no reply to {what} within {timeout:.0f}s")
            try:
                msg = self._responses.get(timeout=min(left, 1.0))
            except queue.Empty:
                if self.process.poll() is not None and self._responses.empty():
                    raise WorkerError(f"{self.label} exited with code {self.process.returncode} during {what}; "
                                      "see the log above") from None
                continue
            if msg.get("type") == "eof":
                raise WorkerError(f"{self.label} closed its output during {what} "
                                  f"(exit code {self.process.poll()}); see the log above")
            if rid is None and msg.get("type") == "ready":
                return msg
            if rid is not None and msg.get("id") == rid:
                return msg
            # stale reply from a request that already timed out: drop it

    def _call(self, op: str, timeout: float | None = None, **payload) -> dict:
        with self._lock:
            if self.process.poll() is not None:
                raise WorkerError(f"{self.label} is not running (exit code {self.process.returncode})")
            self._id += 1
            rid = self._id
            assert self.process.stdin
            self.process.stdin.write(json.dumps({"id": rid, "op": op, **payload}, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            msg = self._wait(rid, timeout or self.timeout, op)
        if not msg.get("success"):
            raise WorkerError(f"{self.label} {op} failed: {msg.get('error')}")
        return msg

    # -- Policy API ------------------------------------------------------------------------
    @property
    def info(self) -> ModelInfo:
        return self._info

    def reset(self) -> None:
        self._call("reset", timeout=30)

    def act(self, obs: Observation) -> PolicyOutput:
        t0 = time.perf_counter()
        msg = self._call("predict", observation=encode_observation(obs))
        return output_from_dict(msg["output"], (time.perf_counter() - t0) * 1000)

    def query(self, obs: Observation, question: str) -> QueryResult:
        t0 = time.perf_counter()
        msg = self._call("query", observation=encode_observation(obs), question=question)
        return query_from_dict(msg["output"], (time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        p = getattr(self, "process", None)
        if p is None or p.poll() is not None:
            return
        try:
            assert p.stdin
            p.stdin.write('{"op":"shutdown"}\n')
            p.stdin.flush()
            p.stdin.close()
            p.wait(timeout=10)
        except Exception:
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


# ------------------------------------------------------------------------------------------
# command builders
# ------------------------------------------------------------------------------------------
def _settings_arg(settings: dict) -> str:
    return base64.b64encode(json.dumps(settings or {}).encode()).decode()


def stage_package(cache_root: str | None = None) -> Path:
    """Copy the pure-Python `alpasandbox` package to a private directory and return that directory.

    Adding it to the worker's PYTHONPATH makes the worker importable in any environment without
    exposing the rest of this interpreter's site-packages (whose numpy/torch binaries would clash).
    """
    pkg = Path(__file__).resolve().parents[1]
    root = Path(cache_root or os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "alpasandbox" / "worker-src"
    dest = root / __version__
    target = dest / "alpasandbox"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(pkg, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "web"))
    return dest


def subprocess_command(python: str, model: str, settings: dict) -> list[str]:
    return [python, "-u", "-m", "alpasandbox.worker", "--model", model, "--settings-b64", _settings_arg(settings)]


def windows_to_wsl_path(path: str, distro: str | None = None) -> str:
    """Convert C:\\a\\b to /mnt/c/a/b, asking `wslpath` when available (honours custom automount roots)."""
    if os.name == "nt" and shutil.which("wsl.exe"):
        try:
            cmd = ["wsl.exe"] + (["-d", distro] if distro else []) + ["--exec", "wslpath", "-a", path]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return f"/mnt/{p[0].lower()}{p[2:]}"
    return p


def _sh_path(p: str) -> str:
    """Shell-quote a path but keep a leading ~ expandable."""
    if p == "~":
        return '"$HOME"'
    if p.startswith("~/"):
        return '"$HOME"/' + shlex.quote(p[2:])
    return shlex.quote(p)


def wsl_command(model: str, settings: dict, *, distro: str = "", workdir: str = "", venv: str = "",
                pythonpath: str = "", python: str = "python", env: dict | None = None) -> list[str]:
    lines = ["set -e"]
    if workdir:
        lines.append(f"cd {_sh_path(workdir)}")
    if venv:
        lines.append(f". {_sh_path(venv.rstrip('/') + '/bin/activate')}")
    if pythonpath:
        lines.append(f'export PYTHONPATH={_sh_path(pythonpath)}"${{PYTHONPATH:+:$PYTHONPATH}}"')
    lines.append("export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1")
    for k, v in (env or {}).items():
        lines.append(f"export {k}={shlex.quote(str(v))}")
    lines.append("exec " + " ".join(shlex.quote(x) for x in
                                     [python, "-u", "-m", "alpasandbox.worker", "--model", model,
                                      "--settings-b64", _settings_arg(settings)]))
    head = ["wsl.exe"] + (["-d", distro] if distro else [])
    return head + ["--exec", "bash", "--noprofile", "--norc", "-c", "\n".join(lines)]


def start_subprocess_worker(model: str, settings: dict, cfg: dict, log: LogFn | None = None) -> StdioWorkerPolicy:
    python = cfg.get("python") or sys.executable
    env = {**os.environ, **{k: str(v) for k, v in (cfg.get("env") or {}).items()}}
    pp = cfg.get("pythonpath", "auto")
    if pp == "auto":
        pp = str(stage_package())
    if pp:
        env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return StdioWorkerPolicy(subprocess_command(python, model, settings), cwd=cfg.get("cwd") or None, env=env,
                             startup_timeout=float(cfg.get("startup_timeout", 1800)),
                             timeout=float(cfg.get("timeout", 300)), log=log, label=f"{model}@subprocess")


def start_wsl_worker(model: str, settings: dict, cfg: dict, log: LogFn | None = None) -> StdioWorkerPolicy:
    if os.name != "nt":
        raise RuntimeError("The 'wsl' runtime drives WSL from Windows. On Linux (or inside WSL) use "
                           "'local' or 'subprocess' instead.")
    if not shutil.which("wsl.exe"):
        raise RuntimeError("wsl.exe not found. Install WSL2 (`wsl --install`) or use another runtime.")
    distro = cfg.get("distro", "")
    pp = cfg.get("pythonpath", "auto")
    if pp == "auto":
        pp = windows_to_wsl_path(str(stage_package()), distro or None)
    cmd = wsl_command(model, settings, distro=distro, workdir=cfg.get("workdir", ""), venv=cfg.get("venv", ""),
                      pythonpath=pp, python=cfg.get("python", "python"), env=cfg.get("env"))
    return StdioWorkerPolicy(cmd, startup_timeout=float(cfg.get("startup_timeout", 1800)),
                             timeout=float(cfg.get("timeout", 300)), log=log,
                             label=f"{model}@wsl{':' + distro if distro else ''}")


# ------------------------------------------------------------------------------------------
# HTTP
# ------------------------------------------------------------------------------------------
class RemotePolicy(Policy):
    """Talks to `alpasandbox serve-model` over HTTP, so the GPU can live on another machine."""

    def __init__(self, url: str, timeout: float = 300.0, token: str | None = None):
        import httpx

        self.url = url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)
        r = self.client.get(f"{self.url}/v1/info")
        r.raise_for_status()
        self._info = ModelInfo.from_dict(r.json())
        self._info.extra = {**self._info.extra, "runtime": f"remote {self.url}"}

    @property
    def info(self) -> ModelInfo:
        return self._info

    def reset(self) -> None:
        self.client.post(f"{self.url}/v1/reset").raise_for_status()

    def act(self, obs: Observation) -> PolicyOutput:
        t0 = time.perf_counter()
        r = self.client.post(f"{self.url}/v1/drive", json=encode_observation(obs))
        r.raise_for_status()
        return output_from_dict(r.json(), (time.perf_counter() - t0) * 1000)

    def query(self, obs: Observation, question: str) -> QueryResult:
        t0 = time.perf_counter()
        r = self.client.post(f"{self.url}/v1/query", json={"observation": encode_observation(obs), "question": question})
        r.raise_for_status()
        return query_from_dict(r.json(), (time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        self.client.close()


__all__ = ["StdioWorkerPolicy", "RemotePolicy", "WorkerError", "start_subprocess_worker", "start_wsl_worker",
           "wsl_command", "subprocess_command", "windows_to_wsl_path", "stage_package"]
