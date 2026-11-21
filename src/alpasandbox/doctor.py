"""`alpasandbox doctor`: check the environment for the configured setup and say how to fix gaps."""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from . import __version__

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"
_SYM = {OK: "✓", WARN: "!", FAIL: "✗", INFO: "·"}
_COLOR = {OK: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m", INFO: "\033[90m"}


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def in_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, status: str, what: str, detail: str = "", fix: str = "") -> None:
        self.rows.append((status, what, detail, fix))

    def print(self) -> int:
        color = _supports_color()
        for status, what, detail, fix in self.rows:
            sym = _SYM[status]
            if color:
                sym = f"{_COLOR[status]}{sym}\033[0m"
            print(f" {sym} {what:30s} {detail}")
            if fix and status in (WARN, FAIL):
                print(f"   {'':30s} → {fix}")
        fails = sum(r[0] == FAIL for r in self.rows)
        warns = sum(r[0] == WARN for r in self.rows)
        print(f"\n{fails} problem(s), {warns} warning(s).")
        return 1 if fails else 0


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_torch(r: Report, need_gpu_gb: float) -> None:
    if not _has("torch"):
        r.add(FAIL, "PyTorch", "not installed in this Python",
              "install the model package (it pins torch 2.8), or use --runtime subprocess/wsl/remote")
        return
    import torch
    r.add(OK, "PyTorch", torch.__version__)
    if not torch.cuda.is_available():
        r.add(FAIL, "CUDA GPU", "torch.cuda.is_available() is False",
              "install a CUDA build of PyTorch and an NVIDIA driver; on WSL use the Windows driver, not a Linux one")
        return
    props = torch.cuda.get_device_properties(0)
    gb = props.total_memory / 1024**3
    status = OK if gb >= need_gpu_gb * 0.97 else WARN   # "16 GB" cards report ~15.99 GiB
    r.add(status, "CUDA GPU", f"{props.name}, {gb:.1f} GB, CUDA {torch.version.cuda}",
          f"this model is validated with >= {need_gpu_gb:.0f} GB; expect out-of-memory errors")
    bf16 = torch.cuda.is_bf16_supported()
    r.add(OK if bf16 else WARN, "BF16 support", str(bf16), "Alpamayo runs in bfloat16; pre-Ampere GPUs are not supported")


def run_doctor(cfg: dict) -> int:
    r = Report()
    m, rt_name, sim = cfg["model"], cfg["model"]["runtime"], cfg["sim"]
    print(f"alpasandbox {__version__} doctor\n")

    # --- host ---
    osname = "WSL2" if in_wsl() else platform.system()
    r.add(INFO, "Host", f"{osname} {platform.release()}, Python {sys.version.split()[0]} ({sys.executable})")
    r.add(INFO, "Config file", cfg.get("_path") or "none (defaults; `alpasandbox init` writes one)")
    r.add(INFO, "Setup", f"model={m['name']} runtime={rt_name} backend={sim['backend']}")
    missing = [p for p in ("numpy", "PIL", "yaml", "fastapi", "uvicorn", "httpx") if not _has(p)]
    r.add(FAIL if missing else OK, "Core packages", ", ".join(missing) + " missing" if missing else "all present",
          "pip install -U alpasandbox")

    # --- model ---
    from .models.registry import BUILTIN, canonical
    name = canonical(m["name"])
    entry = BUILTIN.get(name)
    if rt_name == "local":
        if entry and entry.requires:
            if _has(entry.requires):
                r.add(OK, f"{entry.display_name} package", entry.requires)
            else:
                r.add(FAIL, f"{entry.display_name} package", f"'{entry.requires}' not importable here", entry.install_hint)
            if sys.version_info[:2] != (3, 12):
                r.add(WARN, "Python for the model", f"{sys.version_info[0]}.{sys.version_info[1]}",
                      "upstream Alpamayo pins Python 3.12; consider --runtime subprocess with its own venv")
            _check_torch(r, 16 if name == "edge" else 80)
            hf = os.environ.get("HF_TOKEN") or (Path.home() / ".cache" / "huggingface" / "token").exists()
            r.add(OK if hf else WARN, "Hugging Face login", "token found" if hf else "no token found",
                  "run `hf auth login` (accept the model terms on its Hugging Face page first)")
        elif entry is None:
            r.add(INFO, "Model", f"'{name}' is a plugin or module path; it is checked when loaded")
        else:
            r.add(OK, "Model", f"{entry.display_name} needs no GPU")
    elif rt_name == "subprocess":
        py = cfg["runtime"]["subprocess"].get("python") or sys.executable
        exists = Path(py).exists() or shutil.which(py)
        r.add(OK if exists else FAIL, "Worker Python", py, "set [runtime.subprocess] python to the model env's python")
        if exists and entry and entry.requires:
            code = f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{entry.requires}') else 1)"
            ok = subprocess.run([py, "-c", code], capture_output=True).returncode == 0
            r.add(OK if ok else FAIL, f"{entry.display_name} in worker", entry.requires, f"in that env: {entry.install_hint}")
    elif rt_name == "wsl":
        _check_wsl(r, cfg, entry)
    elif rt_name == "remote":
        url = cfg["runtime"]["remote"].get("url", "")
        if not url:
            r.add(FAIL, "Remote URL", "not set", "--remote-url http://gpu-host:8100")
        else:
            try:
                import httpx
                d = httpx.get(url.rstrip("/") + "/healthz", timeout=5).json()
                r.add(OK, "Remote model server", f"{url} serves '{d.get('model')}' (v{d.get('version')})")
            except Exception as exc:
                r.add(FAIL, "Remote model server", f"{url}: {type(exc).__name__}",
                      "start `alpasandbox serve-model --model edge` there and open the port in its firewall")

    # --- simulator ---
    if sim["backend"] == "carla":
        c = cfg["carla"]
        r.add(OK if _has("carla") else FAIL, "CARLA Python API", "importable" if _has("carla") else "missing",
              "pip install carla==<your server version> (client and server versions must match)")
        up = _port_open(c["host"], int(c["port"]))
        hint = ("start CarlaUE4 / CarlaUnreal first" + ("; from WSL use the Windows host IP "
                "(`ip route | awk '/default/ {print $3}'`) and allow CARLA through the Windows firewall" if in_wsl() else ""))
        r.add(OK if up else FAIL, "CARLA server", f"{c['host']}:{c['port']} {'reachable' if up else 'not reachable'}", hint)
    else:
        r.add(OK, "Simulator", "builtin (schematic imagery, no install)")
    return r.print()


def _check_wsl(r: Report, cfg: dict, entry) -> None:
    w = cfg["runtime"]["wsl"]
    if os.name != "nt":
        r.add(FAIL, "WSL runtime", "only works from Windows",
              "inside WSL or on Linux use --runtime local (or subprocess)")
        return
    if not shutil.which("wsl.exe"):
        r.add(FAIL, "wsl.exe", "not found", "wsl --install, then install Ubuntu-24.04")
        return
    out = subprocess.run(["wsl.exe", "-l", "-q"], capture_output=True)
    distros = [d.strip() for d in out.stdout.decode("utf-16-le", "ignore").replace("\x00", "").splitlines() if d.strip()]
    want = w.get("distro") or (distros[0] if distros else "")
    r.add(OK if want in distros else FAIL, "WSL distro", f"{want or '(none)'}; installed: {', '.join(distros) or 'none'}",
          "set [runtime.wsl] distro to one of the installed distros")
    if want not in distros:
        return
    from .models.runtimes import _sh_path
    venv = w.get("venv", "")
    script = "set -e; " + (f"cd {_sh_path(w['workdir'])}; " if w.get("workdir") else "") + \
        (f". {_sh_path(venv.rstrip('/') + '/bin/activate')}; " if venv else "") + \
        f"{w.get('python', 'python')} -c 'import sys;print(sys.version.split()[0])'; " + \
        (f"{w.get('python', 'python')} -c 'import {entry.requires}' && echo MODEL_OK; " if entry and entry.requires else "") + \
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"
    res = subprocess.run(["wsl.exe", "-d", want, "--exec", "bash", "--noprofile", "--norc", "-c", script],
                         capture_output=True, text=True, timeout=120)
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    if res.returncode != 0 and not lines:
        r.add(FAIL, "WSL Python env", (res.stderr or "failed").strip()[-200:],
              "check [runtime.wsl] workdir and venv; the venv must contain the upstream model package")
        return
    r.add(OK, "WSL Python", lines[0] if lines else "?")
    if entry and entry.requires:
        ok = "MODEL_OK" in lines
        r.add(OK if ok else FAIL, f"{entry.display_name} in WSL", entry.requires, f"inside WSL: {entry.install_hint}")
    gpu = [ln for ln in lines if "," in ln and "MiB" in ln]
    r.add(OK if gpu else WARN, "GPU inside WSL", gpu[0] if gpu else "nvidia-smi found no GPU",
          "update the NVIDIA Windows driver; do not install a Linux NVIDIA driver inside WSL")
