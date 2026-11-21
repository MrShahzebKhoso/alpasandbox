"""Configuration: defaults < alpasandbox.toml < environment < command-line flags.

`alpasandbox init` writes a commented starter file. See docs/configuration.md.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

DEFAULTS: dict = {
    "app": {"host": "127.0.0.1", "port": 8000, "scenario": "pedestrian_crossing", "realtime": True,
            "autoplay": False, "open_browser": True},
    "sim": {"backend": "builtin", "image_size": "640x360", "scenarios_dir": ""},
    "carla": {"host": "localhost", "port": 2000, "town": "scenario", "traffic": 0, "spawn_index": 0,
              "tm_port": 8010, "weather": "ClearNoon", "start_speed": -1,
              "ego_blueprint": "vehicle.tesla.model3", "follow_spectator": True},
    "model": {"name": "oracle", "runtime": "local", "settings": {}},
    "runtime": {
        "subprocess": {"python": "", "cwd": "", "pythonpath": "auto", "env": {}, "timeout": 300,
                       "startup_timeout": 1800},
        "wsl": {"distro": "", "workdir": "", "venv": "", "python": "python", "pythonpath": "auto",
                "env": {}, "timeout": 300, "startup_timeout": 1800},
        "remote": {"url": "", "token": "", "timeout": 300},
    },
}

ENV = {  # environment variable -> config path
    "ALPASANDBOX_MODEL": ("model", "name"),
    "ALPASANDBOX_RUNTIME": ("model", "runtime"),
    "ALPASANDBOX_BACKEND": ("sim", "backend"),
    "ALPASANDBOX_REMOTE_URL": ("runtime", "remote", "url"),
    "ALPASANDBOX_TOKEN": ("runtime", "remote", "token"),
    "ALPASANDBOX_WSL_DISTRO": ("runtime", "wsl", "distro"),
    "CARLA_HOST": ("carla", "host"),
}

STARTER = '''# alpasandbox.toml: settings for `alpasandbox app` and `alpasandbox run`.
# Command-line flags override this file. Delete what you do not need.

[app]
scenario = "pedestrian_crossing"
port = 8000
realtime = true          # pace the sim at 10 Hz; false = as fast as the model allows

[sim]
backend = "builtin"      # builtin (no install) | carla
image_size = "640x360"   # per camera; all cameras share one size

[carla]
host = "localhost"       # from WSL with CARLA on Windows, use the Windows host IP
port = 2000
town = "scenario"        # build the scenario's road; or a map name, e.g. "Town10HD_Opt"
traffic = 0              # autopilot cars to add (named towns only)
weather = "ClearNoon"    # any carla.WeatherParameters preset, or "default"
tm_port = 8010           # Traffic Manager port; must not clash with the app port

[model]
name = "edge"            # edge | super | oracle | cruise | your plugin | module:Class
runtime = "local"        # local | subprocess | wsl | remote

[model.settings]         # passed to the model adapter
# model_id = "nvidia/Alpamayo2-Edge"   # or a local checkpoint directory
# num_traj_sets = 3                    # draw alternative plans in the bird's-eye view

[runtime.wsl]            # Windows: CARLA on Windows, model inside WSL2
distro = "Ubuntu-24.04"
workdir = "~/alpamayo2-edge"           # upstream repo checkout inside WSL
venv = ".venv"                         # relative to workdir, or absolute
# env = { HF_HUB_OFFLINE = "1" }       # after the first download

[runtime.subprocess]     # model in another Python on this machine
# python = "/path/to/alpamayo2-edge/.venv/bin/python"

[runtime.remote]         # model served by `alpasandbox serve-model` elsewhere
# url = "http://gpu-box:8100"
'''


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set(d: dict, path: tuple, value) -> None:
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def find_config(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        return p
    for p in (Path("alpasandbox.toml"), Path.home() / ".config" / "alpasandbox" / "alpasandbox.toml"):
        if p.exists():
            return p
    return None


def load_config(explicit: str | None = None) -> tuple[dict, Path | None]:
    cfg = copy.deepcopy(DEFAULTS)
    path = find_config(explicit)
    if path:
        with open(path, "rb") as f:
            cfg = _merge(cfg, tomllib.load(f))
    for var, keypath in ENV.items():
        if os.environ.get(var):
            _set(cfg, keypath, os.environ[var])
    return cfg, path


def parse_value(v: str):
    """`--set key=value` values: JSON-ish scalars."""
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v
