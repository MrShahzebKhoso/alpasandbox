"""Scenarios: small YAML files describing the road, the ego start and scripted actors.

Search order: explicit path, then each directory in `SEARCH_DIRS` (user dirs first, then the
built-in ones). Add your own with `--scenarios-dir DIR` or ALPASANDBOX_SCENARIOS=DIR1:DIR2.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

BUILTIN_DIR = Path(__file__).parent / "scenarios"
SEARCH_DIRS: list[Path] = [Path(p) for p in os.environ.get("ALPASANDBOX_SCENARIOS", "").split(os.pathsep) if p]
SEARCH_DIRS.append(BUILTIN_DIR)


@dataclass
class ActorSpec:
    kind: str                 # pedestrian | vehicle | cone
    x: float                  # metres along the road from the ego start (ego starts at x=0)
    y: float                  # lateral offset from the ego-lane centre line, left positive
    vx: float = 0.0           # along-road speed (m/s); negative = oncoming
    vy: float = 0.0           # lateral speed (m/s), left positive
    start_t: float = 0.0      # actor stays put until this time
    decel: float = 0.0        # m/s^2 braking applied along its heading
    decel_start_t: float = 0.0
    lateral_start_t: float | None = None   # lateral motion starts here (default: start_t)
    y_target: float | None = None          # lateral motion stops at this offset (lane change)


@dataclass
class Scenario:
    name: str
    description: str = ""
    duration: float = 20.0
    ego_speed: float = 8.0
    nav_text: str = "Continue straight."
    road_amplitude: float = 0.0     # S-curve amplitude (m); 0 = straight
    road_wavelength: float = 200.0
    tags: list[str] = field(default_factory=list)
    actors: list[ActorSpec] = field(default_factory=list)


def _path(name_or_path: str) -> Path | None:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p
    for d in SEARCH_DIRS:
        for ext in (".yaml", ".yml"):
            if (d / f"{name_or_path}{ext}").exists():
                return d / f"{name_or_path}{ext}"
    return None


def load_scenario(name_or_path: str) -> Scenario:
    p = _path(name_or_path)
    if p is None:
        raise FileNotFoundError(f"Scenario '{name_or_path}' not found. Available: {list_scenarios()}")
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    d.setdefault("name", p.stem)
    allowed = {f.name for f in fields(ActorSpec)}
    actors = []
    for a in d.pop("actors", []) or []:
        bad = set(a) - allowed
        if bad:
            raise ValueError(f"{p.name}: unknown actor keys {sorted(bad)} (allowed: {sorted(allowed)})")
        actors.append(ActorSpec(**a))
    bad = set(d) - {f.name for f in fields(Scenario)}
    if bad:
        raise ValueError(f"{p.name}: unknown scenario keys {sorted(bad)}")
    return Scenario(actors=actors, **d)


def list_scenarios() -> list[str]:
    names: dict[str, None] = {}
    for d in SEARCH_DIRS:
        if d.is_dir():
            for p in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
                names.setdefault(p.stem, None)
    return sorted(names)


def describe_scenarios() -> list[dict]:
    out = []
    for n in list_scenarios():
        try:
            s = load_scenario(n)
            out.append({"name": n, "description": s.description, "duration": s.duration,
                        "nav_text": s.nav_text, "tags": s.tags, "actors": len(s.actors)})
        except Exception as exc:  # a broken user file should not take the app down
            out.append({"name": n, "description": f"(invalid: {exc})", "duration": 0, "nav_text": "",
                        "tags": [], "actors": 0})
    return out
