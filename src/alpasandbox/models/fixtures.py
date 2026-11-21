"""GPU-free test fixtures. They let the whole app run anywhere (CI, laptops, the docs demo).

* `oracle` reads simulator ground truth (NOT images). It is a rule-based planner, not a model.
* `cruise` drives straight at constant speed: a blind lower bound.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import ModelInfo, Observation, PolicyOutput, QueryResult
from ..rig import EDGE_DRIVE_SPEC, EDGE_VQA_SPEC
from .base import Policy

PHRASES = {"pedestrian": "A pedestrian is entering the lane ahead, so yield and stop before the crossing",
           "vehicle": "The lead vehicle ahead is slowing down, so decelerate to keep a safe gap",
           "cone": "A traffic cone blocks the ego lane ahead, so slow down and stop before it"}


class OraclePolicy(Policy):
    def __init__(self, cruise: float = 8.0, n: int = 64, dt: float = 0.1, decel: float = 3.0,
                 standoff: float = 7.0):
        self.cruise, self.n, self.dt, self.decel, self.standoff = cruise, n, dt, decel, standoff
        self._info = ModelInfo(
            "oracle", "Oracle (ground truth)", EDGE_DRIVE_SPEC, EDGE_VQA_SPEC,
            description="Rule-based planner that reads simulator ground truth. A test fixture, "
                        "not a driving model. Answers questions from templates.",
            privileged=True)

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _ahead(self, p):
        e = p["ego"]
        c, s = math.cos(e.yaw), math.sin(e.yaw)
        for a in p.get("actors", []):
            dx, dy = a["x"] - e.x, a["y"] - e.y
            fx, fy = c * dx + s * dy, -s * dx + c * dy
            yield a, fx, fy, c * a.get("vx", 0) + s * a.get("vy", 0), -s * a.get("vx", 0) + c * a.get("vy", 0)

    def act(self, obs: Observation) -> PolicyOutput:
        p = obs.privileged
        if not p or "lane_y" not in p:
            raise RuntimeError("The oracle needs a backend that exposes ground truth (the builtin backend)")
        e, lane_y = p["ego"], p["lane_y"]
        c, s = math.cos(e.yaw), math.sin(e.yaw)
        stop_s, why = None, ""
        for a, fx, fy, _vfx, vfy in self._ahead(p):
            if not 0 < fx < 45:
                continue
            eta = fx / max(e.v, 2.0)
            if abs(fy + vfy * eta) < 2.0 or abs(fy) < 2.0:
                gap = fx - self.standoff
                if stop_s is None or gap < stop_s:
                    stop_s, why = max(gap, 0.0), f"{PHRASES.get(a['kind'], 'Obstacle ahead; stopping')} (~{fx:.0f} m)."
        off = e.y - lane_y(e.x)
        pts, dist, v = [], 0.0, e.v
        for _ in range(self.n):
            target = self.cruise
            if stop_s is not None:
                target = min(target, math.sqrt(max(0.0, 2 * self.decel * (stop_s - dist))))
            v = min(v + 2.0 * self.dt, target) if v < target else max(v - 6.0 * self.dt, target)
            dist += v * self.dt
            wx = e.x + dist * c
            wy = lane_y(wx) + off * (1 - min(1.0, dist / 20.0))
            dx, dy = wx - e.x, wy - e.y
            pts.append((c * dx + s * dy, -s * dx + c * dy))
        return PolicyOutput(np.asarray(pts), self.dt, reasoning=why, meta={"nominal": stop_s is None})

    def query(self, obs: Observation, question: str) -> QueryResult:
        p = obs.privileged or {}
        if "ego" not in p:
            return QueryResult("The oracle can only describe scenes from the builtin backend.")
        seen = []
        for a, fx, fy, _, _ in self._ahead(p):
            if math.hypot(fx, fy) > 80:
                continue
            side = "ahead" if fx > 0 else "behind"
            lat = "in the ego lane" if abs(fy) < 1.8 else ("to the left" if fy > 0 else "to the right")
            seen.append(f"a {a['kind']} {abs(fx):.0f} m {side}, {lat}")
        scene = ("I see " + "; ".join(seen) + ".") if seen else "The road around the ego vehicle is clear."
        return QueryResult(f"{scene} Ego speed is {p['ego'].v * 3.6:.0f} km/h. "
                           "(Oracle answer from ground-truth templates, not a language model.)",
                           meta={"oracle": True})


class CruisePolicy(Policy):
    def __init__(self, speed: float = 8.0, n: int = 64, dt: float = 0.1):
        self.speed, self.n, self.dt = speed, n, dt
        self._info = ModelInfo("cruise", "Cruise (blind baseline)", EDGE_DRIVE_SPEC,
                               description="Drives straight at constant speed. A plumbing test and a floor.")

    @property
    def info(self) -> ModelInfo:
        return self._info

    def act(self, obs: Observation | None) -> PolicyOutput:
        s = self.speed * self.dt * np.arange(1, self.n + 1)
        return PolicyOutput(np.stack([s, np.zeros_like(s)], 1), self.dt, meta={"nominal": True})
