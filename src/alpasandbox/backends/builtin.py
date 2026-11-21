"""Zero-dependency backend: kinematic bicycle, procedural road, scripted actors, PIL cameras.
Runs on CPU anywhere (CI, laptops). Imagery is schematic, so use it to test plumbing, not
to judge the model's real-world driving."""
from __future__ import annotations

import math

from ..core import Control, EgoState, StepResult
from ..rig import REFERENCE_RIG
from ..scenario import Scenario
from .base import Backend
from .render import render_camera

SIZES = {"pedestrian": (0.5, 0.5, 1.7), "vehicle": (4.5, 1.9, 1.5), "cone": (0.4, 0.4, 0.7)}
WHEELBASE, MAX_STEER, LANE_HALF = 2.8, 0.5, 1.75
ROAD_EDGES = (-LANE_HALF, 5.25)          # right edge, left edge (relative to ego-lane centre)


def _circles(x, y, yaw, kind):
    if kind == "vehicle":
        c, s = math.cos(yaw), math.sin(yaw)
        return [(x + c * d, y + s * d, 1.1) for d in (-1.2, 0.0, 1.2)]
    return [(x, y, 0.4 if kind == "pedestrian" else 0.3)]


class BuiltinBackend(Backend):
    name = "builtin"
    wheelbase = WHEELBASE
    max_steer_rad = MAX_STEER

    def __init__(self, image_size=(480, 270), rig=REFERENCE_RIG, dt=0.1, render=True):
        self.image_size, self.rig, self.dt, self.render = image_size, rig, dt, render

    # --- road ---------------------------------------------------------------
    def lane_y(self, x: float) -> float:
        a = self.sc.road_amplitude
        return a * math.sin(2 * math.pi * x / self.sc.road_wavelength) if a else 0.0

    def _lane_slope(self, x: float) -> float:
        a = self.sc.road_amplitude
        return a * 2 * math.pi / self.sc.road_wavelength * math.cos(2 * math.pi * x / self.sc.road_wavelength) if a else 0.0

    # --- lifecycle ----------------------------------------------------------
    def reset(self, scenario: Scenario) -> StepResult:
        self.sc, self.t, self._hit = scenario, 0.0, set()
        self.ego = EgoState(0.0, 0.0, self.lane_y(0.0), math.atan(self._lane_slope(0.0)), scenario.ego_speed)
        self.actors = []
        for i, a in enumerate(scenario.actors):
            yaw = math.atan2(0.0 if a.lateral_start_t else a.vy, a.vx) if (a.vx or a.vy) else 0.0
            self.actors.append({"id": i, "kind": a.kind, "x": a.x, "y": self.lane_y(a.x) + a.y,
                                "vx": a.vx, "vy": a.vy, "yaw": yaw, "spec": a, "size": SIZES[a.kind]})
        return self._result(False)

    def step(self, c: Control) -> StepResult:
        h, e = 0.02, self.ego
        for _ in range(int(round(self.dt / h))):
            acc = 3.0 * min(max(c.throttle, 0), 1) - 8.0 * min(max(c.brake, 0), 1) - 0.05 * e.v
            e.v = max(0.0, e.v + acc * h)
            e.steer = max(-1.0, min(1.0, c.steer)) * MAX_STEER
            e.yaw += e.v / WHEELBASE * math.tan(e.steer) * h
            e.x += e.v * math.cos(e.yaw) * h
            e.y += e.v * math.sin(e.yaw) * h
            self.t += h
            self._move_actors(h)
        e.t = self.t
        collided = self._check_collision()
        return self._result(collided)

    # --- internals ----------------------------------------------------------
    def _move_actors(self, h):
        for a in self.actors:
            s = a["spec"]
            if self.t < s.start_t:
                continue
            if s.decel and self.t >= s.decel_start_t:
                sp = math.hypot(a["vx"], a["vy"])
                if sp > 0:
                    k = max(0.0, sp - s.decel * h) / sp
                    a["vx"], a["vy"] = a["vx"] * k, a["vy"] * k
            lat_start = s.start_t if s.lateral_start_t is None else s.lateral_start_t
            vy = a["vy"] if self.t >= lat_start else 0.0
            if s.y_target is not None and vy:
                rel = a["y"] - self.lane_y(a["x"])
                if (vy > 0 and rel >= s.y_target) or (vy < 0 and rel <= s.y_target):
                    a["vy"] = vy = 0.0
            a["x"] += a["vx"] * h
            a["y"] += vy * h
            if a["vx"] or vy:
                a["yaw"] = math.atan2(vy, a["vx"]) if a["kind"] != "pedestrian" or vy else a["yaw"]

    def _check_collision(self) -> bool:
        e = self.ego
        ego_c = [(e.x + math.cos(e.yaw) * d, e.y + math.sin(e.yaw) * d, 1.1) for d in (0.3, 1.8, 3.3)]
        new = False
        for a in self.actors:
            hit = any(math.hypot(ax - ex, ay - ey) < ar + er
                      for ex, ey, er in ego_c for ax, ay, ar in _circles(a["x"], a["y"], a["yaw"], a["kind"]))
            if hit and a["id"] not in self._hit:
                self._hit.add(a["id"])
                new = True
        return new

    def _lines(self, x0: float) -> list[dict]:
        xs = range(int(x0) - 40, int(x0) + 110, 2)
        lo, hi = ROAD_EDGES
        return [{"kind": kind, "pts": [[x, round(self.lane_y(x) + off, 2)] for x in xs]}
                for kind, off in (("edge", lo), ("edge", hi), ("center", LANE_HALF))]

    def _result(self, collided: bool) -> StepResult:
        e = self.ego
        slope = self._lane_slope(e.x)
        off = (e.y - self.lane_y(e.x)) / math.sqrt(1 + slope**2)
        actors = [{k: a[k] for k in ("id", "kind", "x", "y", "yaw", "size")} for a in self.actors]
        images = {}
        if self.render:
            images = {c.cam_id: render_camera(c, e, actors, self.lane_y, self.image_size, ROAD_EDGES)
                      for c in self.cameras()}
        priv = {"actors": [{**a, "vx": s["vx"], "vy": s["vy"]} for a, s in zip(actors, self.actors, strict=False)],
                "lane_y": self.lane_y, "road_edges": ROAD_EDGES, "ego": EgoState(**vars(e)),
                "lines": self._lines(e.x)}
        return StepResult(t=self.t, ego=EgoState(**vars(e)), images=images, collision=collided,
                          off_road=not (ROAD_EDGES[0] - 0.5 < off < ROAD_EDGES[1] + 0.5),
                          lane_offset=off, done=self.t >= self.sc.duration - 1e-9, privileged=priv)
