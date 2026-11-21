"""Closed-loop metrics. 'Incidents' = new collisions + off-road events. NVIDIA reports km per
*at-fault* incident; we cannot judge fault, so treat ours as an upper bound on incidents."""
from __future__ import annotations

import math

from .core import StepResult


class Metrics:
    def __init__(self):
        self.distance = 0.0
        self.collisions = self.off_road_events = self.steps = self.reasoning_steps = 0
        self.lane_sq = 0.0
        self.latencies: list[float] = []
        self.speeds: list[float] = []
        self.max_decel = 0.0
        self._prev = None
        self._prev_v: float | None = None
        self._prev_t: float | None = None
        self._was_off = False

    def update(self, res: StepResult, latency_ms: float = 0.0, reasoned: bool = False):
        if self._prev is not None:
            self.distance += math.hypot(res.ego.x - self._prev[0], res.ego.y - self._prev[1])
        if self._prev_v is not None and self._prev_t is not None and res.t > self._prev_t:
            self.max_decel = max(self.max_decel, (self._prev_v - res.ego.v) / (res.t - self._prev_t))
        self._prev, self._prev_v, self._prev_t = (res.ego.x, res.ego.y), res.ego.v, res.t
        self.collisions += int(res.collision)
        self.off_road_events += int(res.off_road and not self._was_off)
        self._was_off = res.off_road
        self.lane_sq += res.lane_offset**2
        self.steps += 1
        self.reasoning_steps += int(reasoned)
        self.latencies.append(latency_ms)
        self.speeds.append(res.ego.v)

    def summary(self) -> dict:
        inc = self.collisions + self.off_road_events
        n = max(self.steps, 1)
        lat = sorted(self.latencies) or [0.0]
        return {
            "distance_m": round(self.distance, 1), "collisions": self.collisions,
            "off_road_events": self.off_road_events, "incidents": inc,
            "km_per_incident": round(self.distance / 1000 / inc, 3) if inc else None,
            "lane_offset_rms_m": round(math.sqrt(self.lane_sq / n), 3),
            "mean_speed_mps": round(sum(self.speeds) / n, 2),
            "max_decel_mps2": round(self.max_decel, 2),
            "mean_policy_latency_ms": round(sum(self.latencies) / n, 1),
            "p95_policy_latency_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
            "reasoning_fraction": round(self.reasoning_steps / n, 3),
            "steps": self.steps,
        }
