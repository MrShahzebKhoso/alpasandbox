"""Waypoints -> steering/throttle/brake. Pure pursuit + proportional speed control.

`max_steer` must be the vehicle's real maximum road-wheel angle (radians): `Control.steer = 1`
means full lock, so a wrong value scales every steering command. Backends expose it as
`max_steer_rad` (CARLA reads it from the vehicle's physics) and the engine passes it in.
"""
from __future__ import annotations

import math

import numpy as np

from .core import Control, PolicyOutput


class Controller:
    def __init__(self, wheelbase: float = 2.8, max_steer: float = 0.5, kp: float = 1.2,
                 max_accel: float = 3.0, max_decel: float = 8.0, launch_speed: float = 1.0,
                 launch_throttle: float = 0.4, stop_dist: float = 2.0, look_time: float = 1.5,
                 look_min: float = 5.0, look_max: float = 30.0):
        self.wheelbase, self.max_steer, self.kp = wheelbase, max_steer, kp
        self.max_accel, self.max_decel = max_accel, max_decel
        # Starting from rest: a plan whose peak speed reaches `launch_speed` is a "go", and gets at least
        # `launch_throttle` until the car rolls at `launch_speed`. Vehicles have a throttle dead band
        # from standstill: CARLA's Tesla never moves at 0.1 and needs ~2 s of 0.2 before it creeps.
        # A plan that ends within `stop_dist` metres means "stay stopped": hold the brake.
        self.launch_speed, self.launch_throttle, self.stop_dist = launch_speed, launch_throttle, stop_dist
        self.look_time, self.look_min, self.look_max = look_time, look_min, look_max
        self.debug: dict = {}      # internals of the last call, for traces (not part of the control)

    def __call__(self, out: PolicyOutput, v: float) -> Control:
        wp, dt = np.asarray(out.waypoints, float), out.dt
        if len(wp) == 0 or not np.isfinite(wp).all():
            self.debug = {"branch": "invalid_plan"}
            return Control(brake=1.0)
        # Speed the plan asks for one second ahead: the instantaneous plan speed there, not the
        # average from the origin, so the car follows the plan's speed profile while moving.
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(np.vstack([[0.0, 0.0], wp]), axis=0).T))])

        def speed_at(t: float) -> float:
            i = int(np.clip(round(t / dt), 1, len(s) - 2))
            return float(s[i + 1] - s[i - 1]) / (2 * dt)

        v_target = speed_at(1.0)
        # Go/stop is decided from the whole plan. From a standstill, VLA plans put the motion late
        # (Edge at a green light: ~0.1 m in 1 s, ~0.5 m in 3 s, 12 m in 6.4 s), and a fresh plan every
        # tick keeps the start "two seconds from now". Reading only the first seconds left the car parked.
        v_plan = np.diff(s) / dt
        v_peak = float(np.max(v_plan))
        stop = bool(s[-1] < self.stop_dist)
        # The same holds for stopping: Edge's "stop for the red light" plans brake late (speed at 0.5 s is
        # still ~the current speed). Tracking the first second of a fresh plan every tick never reaches
        # the braking, and the stop point slides ahead of the car. If the plan comes to rest (after
        # moving: a standing start is not a stop), decelerate at least enough to stop where it stops.
        moving = np.flatnonzero(v_plan >= 0.3)
        rest = np.flatnonzero(v_plan[moving[0]:] < 0.3) + moving[0] if len(moving) else np.array([], int)
        s_stop = float(s[rest[0]]) if len(rest) else None
        a_stop = v * v / (2 * max(s_stop, 0.5)) if s_stop is not None and v > 0.5 else 0.0
        launch = (not stop and s_stop is None and v < self.launch_speed and v_peak >= self.launch_speed)

        # Pure pursuit on the plan point `look_time` ahead (5-30 m). Edge's lateral corrections also come
        # late in the plan: drifting at 17 m/s, its plan is still straight at 15 m and bends after that.
        # A 15 m cap saw almost none of it and the car held a 2.5 deg heading error into a divider.
        look = min(self.look_max, max(self.look_min, self.look_time * max(v, float(s[-1]) / (len(s) - 1) / dt)))
        dist = np.hypot(wp[:, 0], wp[:, 1])
        idx = int(np.argmax(dist >= look)) if (dist >= look).any() else len(wp) - 1
        lx, ly = wp[idx]
        ld = max(math.hypot(lx, ly), 1e-3)
        wheel_angle = math.atan(self.wheelbase * 2 * ly / ld**2)
        steer = wheel_angle / self.max_steer

        a = self.kp * (v_target - v)
        if a_stop > 0:
            a = min(a, -a_stop)
        throttle = max(0.0, min(1.0, a / self.max_accel))
        brake = max(0.0, min(1.0, -a / self.max_decel))
        branch = "track" if a >= 0 else ("stop_point" if a_stop > 0 and a == -a_stop else "slow")
        if launch:
            throttle, brake, branch = max(throttle, self.launch_throttle), 0.0, "launch"
        elif stop and v < 0.5:
            throttle, brake, branch = 0.0, max(brake, 0.6), "hold"

        def s_at(t: float) -> float:
            return round(float(s[min(len(s) - 1, round(t / dt))]), 3)

        self.debug = {"s_1s": s_at(1.0), "s_2s": s_at(2.0), "s_3s": s_at(3.0), "s_end": round(float(s[-1]), 3),
                      "x_end": round(float(wp[-1, 0]), 3), "v_target": round(v_target, 3),
                      "v_peak": round(v_peak, 3), "v": round(float(v), 3), "a": round(float(a), 3),
                      "s_stop": None if s_stop is None else round(s_stop, 2), "a_stop": round(a_stop, 3),
                      "stop": stop, "branch": branch,
                      "look_m": round(look, 2), "look_y": round(float(ly), 3), "steer": round(float(steer), 4)}
        return Control(steer=max(-1.0, min(1.0, steer)), throttle=throttle, brake=brake)
