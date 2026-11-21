"""Longitudinal control from rest. Regression for the CARLA "ego never moves on green" bug (0.2.x)."""
import math

import numpy as np
import pytest

from alpasandbox.controller import Controller
from alpasandbox.core import PolicyOutput

T = np.arange(1, 65) * 0.1          # Edge: 64 waypoints at 0.1 s


def _plan(s):
    return PolicyOutput(waypoints=np.stack([s, np.zeros_like(s)], 1), dt=0.1)


def _late_go(delay=2.5, dist=12.0):
    """What Edge plans at a green light from a standstill: nothing for a while, then 12 m."""
    tau = np.clip(T - delay, 0, None)
    return _plan(0.5 * (2 * dist / (T[-1] - delay) ** 2) * tau**2)


def _stay():
    return _plan(np.full_like(T, 0.01))


def test_late_go_plan_from_rest_launches():
    c = Controller()
    ctrl = c(_late_go(), 0.0)
    assert c.debug["branch"] == "launch"
    assert ctrl.throttle >= 0.4 and ctrl.brake == 0.0


def test_stay_plan_holds_brake():
    c = Controller()
    ctrl = c(_stay(), 0.0)
    assert c.debug["branch"] == "hold" and ctrl.throttle == 0.0 and ctrl.brake >= 0.6


def test_plan_to_stop_while_moving_brakes():
    v = 8.0
    s = np.minimum(v * T - 0.5 * 4.0 * T**2, v**2 / 8.0)      # 4 m/s^2 stop within 8 m
    ctrl = Controller()(_plan(s), v)
    assert ctrl.brake > 0.2 and ctrl.throttle == 0.0


def test_invalid_plan_brakes():
    assert Controller()(_plan(np.full_like(T, np.nan)), 3.0).brake == 1.0


class _DeadBandCar:
    """Point-mass car with a throttle dead band and an engagement delay from rest, like CARLA's Tesla
    (measured: never moves at throttle 0.1, starts after ~2.2 s at 0.2, ~0.9 s at 0.4)."""

    def __init__(self, dt=0.1):
        self.dt, self.v, self.x, self.engaged = dt, 0.0, 0.0, 0.0

    def step(self, ctrl):
        if self.v < 0.05:
            self.engaged = self.engaged + self.dt * max(0.0, ctrl.throttle - 0.15) / 0.25 if ctrl.throttle > 0.15 else 0.0
            if self.engaged < 0.8:          # still pulling away: no motion yet
                return
        acc = 6.0 * max(0.0, ctrl.throttle - 0.12) - 9.0 * ctrl.brake - 0.05 * self.v
        self.v = max(0.0, self.v + acc * self.dt)
        self.x += self.v * self.dt


def _edge_like_policy(v):
    """Late start while stationary; once rolling, keep going with mild acceleration."""
    if v < 0.5:
        return _late_go()
    return _plan(v * T + 0.5 * 0.8 * T**2)


def test_closed_loop_pulls_away_from_rest_within_3s():
    car, c = _DeadBandCar(), Controller()
    for _ in range(30):
        car.step(c(_edge_like_policy(car.v), car.v))
    assert car.v > 2.0, f"speed after 3 s: {car.v:.2f} m/s"


def test_closed_loop_stay_plan_never_moves():
    car, c = _DeadBandCar(), Controller()
    for _ in range(50):
        car.step(c(_stay(), car.v))
    assert car.x == 0.0


def _late_stop(v, d, cruise_s=1.0):
    """Edge-style stop plan: keep speed for `cruise_s`, then brake evenly towards rest `d` metres ahead,
    but never harder than a stop taking 4 s. Like Edge's red-light plans, it does not sharpen as the line
    nears; it just stops further on, so tracking only the plan's first second runs the line."""
    a = min(v * v / (2 * max(d - v * cruise_s, 0.1)), max(v / 4.0, 0.5))
    tb = np.clip(T - cruise_s, 0.0, v / a)          # braking time, frozen once the plan is at rest
    return _plan(v * np.minimum(T, cruise_s) + v * tb - 0.5 * a * tb**2)


def test_closed_loop_late_braking_plans_stop_near_the_line():
    car, c = _DeadBandCar(), Controller()
    car.v, line = 8.0, 30.0
    for _ in range(150):
        car.step(c(_late_stop(car.v, max(line - car.x, 0.0)) if car.v > 0.05 else _stay(), car.v))
    # the plan itself slides ~1 m at low speed (4 s stop cap); first-second tracking alone overruns by ~50 m
    assert car.v < 0.05 and car.x <= line + 1.5, f"stopped={car.v < 0.05} at x={car.x:.1f} m (line {line} m)"


def _late_recentre(offset, heading, v):
    """Edge-style lateral plan: carry on as now for a while, then ease back to the lane centre by 6.4 s.
    `offset` (m, left +) and `heading` (rad, left +) are the car's error relative to the lane. In the ego
    frame the lane centre is y = -offset - heading * x; "carry on" is y = 0."""
    x = v * T
    u = np.clip((T - 0.8) / 2.0, 0.0, 1.0)
    w = u * u * (3 - 2 * u)      # late, like Edge in CARLA: ~3% of the correction at 1 s, ~50% at 2 s
    return PolicyOutput(waypoints=np.stack([x, w * (-offset - heading * x)], 1), dt=0.1)


def _lateral_run(controller, v=17.0, heading=math.radians(2.5), steps=60, wheelbase=2.8, max_steer=1.22):
    """Kinematic bicycle at constant speed; returns the largest |offset| from the lane centre."""
    controller.wheelbase, controller.max_steer = wheelbase, max_steer
    y, yaw, worst = 0.0, heading, 0.0
    for _ in range(steps):
        ctrl = controller(_late_recentre(y, yaw, v), v)
        delta = ctrl.steer * max_steer
        yaw += v / wheelbase * math.tan(delta) * 0.1
        y += v * math.sin(yaw) * 0.1
        worst = max(worst, abs(y))
    return worst, abs(y)


@pytest.mark.parametrize("v", [8.0, 17.0])
def test_closed_loop_late_lateral_correction_recentres(v):
    # Regression: with a fixed 15 m lookahead, a 2.5 deg heading error at 17 m/s drifted 3.8 m (CARLA:
    # into a divider). The time-based lookahead sees Edge's late correction and recentres.
    worst, final = _lateral_run(Controller(), v=v)
    assert worst < 1.0 and final < 0.3, f"worst offset {worst:.2f} m, final {final:.2f} m"
