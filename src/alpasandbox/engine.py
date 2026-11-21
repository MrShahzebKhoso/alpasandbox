"""The closed loop, one tick at a time: backend -> observation -> model -> controller -> backend.

`Episode` is shared by the headless `Runner` and the interactive web `Session`.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .backends.base import Backend
from .controller import Controller
from .core import Control, Observation, PolicyOutput, QueryResult, StepResult
from .metrics import Metrics
from .models.base import Policy
from .rig import CAMERA_NAMES, EgoHistory, FrameBuffer, build_observation
from .scenario import Scenario

DISPLAY_CAMERAS = (0, 1, 2, 3, 5, 6)


@dataclass
class Tick:
    step: int
    result: StepResult
    output: PolicyOutput
    control: Control
    metrics: dict
    nav_text: str
    debug: dict = field(default_factory=dict)   # controller + backend internals, for traces


class Episode:
    def __init__(self, backend: Backend, policy: Policy, controller: Controller | None = None):
        self.backend, self.policy = backend, policy
        self._auto_controller = controller is None
        self.controller = controller or Controller()
        info = policy.info
        specs = [info.drive_spec] + ([info.vqa_spec] if info.vqa_spec else [])
        self.cameras = sorted(set(DISPLAY_CAMERAS).union(*[s.camera_ids for s in specs]))
        missing = [c for c in self.cameras if c not in {r.cam_id for r in backend.rig}]
        if missing:
            raise ValueError(f"backend rig lacks cameras {missing} ({[CAMERA_NAMES.get(c) for c in missing]})")
        backend.set_cameras(self.cameras)
        self.horizon = max(s.horizon for s in specs) + backend.dt
        self.scenario: Scenario | None = None
        self.result: StepResult | None = None
        self.last: Tick | None = None
        self.nav_text = ""
        self.step_count = 0

    def reset(self, scenario: Scenario) -> StepResult:
        self.scenario, self.nav_text, self.step_count = scenario, scenario.nav_text, 0
        self.fb, self.eh, self.metrics = FrameBuffer(self.backend.dt, self.horizon), EgoHistory(), Metrics()
        self.result = self.backend.reset(scenario)
        if self._auto_controller:   # vehicle geometry is only known after the backend spawns the ego
            self.controller.wheelbase = float(getattr(self.backend, "wheelbase", 2.8))
            self.controller.max_steer = float(getattr(self.backend, "max_steer_rad", 0.5))
        self.fb.push(self.result.images)
        self.eh.push(self.result.ego)
        self.policy.reset()
        self.last = None
        return self.result

    @property
    def done(self) -> bool:
        return self.result is None or self.result.done

    def observation(self, vqa: bool = False) -> Observation:
        assert self.result is not None, "call reset() first"
        info = self.policy.info
        spec = info.vqa_spec if vqa else info.drive_spec
        if spec is None:
            raise NotImplementedError(f"{info.display_name} does not answer questions")
        return build_observation(self.result.t, self.fb, self.eh, spec, self.nav_text, self.result.privileged)

    def step(self) -> Tick:
        assert self.result is not None, "call reset() first"
        out = self.policy.act(self.observation())
        ctrl = self.controller(out, self.result.ego.v)
        self.result = self.backend.step(ctrl)
        self.fb.push(self.result.images)
        self.eh.push(self.result.ego)
        self.step_count += 1
        self.metrics.update(self.result, out.latency_ms, bool(out.reasoning))
        debug = {"controller": dict(getattr(self.controller, "debug", {}) or {}),
                 "backend": self.backend.debug_state()}
        self.last = Tick(self.step_count, self.result, out, ctrl, self.metrics.summary(), self.nav_text, debug)
        return self.last

    def ask(self, question: str) -> QueryResult:
        return self.policy.query(self.observation(vqa=True), question)


class Runner:
    """Headless closed loop. Returns a metrics summary dict."""

    def __init__(self, backend: Backend, policy: Policy, controller: Controller | None = None,
                 on_frame: Callable[[dict], None] | None = None, stop_on_incident: bool = True,
                 realtime: bool = False, after_reset: Callable[[Backend], None] | None = None,
                 max_steps: int | None = None, after_step: Callable[[Backend, Tick], None] | None = None):
        self.backend, self.policy, self.controller = backend, policy, controller
        self.after_reset, self.max_steps, self.after_step = after_reset, max_steps, after_step
        self.on_frame, self.stop_on_incident, self.realtime = on_frame, stop_on_incident, realtime
        self.stop_event = threading.Event()

    def run(self, scenario: Scenario) -> dict:
        ep = Episode(self.backend, self.policy, self.controller)
        res = ep.reset(scenario)
        if self.after_reset:
            self.after_reset(self.backend)
        status = "completed"
        while not ep.done and (self.max_steps is None or ep.step_count < self.max_steps):
            if self.stop_event.is_set():
                status = "stopped"
                break
            t0 = time.perf_counter()
            tick = ep.step()
            res = tick.result
            if self.after_step:
                self.after_step(self.backend, tick)
            if self.on_frame:
                self.on_frame({"result": res, "output": tick.output, "control": tick.control,
                               "metrics": tick.metrics, "scenario": scenario.name, "tick": tick})
            if self.stop_on_incident and (res.collision or res.off_road):
                status = "collision" if res.collision else "off_road"
                break
            if self.realtime:
                time.sleep(max(0.0, self.backend.dt - (time.perf_counter() - t0)))
        return {"scenario": scenario.name, "model": self.policy.info.name, "status": status,
                "duration_s": round(res.t, 1), **ep.metrics.summary()}
