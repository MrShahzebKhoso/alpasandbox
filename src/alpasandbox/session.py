"""Interactive session behind the web app: one simulation thread, commands in, events out.

Everything that touches the model or the simulator runs on the session thread, so a question
asked mid-drive is answered between two ticks (the sim waits, exactly like synchronous mode).
"""
from __future__ import annotations

import base64
import inspect
import io
import json
import math
import queue
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable

import numpy as np
from PIL import Image

from . import __repo_url__, __version__
from .backends.base import Backend
from .engine import DISPLAY_CAMERAS, Episode, Tick
from .models.base import Policy
from .rig import CAMERA_NAMES
from .scenario import describe_scenarios, load_scenario

Emit = Callable[[dict], None]

NAV_PRESETS = ["Continue straight.", "Turn left at the next intersection.", "Turn right in 30m.",
               "Change to the left lane.", "Slow down and pull over to the right.", "Stop at the next safe spot."]
QUESTION_PRESETS = ["What should the ego vehicle do next, and why?", "Describe the scene around the car.",
                    "Are there any pedestrians or cyclists nearby?", "Is it safe to change to the left lane?",
                    "What is the most important hazard right now?"]


def _jpeg_uri(frame: np.ndarray, quality: int = 74) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _to_ego(e, x: float, y: float) -> list[float]:
    c, s = math.cos(e.yaw), math.sin(e.yaw)
    return [round(c * (x - e.x) + s * (y - e.y), 2), round(-s * (x - e.x) + c * (y - e.y), 2)]


def _round_pts(a) -> list[list[float]]:
    return np.asarray(a, float)[:, :2].round(2).tolist() if len(a) else []


class Session:
    def __init__(self, backend_factory: Callable[[], Backend], policy_factory: Callable[[], Policy],
                 scenario: str, emit: Emit, *, realtime: bool = True, autoplay: bool = False,
                 stop_on_incident: bool = True, runtime_label: str = "local"):
        self.backend_factory, self.policy_factory = backend_factory, policy_factory
        self.scenario_name, self.emit_fn = scenario, emit
        self.realtime, self.autoplay, self.stop_on_incident = realtime, autoplay, stop_on_incident
        self.runtime_label = runtime_label
        self.commands: queue.Queue = queue.Queue()
        self.state, self.error = "starting", ""
        self.backend: Backend | None = None
        self.policy: Policy | None = None
        self.episode: Episode | None = None
        self.last_frame: dict | None = None
        self.chat: list[dict] = []
        self.reasoning_log: deque = deque(maxlen=200)
        self.logs: deque = deque(maxlen=400)
        self.trace: list[dict] = []
        self.summary: dict | None = None
        self.scenarios = describe_scenarios()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._main, name="alpasandbox-session", daemon=True)

    # ---------------------------------------------------------------- public API (any thread)
    def start(self) -> None:
        self.thread.start()

    def command(self, cmd: str, **kw) -> None:
        self.commands.put((cmd, kw))

    def shutdown(self) -> None:
        self._stop.set()
        self.commands.put(("quit", {}))
        self.thread.join(timeout=15)

    def log(self, line: str, level: str = "info") -> None:
        entry = {"type": "log", "level": level, "message": line, "time": time.time()}
        self.logs.append(entry)
        self.emit_fn(entry)
        print(line, flush=True)

    def info(self) -> dict:
        p = self.policy.info.to_dict() if self.policy else None
        return {
            "type": "hello", "version": __version__, "repo_url": __repo_url__, "state": self.state, "error": self.error,
            "model": p, "runtime": self.runtime_label,
            "backend": getattr(self.backend, "name", None), "scenario": self.scenario_name,
            "scenarios": self.scenarios, "nav_presets": NAV_PRESETS, "question_presets": QUESTION_PRESETS,
            "camera_names": {str(k): v for k, v in CAMERA_NAMES.items()}, "display_cameras": list(DISPLAY_CAMERAS),
            "realtime": self.realtime, "nav_text": self.episode.nav_text if self.episode else "",
            "chat": self.chat[-50:], "reasoning_log": list(self.reasoning_log)[-60:],
            "logs": list(self.logs)[-120:], "summary": self.summary,
            "drive": [{"t": r["t"], "v": round(r["v"] * 3.6, 1), "brake": r["brake"], "reason": bool(r["reasoning"]),
                       "inc": r["collision"] or r["off_road"]} for r in self.trace],
            "duration": self.episode.scenario.duration if self.episode and self.episode.scenario else 0,
        }

    def trace_jsonl(self) -> str:
        return "".join(json.dumps(r) + "\n" for r in self.trace)

    # ---------------------------------------------------------------- session thread
    def _set_state(self, state: str, error: str = "") -> None:
        self.state, self.error = state, error
        self.emit_fn({"type": "state", "state": state, "error": error,
                      "step": self.episode.step_count if self.episode else 0})

    def _main(self) -> None:
        try:
            self._set_state("loading")
            self.log("Starting simulator backend...")
            self.backend = self.backend_factory()
            self.log(f"Backend ready: {self.backend.name}. Loading model (first load can take minutes)...")
            t0 = time.perf_counter()
            takes_log = "log" in inspect.signature(self.policy_factory).parameters
            self.policy = self.policy_factory(log=self.log) if takes_log else self.policy_factory()
            info = self.policy.info
            self.log(f"Model ready: {info.display_name} in {time.perf_counter() - t0:.1f}s"
                     f"{' (VQA enabled)' if info.supports_vqa else ''}")
            self.episode = Episode(self.backend, self.policy)
            self.emit_fn(self.info())
            self._reset(self.scenario_name)
        except Exception as exc:
            traceback.print_exc()
            self.log(f"{type(exc).__name__}: {exc}", "error")
            self._set_state("error", f"{type(exc).__name__}: {exc}")
            self.emit_fn(self.info())
            self._idle_until_quit()
            return
        try:
            self._loop()
        finally:
            for obj in (self.policy, self.backend):
                try:
                    obj and obj.close()
                except Exception:
                    pass

    def _idle_until_quit(self) -> None:
        while not self._stop.is_set():
            try:
                cmd, _ = self.commands.get(timeout=0.5)
            except queue.Empty:
                continue
            if cmd == "quit":
                return

    def _loop(self) -> None:
        while not self._stop.is_set():
            running = self.state == "running"
            try:
                cmd, kw = self.commands.get(block=not running, timeout=None if not running else 0)
                if cmd == "quit":
                    return
                self._handle(cmd, kw)
                continue          # drain all pending commands before the next tick
            except queue.Empty:
                pass
            if self.state == "running":
                self._tick()

    def _handle(self, cmd: str, kw: dict) -> None:
        try:
            if cmd == "play":
                if self.state in ("finished",):
                    self._reset(self.scenario_name)
                if self.state in ("ready", "paused"):
                    self._set_state("running")
            elif cmd == "pause":
                if self.state == "running":
                    self._set_state("paused")
            elif cmd == "step":
                if self.state in ("ready", "paused"):
                    self._tick()
                    if self.state == "running":
                        self._set_state("paused")
            elif cmd == "reset":
                self._reset(kw.get("scenario") or self.scenario_name, autoplay=kw.get("autoplay"))
            elif cmd == "nav":
                text = str(kw.get("text", "")).strip()[:300]
                if text and self.episode:
                    changed = text != self.episode.nav_text
                    self.episode.nav_text = text
                    self.emit_fn({"type": "nav", "nav_text": text, "changed": changed})
                    if changed:
                        self.log(f"Navigation command set: “{text}”")
            elif cmd == "ask":
                self._ask(str(kw.get("question", "")).strip()[:1000], kw.get("id"))
            elif cmd == "signal":
                if not hasattr(self.backend, "set_signal"):
                    raise RuntimeError("this simulator has no traffic lights")
                state = kw.get("state") or None
                result = self.backend.set_signal(state)
                self.log(f"Traffic light {'handed back to CARLA' if result == 'released' else 'held at ' + result.lower()}")
                self.emit_fn({"type": "toast", "level": "info",
                              "message": "Traffic light back on its normal cycle" if result == "released"
                              else f"Traffic light set to {result.lower()}. The model sees it only through the cameras."})
            elif cmd == "realtime":
                self.realtime = bool(kw.get("value", True))
                self.emit_fn({"type": "settings", "realtime": self.realtime})
            elif cmd == "stop_on_incident":
                self.stop_on_incident = bool(kw.get("value", True))
            else:
                self.log(f"Unknown command: {cmd}", "warn")
        except Exception as exc:
            traceback.print_exc()
            self.log(f"{cmd} failed: {type(exc).__name__}: {exc}", "error")
            self.emit_fn({"type": "toast", "level": "error", "message": f"{cmd} failed: {exc}"})

    def _reset(self, name: str, autoplay: bool | None = None) -> None:
        assert self.episode
        sc = load_scenario(name)
        self.scenario_name, self.trace, self.summary = name, [], None
        self.reasoning_log.clear()
        self.log(f"Scenario “{name}”: {sc.description}")
        res = self.episode.reset(sc)
        for what in getattr(self.backend, "unplaced", []):
            self.log(f"Could not place the scenario's {what}: something is in the way at that spot in this map", "warn")
        if getattr(self.backend, "traffic", 0):
            self.log(f"Traffic: {getattr(self.backend, 'traffic_spawned', 0)} autopilot vehicles "
                     f"(requested {self.backend.traffic}; at most one per free spawn point)")
        self.last_frame = self._frame_message(None, res)
        self.emit_fn({"type": "episode", "scenario": name, "nav_text": self.episode.nav_text,
                      "duration": sc.duration})
        self.emit_fn(self.last_frame)
        play = self.autoplay if autoplay is None else autoplay
        self._set_state("running" if play else "ready")

    def _tick(self) -> None:
        ep = self.episode
        assert ep
        t0 = time.perf_counter()
        try:
            tick = ep.step()
        except Exception as exc:
            traceback.print_exc()
            self.log(f"Model/simulator error: {type(exc).__name__}: {exc}", "error")
            self._set_state("paused", f"{type(exc).__name__}: {exc}")
            self.emit_fn({"type": "toast", "level": "error", "message": f"Step failed: {exc}"})
            return
        msg = self._frame_message(tick, tick.result)
        self.last_frame = msg
        self._record(tick)
        self.emit_fn(msg)
        res = tick.result
        incident = res.collision or res.off_road
        if incident:
            what = "Collision" + (f" with {res.privileged.get('collided_with')}" if res.privileged.get("collided_with")
                                  else "") if res.collision else "Left the road"
            self.log(f"{what} at t={res.t:.1f}s", "warn")
        if ep.done or (incident and self.stop_on_incident):
            status = "completed" if ep.done and not incident else ("collision" if res.collision else "off_road")
            self.summary = {"scenario": self.scenario_name, "model": self.policy.info.name if self.policy else "",
                            "status": status, "duration_s": round(res.t, 1), **tick.metrics}
            self.emit_fn({"type": "summary", "summary": self.summary})
            self.log(f"Episode finished: {status}. {tick.metrics['distance_m']} m driven, "
                     f"{tick.metrics['incidents']} incident(s).", "info" if status == "completed" else "warn")
            self._set_state("finished")
            return
        if self.realtime and self.backend:
            time.sleep(max(0.0, self.backend.dt - (time.perf_counter() - t0)))

    def _record(self, tick: Tick) -> None:
        r, o, c = tick.result, tick.output, tick.control
        self.trace.append({
            "step": tick.step, "t": round(r.t, 2), "x": round(r.ego.x, 3), "y": round(r.ego.y, 3),
            "yaw": round(r.ego.yaw, 4), "v": round(r.ego.v, 3), "lane_offset": round(r.lane_offset, 3),
            "collision": r.collision, "off_road": r.off_road, "steer": round(c.steer, 4),
            "throttle": round(c.throttle, 3), "brake": round(c.brake, 3), "nav_text": tick.nav_text,
            "latency_ms": round(o.latency_ms, 1), "reasoning": o.reasoning,
            "waypoints": np.asarray(o.waypoints, float).round(3).tolist(), "meta": o.meta,
            "signal": (r.privileged or {}).get("traffic_light"),
            "collided_with": (r.privileged or {}).get("collided_with", ""), "debug": tick.debug})
        if o.reasoning and (not self.reasoning_log or self.reasoning_log[-1]["text"] != o.reasoning):
            entry = {"t": round(r.t, 1), "step": tick.step, "text": o.reasoning}
            self.reasoning_log.append(entry)

    def _ask(self, question: str, qid) -> None:
        if not question:
            return
        entry = {"id": qid or f"q{len(self.chat) + 1}", "question": question, "t": None, "answer": None,
                 "latency_ms": None, "error": None}
        self.chat.append(entry)
        if not (self.policy and self.policy.info.supports_vqa):
            entry["error"] = "This model does not answer free-form questions."
            self.emit_fn({"type": "answer", **entry})
            return
        assert self.episode and self.episode.result
        entry["t"] = round(self.episode.result.t, 1)
        self.emit_fn({"type": "thinking", "id": entry["id"], "question": question})
        try:
            res = self.episode.ask(question)
            entry["answer"], entry["latency_ms"] = res.answer, round(res.latency_ms)
        except Exception as exc:
            traceback.print_exc()
            entry["error"] = f"{type(exc).__name__}: {exc}"
        self.emit_fn({"type": "answer", **entry})

    # ---------------------------------------------------------------- messages
    def _frame_message(self, tick: Tick | None, res) -> dict:
        e = res.ego
        cams = {str(c): _jpeg_uri(img) for c, img in res.images.items() if c in DISPLAY_CAMERAS}
        p = res.privileged or {}
        actors = [{"kind": a["kind"], "pos": _to_ego(e, a["x"], a["y"]), "yaw": round(a["yaw"] - e.yaw, 3),
                   "size": [round(float(s), 2) for s in a["size"][:2]]} for a in p.get("actors", [])]
        lines = []
        for ln in p.get("lines", []):
            pts = [_to_ego(e, x, y) for x, y in ln["pts"]]
            if ln["kind"] in ("surface", "crosswalk"):     # polygons: keep whole, the canvas clips them
                if not any(-40 < q[0] < 110 and abs(q[1]) < 60 for q in pts):
                    continue
            else:
                pts = [q for q in pts if -40 < q[0] < 110 and abs(q[1]) < 60]
            if len(pts) > 1:
                lines.append({"kind": ln["kind"], "pts": pts})
        ep = self.episode
        trail = ep.eh.arrays(20)[0][:, :2].round(2).tolist() if ep else []
        msg = {"type": "frame", "t": round(res.t, 2), "step": ep.step_count if ep else 0,
               "duration": ep.scenario.duration if ep and ep.scenario else 0,
               "speed": round(e.v, 2), "cams": cams, "actors": actors, "lines": lines, "trail": trail,
               "collision": res.collision, "off_road": res.off_road, "lane_offset": round(res.lane_offset, 2),
               "nav_text": ep.nav_text if ep else "", "signal": p.get("traffic_light")}
        if tick is not None:
            o, c = tick.output, tick.control
            msg.update(plan=_round_pts(o.waypoints), candidates=[_round_pts(cand) for cand in o.candidates],
                       reasoning=o.reasoning, nominal=bool(o.meta.get("nominal", not o.reasoning)),
                       latency_ms=round(o.latency_ms, 1), control={"steer": round(c.steer, 3),
                       "throttle": round(c.throttle, 3), "brake": round(c.brake, 3)},
                       metrics=tick.metrics, meta={k: v for k, v in o.meta.items() if k in
                                                   ("meta_action", "num_candidates", "model_latency_ms")})
        else:
            msg.update(plan=[], candidates=[], reasoning="", nominal=True, latency_ms=None,
                       control={"steer": 0, "throttle": 0, "brake": 0}, metrics=None, meta={})
        return msg
