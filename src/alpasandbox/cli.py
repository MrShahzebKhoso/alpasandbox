"""Command line: `alpasandbox app | run | serve-model | smoke | doctor | models | scenarios | init`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import STARTER, load_config, parse_value


# ------------------------------------------------------------------------------------------ options
def _add_model_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("model")
    g.add_argument("--model", "-m", help="edge | super | oracle | cruise | plugin name | module:Class")
    g.add_argument("--runtime", "-r", choices=["local", "subprocess", "wsl", "remote"], help="where the model runs")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="model setting, repeatable (e.g. --set num_traj_sets=3)")
    g.add_argument("--remote-url", help="remote runtime: http://host:8100")
    g.add_argument("--wsl-distro", help="wsl runtime: distro name, e.g. Ubuntu-24.04")
    g.add_argument("--wsl-workdir", help="wsl runtime: directory to cd into (e.g. ~/alpamayo2-edge)")
    g.add_argument("--wsl-venv", help="wsl runtime: venv path, relative to workdir or absolute")
    g.add_argument("--worker-python", help="subprocess runtime: python executable of the model's env")


def _add_sim_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("simulator")
    g.add_argument("--backend", "-b", choices=["builtin", "carla"])
    g.add_argument("--image-size", help="WxH per camera, e.g. 640x360")
    g.add_argument("--carla-host")
    g.add_argument("--carla-port", type=int)
    g.add_argument("--carla-town", help='"scenario" (default), "current", or a map such as Town10HD_Opt')
    g.add_argument("--traffic", type=int, help="CARLA: autopilot vehicles to spawn (named towns)")
    g.add_argument("--weather", help="CARLA weather preset, e.g. ClearNoon, CloudyNoon, WetNoon")
    g.add_argument("--scenarios-dir", help="extra directory of scenario YAML files")


def _common(sp: argparse.ArgumentParser, sim: bool = True) -> None:
    sp.add_argument("--config", "-c", help="path to alpasandbox.toml (default: ./alpasandbox.toml)")
    _add_model_opts(sp)
    if sim:
        _add_sim_opts(sp)


def _resolve(a: argparse.Namespace) -> dict:
    cfg, path = load_config(getattr(a, "config", None))
    cfg["_path"] = str(path) if path else None
    m, rt, sim, carla = cfg["model"], cfg["runtime"], cfg["sim"], cfg["carla"]
    if getattr(a, "model", None) and a.model != m["name"]:
        m["settings"] = {}      # [model.settings] in the file belong to the file's model, not a --model override
    for attr, (sect, key) in {"model": (m, "name"), "runtime": (m, "runtime"), "backend": (sim, "backend"),
                              "image_size": (sim, "image_size"), "scenarios_dir": (sim, "scenarios_dir"),
                              "carla_host": (carla, "host"), "carla_port": (carla, "port"),
                              "carla_town": (carla, "town"), "traffic": (carla, "traffic"), "weather": (carla, "weather"),
                              "remote_url": (rt["remote"], "url"), "wsl_distro": (rt["wsl"], "distro"),
                              "wsl_workdir": (rt["wsl"], "workdir"), "wsl_venv": (rt["wsl"], "venv"),
                              "worker_python": (rt["subprocess"], "python")}.items():
        v = getattr(a, attr, None)
        if v is not None:
            sect[key] = v
    for kv in getattr(a, "set", []) or []:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        m.setdefault("settings", {})[k.strip()] = parse_value(v.strip())
    if m["name"].startswith("remote:") and "://" in m["name"]:      # 0.1 compatibility
        rt["remote"]["url"], m["name"], m["runtime"] = m["name"][7:], "edge", "remote"
    elif m["name"].startswith("wsl:"):
        rt["wsl"]["distro"], m["name"], m["runtime"] = m["name"][4:] or rt["wsl"]["distro"], "edge", "wsl"
    if sim.get("scenarios_dir"):
        from . import scenario
        scenario.SEARCH_DIRS.insert(0, Path(sim["scenarios_dir"]).expanduser())
    return cfg


def _image_size(cfg: dict) -> tuple[int, int]:
    w, h = (int(v) for v in str(cfg["sim"]["image_size"]).lower().split("x"))
    return w, h


def backend_factory(cfg: dict):
    def make():
        from .backends import make_backend
        size = _image_size(cfg)
        if cfg["sim"]["backend"] == "carla":
            c = cfg["carla"]
            return make_backend("carla", host=c["host"], port=int(c["port"]), town=c["town"], image_size=size,
                                traffic=int(c["traffic"]), spawn_index=int(c["spawn_index"]),
                                ego_blueprint=c["ego_blueprint"], follow_spectator=bool(c["follow_spectator"]),
                                tm_port=int(c["tm_port"]), weather=str(c["weather"]),
                                start_speed=None if float(c["start_speed"]) < 0 else float(c["start_speed"]))
        return make_backend(cfg["sim"]["backend"], image_size=size)
    return make


def policy_factory(cfg: dict):
    def make(log=None):
        from .models import make_policy
        m = cfg["model"]
        return make_policy(m["name"], m["runtime"], m.get("settings", {}),
                           cfg["runtime"].get(m["runtime"], {}), log=log)
    return make


def _runtime_label(cfg: dict) -> str:
    rt = cfg["model"]["runtime"]
    if rt == "wsl":
        return f"WSL {cfg['runtime']['wsl'].get('distro') or '(default distro)'}"
    if rt == "remote":
        return f"remote {cfg['runtime']['remote'].get('url', '')}"
    return {"local": "local GPU" if cfg["model"]["name"] in ("edge", "super") else "local",
            "subprocess": "separate Python"}.get(rt, rt)


def parse_signal_at(text: str) -> tuple[str, float]:
    """"red@25" -> ("Red", 25.0)."""
    state, sep, dist = text.partition("@")
    try:
        metres = float(dist)
    except ValueError:
        metres = -1.0
    if not sep or state.lower() not in ("red", "yellow", "green") or metres <= 0:
        raise SystemExit(f"--signal-at expects STATE@METRES with STATE red, yellow or green, e.g. red@25 (got {text!r})")
    return state.capitalize(), metres


class SignalTrigger:
    """After each step: once the ego is within `metres` of its light's stop line, set the light (once)."""

    def __init__(self, state: str, metres: float):
        self.state, self.metres, self.fired = state, metres, False

    def __call__(self, backend, tick) -> None:
        d = (tick.debug.get("backend") or {}).get("stop_line_m")
        if not self.fired and d is not None and 0 <= d <= self.metres:
            self.fired = True
            print(f"t={tick.result.t:.1f} s, {d:.1f} m to the stop line: light set to "
                  f"{backend.set_signal(self.state)}", file=sys.stderr)


# ------------------------------------------------------------------------------------------ commands
def cmd_app(a, cfg):
    import uvicorn

    from .app import create_app

    host = a.host or cfg["app"]["host"]
    port = int(a.port or cfg["app"]["port"])
    if cfg["sim"]["backend"] == "carla" and int(cfg["carla"]["tm_port"]) == port:
        raise SystemExit(f"[carla] tm_port and the app port are both {port}; change one of them")
    scenario = a.scenario or cfg["app"]["scenario"]
    realtime = cfg["app"]["realtime"] if a.realtime is None else a.realtime
    app = create_app(backend_factory(cfg), policy_factory(cfg), scenario, realtime=realtime,
                     autoplay=bool(a.autoplay or cfg["app"]["autoplay"]), runtime_label=_runtime_label(cfg))
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"
    print(f"\n  AlpaSandbox {__version__}  ->  {url}\n  model: {cfg['model']['name']} ({_runtime_label(cfg)}), "
          f"simulator: {cfg['sim']['backend']}{', config: ' + cfg['_path'] if cfg['_path'] else ''}\n")
    if cfg["app"]["open_browser"] and not a.no_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


def cmd_run(a, cfg):
    from .engine import Runner
    from .scenario import list_scenarios, load_scenario

    names = list_scenarios() if a.all else (a.scenario or [cfg["app"]["scenario"]])
    policy = policy_factory(cfg)()
    if (a.signal or a.signal_at) and cfg["sim"]["backend"] != "carla":
        raise SystemExit("--signal and --signal-at need the CARLA backend (traffic lights exist only in CARLA towns)")
    trigger_spec = parse_signal_at(a.signal_at) if a.signal_at else None
    backend = backend_factory(cfg)()
    results, tf = [], open(a.trace, "w", encoding="utf-8") if a.trace else None
    try:
        for name in names:
            def on_frame(ev, name=name):
                if tf:
                    r, o, c = ev["result"], ev["output"], ev["control"]
                    tf.write(json.dumps({"scenario": name, "t": round(r.t, 2), "x": round(r.ego.x, 3),
                                         "y": round(r.ego.y, 3), "v": round(r.ego.v, 3), "collision": r.collision,
                                         "collided_with": r.privileged.get("collided_with", ""),
                                         "off_road": r.off_road, "lane_offset": round(r.lane_offset, 3),
                                         "steer": round(c.steer, 4),
                                         "throttle": round(c.throttle, 3), "brake": round(c.brake, 3),
                                         "latency_ms": round(o.latency_ms, 1), "reasoning": o.reasoning,
                                         "waypoints": o.waypoints.round(3).tolist(), "meta": o.meta,
                                         "signal": r.privileged.get("traffic_light"),
                                         "debug": ev["tick"].debug}) + "\n")
            t0 = time.perf_counter()
            hold = (lambda b: print(f"signal held: {b.set_signal(a.signal)}", file=sys.stderr)) if a.signal else None
            trigger = SignalTrigger(*trigger_spec) if trigger_spec else None
            r = Runner(backend, policy, on_frame=on_frame, stop_on_incident=not a.keep_going,
                       after_reset=hold, max_steps=a.steps, after_step=trigger).run(load_scenario(name))
            r["wall_s"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            print(f"{name:24s} {r['status']:10s} {r['distance_m']:7.1f} m  incidents {r['incidents']}  "
                  f"lane rms {r['lane_offset_rms_m']:.2f} m  latency {r['mean_policy_latency_ms']:.0f} ms",
                  file=sys.stderr)
    finally:
        backend.close()
        policy.close()
        if tf:
            tf.close()
    out = {"alpasandbox": __version__, "model": cfg["model"]["name"], "runtime": cfg["model"]["runtime"],
           "backend": cfg["sim"]["backend"], "results": results}
    text = json.dumps(out if len(results) > 1 else results[0], indent=2)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(text)


def cmd_serve(a, cfg):
    import uvicorn

    from .server import create_app

    policy = policy_factory(cfg)()
    print(f"serving {policy.info.display_name} on http://{a.host}:{a.port}  (VQA: {policy.info.supports_vqa})")
    uvicorn.run(create_app(policy), host=a.host, port=a.port, log_level="info")


def cmd_smoke(a, cfg):
    import numpy as np

    from .engine import Episode
    from .models.fixtures import CruisePolicy
    from .scenario import load_scenario

    t0 = time.perf_counter()
    policy = policy_factory(cfg)()
    info = policy.info
    print(f"model: {info.display_name} ready in {time.perf_counter() - t0:.1f}s | drive: "
          f"{info.drive_spec.num_images} images, {info.drive_spec.history_steps} poses | VQA: {info.supports_vqa}")
    backend = backend_factory(cfg)()
    ep = Episode(backend, policy)
    ep.reset(load_scenario(a.scenario or "pedestrian_crossing"))
    warm = CruisePolicy()
    for _ in range(22):   # fill the 2 s history buffers without calling the real model
        res = backend.step(ep.controller(warm.act(None), ep.result.ego.v))
        ep.result = res
        ep.fb.push(res.images)
        ep.eh.push(res.ego)
    obs = ep.observation()
    print(f"observation: {sum(len(v) for v in obs.frames.values())} frames of {obs.frames[1][0].shape}, "
          f"ego history {obs.ego_xyz.shape}, nav '{obs.nav_text}'")
    for i in range(a.calls):
        out = policy.act(obs)
        wp = np.asarray(out.waypoints)
        print(f"plan {i + 1}: {out.latency_ms:7.0f} ms | {wp.shape[0]} waypoints, 1 s -> {wp[min(9, len(wp) - 1)].round(2)}, "
              f"end {wp[-1].round(2)} | {len(out.candidates)} alternatives")
        print(f"         reasoning: {out.reasoning or '(routine driving, none given)'}")
    if info.supports_vqa and a.question:
        q = policy.query(ep.observation(vqa=True), a.question)
        print(f"Q: {a.question}\nA ({q.latency_ms:.0f} ms): {q.answer}")
    backend.close()
    policy.close()


def cmd_models(a, cfg):
    from .models import available_models
    for r in available_models():
        mark = "installed" if r["installed_here"] else "not installed here"
        print(f"{r['name']:8s} {r['display_name']:26s} {mark:20s} {r['gpu']:14s} {r['summary']}")
        if not r["installed_here"] and r["install"]:
            print(f"{'':8s} install: {r['install']}")


def cmd_scenarios(a, cfg):
    from .scenario import describe_scenarios
    for s in describe_scenarios():
        print(f"{s['name']:22s} {s['duration']:4.0f}s  {s['description']}")


def cmd_init(a, cfg):
    p = Path(a.path)
    if p.exists() and not a.force:
        raise SystemExit(f"{p} exists; use --force to overwrite")
    p.write_text(STARTER, encoding="utf-8")
    print(f"wrote {p}. Edit it, then run `alpasandbox doctor` and `alpasandbox app`.")


def cmd_doctor(a, cfg):
    from .doctor import run_doctor
    raise SystemExit(run_doctor(cfg))


# ------------------------------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("alpasandbox", description="Interactive closed-loop sandbox for driving VLA models.")
    p.add_argument("--version", action="version", version=f"alpasandbox {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("app", aliases=["demo"], help="launch the web app")
    _common(s)
    s.add_argument("-s", "--scenario")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--autoplay", action="store_true", help="start driving immediately")
    s.add_argument("--realtime", dest="realtime", action="store_true", default=None)
    s.add_argument("--fast", dest="realtime", action="store_false", help="do not pace at real time")
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(fn=cmd_app)

    s = sub.add_parser("run", help="headless closed-loop run(s); prints metrics JSON")
    _common(s)
    s.add_argument("-s", "--scenario", action="append", help="repeatable")
    s.add_argument("--all", action="store_true", help="run every scenario (a mini benchmark)")
    s.add_argument("--keep-going", action="store_true", help="do not stop an episode at the first incident")
    s.add_argument("--trace", help="write one JSON line per step")
    s.add_argument("--signal", choices=["red", "yellow", "green"],
                   help="CARLA towns: force and hold the ego's traffic light right after reset")
    s.add_argument("--steps", type=int, help="stop each episode after this many steps")
    s.add_argument("--signal-at", metavar="STATE@METRES",
                   help="CARLA towns: switch the ego's light to STATE once the front bumper is within METRES "
                        "of its stop line, e.g. red@25 (tests stopping on approach)")
    s.add_argument("--out", help="write the results JSON here")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("serve-model", help="serve a model over HTTP (run this on the GPU machine)")
    _common(s, sim=False)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8100)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("smoke", help="load the model, call it on a simulated observation, print timings")
    _common(s)
    s.add_argument("-s", "--scenario")
    s.add_argument("--calls", type=int, default=3)
    s.add_argument("--question", default="What should the ego vehicle do next, and why?")
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("doctor", help="check Python, GPU, model packages, CARLA, WSL and your config")
    _common(s)
    s.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("models", help="list available models")
    s.add_argument("--config", "-c")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("scenarios", help="list scenarios")
    s.add_argument("--config", "-c")
    s.add_argument("--scenarios-dir")
    s.set_defaults(fn=cmd_scenarios)

    s = sub.add_parser("init", help="write a starter alpasandbox.toml")
    s.add_argument("path", nargs="?", default="alpasandbox.toml")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_init)
    return p


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    cfg = _resolve(a)
    try:
        a.fn(a, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)


if __name__ == "__main__":
    main()
