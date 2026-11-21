import json

import pytest

from alpasandbox.cli import _resolve, build_parser, main


def test_cli_overrides_and_legacy_specs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["init"])
    a = build_parser().parse_args(["run", "-m", "wsl:Ubuntu-22.04", "--set", "num_traj_sets=3", "--set", "x=true"])
    cfg = _resolve(a)
    assert cfg["model"]["runtime"] == "wsl" and cfg["runtime"]["wsl"]["distro"] == "Ubuntu-22.04"
    assert cfg["model"]["settings"]["num_traj_sets"] == 3 and cfg["model"]["settings"]["x"] is True


def test_cli_run_writes_results(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)        # never pick up a developer's ./alpasandbox.toml (e.g. backend = carla)
    out = tmp_path / "r.json"
    main(["run", "-m", "oracle", "-s", "straight_cruise", "--image-size", "64x36", "--out", str(out)])
    r = json.loads(out.read_text())
    assert r["status"] == "completed" and r["model"] == "oracle"


def test_model_override_drops_file_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "alpasandbox.toml").write_text('[model]\nname = "edge"\n[model.settings]\nrevision = "abc"\n')
    cfg = _resolve(build_parser().parse_args(["app", "--model", "cruise"]))
    assert cfg["model"]["settings"] == {}
    cfg = _resolve(build_parser().parse_args(["app"]))
    assert cfg["model"]["settings"] == {"revision": "abc"}


def test_signal_needs_carla(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="CARLA"):
        main(["run", "-m", "cruise", "-b", "builtin", "-s", "straight_cruise", "--signal", "green"])


def test_trace_has_controller_debug_and_steps_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = tmp_path / "t.jsonl"
    main(["run", "-m", "cruise", "-b", "builtin", "-s", "straight_cruise", "--image-size", "64x36",
          "--steps", "5", "--trace", str(trace)])
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 5
    d = rows[-1]["debug"]["controller"]
    assert {"s_1s", "s_3s", "v_target", "stop", "branch"} <= d.keys()


def test_signal_at_parses_and_rejects_bad_input():
    from alpasandbox.cli import parse_signal_at
    assert parse_signal_at("red@25") == ("Red", 25.0)
    for bad in ("red", "blue@10", "red@x", "red@-3"):
        with pytest.raises(SystemExit):
            parse_signal_at(bad)


def test_signal_trigger_fires_once_inside_the_distance():
    from types import SimpleNamespace

    from alpasandbox.cli import SignalTrigger

    calls = []
    backend = SimpleNamespace(set_signal=lambda st: calls.append(st) or st)
    trig = SignalTrigger("Red", 25.0)
    for d in (40.0, None, 30.0, 24.0, 20.0, -1.0):
        trig(backend, SimpleNamespace(debug={"backend": {"stop_line_m": d}}, result=SimpleNamespace(t=1.0)))
    assert calls == ["Red"]
