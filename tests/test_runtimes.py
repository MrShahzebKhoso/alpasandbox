import os
import sys

import numpy as np
import pytest

from alpasandbox.backends.builtin import BuiltinBackend
from alpasandbox.engine import Episode
from alpasandbox.models import make_policy
from alpasandbox.models.runtimes import WorkerError, windows_to_wsl_path, wsl_command
from alpasandbox.scenario import load_scenario


@pytest.fixture(scope="module")
def obs():
    from alpasandbox.models.fixtures import OraclePolicy
    be = BuiltinBackend(image_size=(96, 54))
    ep = Episode(be, OraclePolicy())
    ep.reset(load_scenario("straight_cruise"))
    for _ in range(5):
        ep.step()
    return ep.observation(), ep.observation(vqa=True)


def test_subprocess_worker_roundtrip(obs):
    logs = []
    p = make_policy("fake_model:EchoModel", "subprocess", {"candidates": 2},
                    {"python": sys.executable, "env": {"PYTHONPATH": os.environ["PYTHONPATH"]}}, log=logs.append)
    try:
        assert p.info.name == "echo" and p.info.supports_vqa
        out = p.act(obs[0])
        assert out.waypoints.shape == (64, 2) and len(out.candidates) == 2
        assert out.reasoning.startswith("22 frames")
        q = p.query(obs[1], "hello?")
        assert q.answer == "hello? | 24 frames"
        p.reset()
        assert any("[worker] ready" in line for line in logs)
    finally:
        p.close()
    assert p.process.poll() is not None


def test_subprocess_worker_reports_startup_errors():
    with pytest.raises(WorkerError, match="ModuleNotFoundError|AttributeError"):
        make_policy("no_such_module_xyz:Nope", "subprocess", {}, {"python": sys.executable}, log=lambda _l: None)


def test_wsl_command_is_quoted_and_complete():
    cmd = wsl_command("edge", {"num_traj_sets": 3}, distro="Ubuntu-24.04", workdir="~/robotics/alpamayo 2",
                      venv=".venv", pythonpath="/mnt/c/Users/me/AppData/Local/alpasandbox/worker-src/0.2.0",
                      env={"HF_HUB_OFFLINE": "1"})
    assert cmd[:5] == ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "bash"]
    script = cmd[-1]
    assert 'cd "$HOME"/\'robotics/alpamayo 2\'' in script
    assert ". .venv/bin/activate" in script
    assert "export HF_HUB_OFFLINE=1" in script
    assert script.splitlines()[-1].startswith("exec python -u -m alpasandbox.worker --model edge --settings-b64 ")


def test_windows_path_conversion():
    assert windows_to_wsl_path(r"C:\Users\me\x") == "/mnt/c/Users/me/x"


def test_wsl_runtime_refuses_off_windows():
    if os.name == "nt":
        pytest.skip("windows")
    with pytest.raises(RuntimeError, match="from Windows"):
        make_policy("edge", "wsl")


def test_http_model_server_roundtrip(obs):
    from fastapi.testclient import TestClient

    from alpasandbox.models.protocol import output_from_dict
    from alpasandbox.server import create_app
    from alpasandbox.wire import encode_observation
    sys.path.insert(0, os.path.dirname(__file__))
    from fake_model import EchoModel

    c = TestClient(create_app(EchoModel(), token="s3cret"))
    assert c.get("/healthz").json()["ok"]
    assert c.get("/v1/info").status_code == 401
    h = {"Authorization": "Bearer s3cret"}
    assert c.get("/v1/info", headers=h).json()["supports_vqa"] is True
    out = output_from_dict(c.post("/v1/drive", json=encode_observation(obs[0]), headers=h).json())
    assert np.asarray(out.waypoints).shape == (64, 2) and len(out.candidates) == 2
    r = c.post("/v1/query", json={"observation": encode_observation(obs[1]), "question": "why?"}, headers=h)
    assert r.json()["answer"] == "why? | 24 frames"
