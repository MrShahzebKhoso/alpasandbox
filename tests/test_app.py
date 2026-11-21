import time

from fastapi.testclient import TestClient

from alpasandbox.app import create_app
from alpasandbox.backends.builtin import BuiltinBackend
from alpasandbox.models.fixtures import OraclePolicy


def _wait(ws, pred, limit=400):
    for _ in range(limit):
        m = ws.receive_json()
        if pred(m):
            return m
    raise AssertionError("message not received")


def test_web_app_drive_ask_and_nav():
    app = create_app(lambda: BuiltinBackend(image_size=(96, 54)), OraclePolicy, "lead_vehicle_braking", realtime=False)
    with TestClient(app) as c:
        for _ in range(100):
            if c.get("/api/info").json()["state"] == "ready":
                break
            time.sleep(0.05)
        assert c.get("/").status_code == 200 and c.get("/static/app.js").status_code == 200
        for path in ("/", "/static/app.js", "/static/app.css"):   # a stale app.js with a fresh page breaks the UI
            assert c.get(path).headers.get("cache-control") == "no-cache", path
        with c.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello" and hello["model"]["supports_vqa"]
            ws.send_json({"cmd": "nav", "text": "Turn right in 30m."})
            assert _wait(ws, lambda m: m["type"] == "nav")["nav_text"] == "Turn right in 30m."
            ws.send_json({"cmd": "step"})
            f = _wait(ws, lambda m: m["type"] == "frame" and m["step"] >= 1)
            assert len(f["plan"]) == 64 and set(f["cams"]) == {"0", "1", "2", "3", "5", "6"}
            assert f["nav_text"] == "Turn right in 30m."
            ws.send_json({"cmd": "ask", "question": "What is ahead?", "id": "q1"})
            a = _wait(ws, lambda m: m["type"] == "answer")
            assert a["id"] == "q1" and "vehicle" in a["answer"]
            ws.send_json({"cmd": "play"})
            s = _wait(ws, lambda m: m["type"] == "summary", limit=5000)
            assert s["summary"]["status"] == "completed"
        trace = c.get("/api/trace.jsonl").text.strip().splitlines()
        assert len(trace) >= 200


def test_web_app_surfaces_model_load_errors():
    def broken():
        raise RuntimeError("weights missing")
    app = create_app(lambda: BuiltinBackend(image_size=(96, 54)), broken, "straight_cruise")
    with TestClient(app) as c:
        for _ in range(100):
            info = c.get("/api/info").json()
            if info["state"] == "error":
                break
            time.sleep(0.05)
        assert info["state"] == "error" and "weights missing" in info["error"]
