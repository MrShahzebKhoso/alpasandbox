import numpy as np
import pytest

from alpasandbox.backends.builtin import BuiltinBackend
from alpasandbox.engine import Episode, Runner
from alpasandbox.models import make_policy
from alpasandbox.models.alpamayo_edge import edge_drive_arrays, edge_vqa_arrays
from alpasandbox.models.alpamayo_super import super_source_arrays
from alpasandbox.models.fixtures import CruisePolicy, OraclePolicy
from alpasandbox.rig import EDGE_DRIVE_SPEC, SUPER_SPEC, build_observation
from alpasandbox.scenario import describe_scenarios, list_scenarios, load_scenario
from alpasandbox.wire import decode_observation, encode_observation

SMALL = (96, 54)


def _episode(policy=None, steps=25, scenario="straight_cruise", cams=None):
    be = BuiltinBackend(image_size=SMALL)
    ep = Episode(be, policy or CruisePolicy())
    if cams:
        be.set_cameras(cams)
    ep.reset(load_scenario(scenario))
    for _ in range(steps):
        ep.step()
    return ep


def test_scenarios_load():
    assert len(list_scenarios()) >= 9
    for d in describe_scenarios():
        assert not d["description"].startswith("(invalid"), d
        load_scenario(d["name"])


def test_unknown_scenario_keys_are_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: bad\nspeed: 3\n")
    with pytest.raises(ValueError):
        load_scenario(str(p))


def test_edge_drive_contract():
    obs = _episode().observation()
    a = edge_drive_arrays(obs)
    assert a["image_frames"].shape[:3] == (22, 1, 3) and a["image_frames"].dtype == np.uint8
    assert list(dict.fromkeys(a["camera_indices"].tolist())) == [0, 1, 2, 3, 5, 6]
    assert (a["absolute_timestamps"] >= 0).all()
    front = a["absolute_timestamps"][a["camera_indices"] == 1]
    assert np.allclose(np.diff(front) / 1e6, [1.0, 0.5, 0.4, 0.1], atol=1e-3)
    assert a["ego_history_xyz"].shape == (1, 20, 3) and np.allclose(a["ego_history_xyz"][0, -1], 0)
    assert a["ego_history_xyz"][0, 0, 0] < -10        # 1.9 s ago at ~8 m/s


def test_edge_vqa_contract():
    ep = _episode(OraclePolicy())
    a = edge_vqa_arrays(ep.observation(vqa=True))
    assert a["image_frames"].shape[:3] == (24, 1, 3)
    for c in (0, 1, 2, 3, 5, 6):
        ts = a["absolute_timestamps"][a["camera_indices"] == c]
        assert np.allclose(np.diff(ts) / 1e6, 0.1, atol=1e-3)


def test_super_source_contract():
    ep = _episode(cams=range(7))
    obs = build_observation(ep.result.t, ep.fb, ep.eh, SUPER_SPEC, "go", None)
    a = super_source_arrays(obs)
    assert a["image_frames"].shape[:3] == (7, 4, 3)
    assert a["camera_indices"].tolist() == list(range(7))
    assert a["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert int(a["ego_t0"][0]) == int(a["absolute_timestamps"][:, -1].max())
    assert a["ego_t0_frame_idx"].tolist() == [3]


def test_wire_roundtrip_drops_ground_truth():
    obs = _episode(OraclePolicy()).observation()
    d = encode_observation(obs)
    assert "privileged" not in d
    back = decode_observation(d)
    assert back.frames[1][0].shape == obs.frames[1][0].shape
    assert np.allclose(back.ego_xyz, obs.ego_xyz, atol=1e-4)


@pytest.mark.parametrize("name", ["pedestrian_crossing", "lead_vehicle_braking", "cone_obstruction",
                                  "cut_in", "occluded_pedestrian", "slow_lead"])
def test_oracle_avoids_obstacles(name):
    out = Runner(BuiltinBackend(image_size=SMALL), OraclePolicy()).run(load_scenario(name))
    assert out["status"] == "completed" and out["collisions"] == 0, out


def test_cruise_hits_blocker():
    out = Runner(BuiltinBackend(image_size=SMALL), CruisePolicy()).run(load_scenario("cone_obstruction"))
    assert out["status"] == "collision"


def test_oracle_follows_curves():
    out = Runner(BuiltinBackend(image_size=SMALL), OraclePolicy()).run(load_scenario("curvy_road"))
    assert out["status"] == "completed" and out["lane_offset_rms_m"] < 0.6, out


def test_history_at_start_reflects_motion():
    be = BuiltinBackend(image_size=SMALL)
    ep = Episode(be, CruisePolicy())
    ep.reset(load_scenario("straight_cruise"))
    assert ep.observation().ego_xyz[0, 0] < -10


def test_only_needed_cameras_are_rendered():
    ep = _episode(CruisePolicy(), steps=1)
    assert sorted(ep.result.images) == [0, 1, 2, 3, 5, 6]


def test_oracle_answers_questions_from_ground_truth():
    ep = _episode(OraclePolicy(), steps=10, scenario="lead_vehicle_braking")
    ans = ep.ask("What is ahead?").answer
    assert "vehicle" in ans and "ego lane" in ans


def test_plugin_by_module_path():
    p = make_policy("fake_model:EchoModel", settings={"candidates": 1})
    out = p.act(_episode().observation())
    assert out.reasoning.startswith("22 frames") and len(out.candidates) == 1


def test_missing_model_package_has_install_hint():
    import importlib.util
    if importlib.util.find_spec("alpamayo2_edge"):
        pytest.skip("upstream installed")
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        make_policy("edge")


def test_controller_respects_max_steer():
    from alpasandbox.controller import Controller
    from alpasandbox.core import PolicyOutput
    s = np.linspace(1, 40, 64)
    left_turn = PolicyOutput(np.stack([s, 0.02 * s**2], 1))
    a = Controller(max_steer=0.5)(left_turn, 8.0).steer
    b = Controller(max_steer=1.2)(left_turn, 8.0).steer
    assert a > 0 and b > 0 and a == pytest.approx(min(1.0, b * 1.2 / 0.5), rel=1e-6)


def test_drive_spec_counts():
    assert EDGE_DRIVE_SPEC.num_images == 22 and SUPER_SPEC.num_images == 28


def test_controller_pulls_away_on_gentle_start():
    """Regression: a plan that starts from rest (12 m in 6.4 s) must give throttle, not brake."""
    from alpasandbox.controller import Controller
    from alpasandbox.core import PolicyOutput
    t = 0.1 * np.arange(1, 65)
    s = 0.5 * (2 * 12 / 6.4**2) * t**2
    c = Controller()(PolicyOutput(np.stack([s, 0 * s], 1)), 0.0)
    assert c.throttle > 0.1 and c.brake == 0.0
    stay = Controller()(PolicyOutput(np.zeros((64, 2)) + [0.01, 0.0]), 0.0)
    assert stay.brake >= 0.6 and stay.throttle == 0.0
