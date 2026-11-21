"""Bird's-eye map geometry for CARLA towns (pure; no CARLA needed)."""
from alpasandbox.backends.carla_backend import _crosswalk_polygons, lane_geometry


def _seg(y0, junction=False, left="edge", right="edge"):
    xs = [0.0, 10.0, 20.0]
    return {"ctr": [(x, y0) for x in xs], "left": [(x, y0 + 1.75) for x in xs],
            "right": [(x, y0 - 1.75) for x in xs], "junction": junction, "left_kind": left, "right_kind": right}


def test_junction_lanes_get_a_road_surface():
    out = lane_geometry([_seg(0.0, junction=True)], [])
    kinds = [k for k, _ in out]
    assert kinds.count("surface") == 1 and "junction" in kinds and "edge" not in kinds
    surface = next(p for k, p in out if k == "surface")
    assert len(surface) == 6          # left edge out, right edge back: one closed polygon


def test_two_lane_road_draws_kerbs_once_and_divider_once():
    # lane A (y=0) has lane B (same direction) on its right; B's right is the kerb
    out = lane_geometry([_seg(0.0, left="edge", right="lane"), _seg(-3.5, left="lane", right="edge")], [])
    kinds = sorted(k for k, _ in out if k != "surface")
    assert kinds == ["edge", "edge", "lane"]


def test_crosswalks_split_into_closed_polygons():
    sq = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    polys = _crosswalk_polygons(sq + [(p[0] + 5, p[1]) for p in sq])
    assert len(polys) == 2 and all(len(p) == 5 for p in polys)
