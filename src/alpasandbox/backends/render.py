"""Tiny dependency-light pinhole renderer (PIL). Good enough to exercise the full
6-camera pipeline and give the viewer something to show. NOT photorealistic: use the
CARLA backend for realistic imagery."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

SKY, GROUND, ROAD, LINE = (140, 185, 235), (78, 118, 78), (72, 72, 78), (235, 235, 235)
SHOULDER = (150, 146, 136)
KIND_COLOR = {"pedestrian": (235, 160, 40), "vehicle": (200, 50, 50), "cone": (255, 110, 20)}


def cam_pose(spec, ego):
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    return (ego.x + c * spec.x - s * spec.y, ego.y + s * spec.x + c * spec.y, spec.z,
            ego.yaw + math.radians(spec.yaw_deg))


def _cam_frame(pts, pose):
    cx, cy, cz, yaw = pose
    d = np.asarray(pts, float) - np.array([cx, cy, cz])
    fwd = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
    left = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
    return np.stack([fwd, left, d[:, 2]], 1)


def project(pts, pose, f, w, h):
    c = _cam_frame(pts, pose)
    safe = np.maximum(c[:, 0], 1e-3)
    return w / 2 - f * c[:, 1] / safe, h / 2 - f * c[:, 2] / safe, c[:, 0]


NEAR = 0.2


def _clip_near(c: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a camera-frame polygon against the plane fwd = NEAR."""
    out = []
    n = len(c)
    for i in range(n):
        a, b = c[i], c[(i + 1) % n]
        ina, inb = a[0] >= NEAR, b[0] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[0]) / (b[0] - a[0])
            out.append(a + t * (b - a))
    return np.asarray(out)


def _poly(draw, pts, pose, f, w, h, fill):
    c = _cam_frame(pts, pose)
    if (c[:, 0] < NEAR).all():
        return
    if (c[:, 0] < NEAR).any():
        c = _clip_near(c)
        if len(c) < 3:
            return
    u, v = w / 2 - f * c[:, 1] / c[:, 0], h / 2 - f * c[:, 2] / c[:, 0]
    if np.abs(u).max() > 1e5 or np.abs(v).max() > 1e5:
        return
    draw.polygon(list(zip(u.tolist(), v.tolist(), strict=False)), fill=fill, outline=fill)


def _box_faces(cx, cy, yaw, L, W, H):
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(a * L / 2, b * W / 2) for a, b in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    base = [(cx + c * a - s * b, cy + s * a + c * b) for a, b in corners]
    bot = [(x, y, 0.0) for x, y in base]
    top = [(x, y, H) for x, y in base]
    return [top] + [[bot[i], bot[(i + 1) % 4], top[(i + 1) % 4], top[i]] for i in range(4)]


@lru_cache(maxsize=8)
def _background_array(w: int, h: int) -> np.ndarray:
    """Sky gradient above the horizon, grass gradient below (cached per image size)."""
    y = np.linspace(0, 1, h)[:, None]
    top, horizon = np.array([92, 142, 214]), np.array([196, 218, 238])
    near, far = np.array([64, 104, 60]), np.array([118, 146, 108])
    sky = top + (horizon - top) * (y / 0.5)
    ground = far + (near - far) * ((y - 0.5) / 0.5)
    rows = np.where(y < 0.5, sky, ground)
    return np.repeat(rows[:, None, :], w, axis=1).clip(0, 255).astype(np.uint8)


def _background(w: int, h: int) -> Image.Image:
    return Image.fromarray(_background_array(w, h))


def render_camera(spec, ego, actors, lane_y, size=(480, 270), road_edges=(-1.75, 5.25)) -> np.ndarray:
    w, h = size
    f = (w / 2) / math.tan(math.radians(spec.fov_deg) / 2)
    pose = cam_pose(spec, ego)
    img = _background(w, h)
    draw = ImageDraw.Draw(img)

    lo, hi = road_edges
    x0 = math.floor(ego.x) - 25
    xs = np.arange(x0, x0 + 140, 1.0)
    for xa, xb in zip(xs[:-1], xs[1:], strict=False):
        ya, yb = lane_y(xa), lane_y(xb)
        _poly(draw, [(xa, ya + lo - .6, 0), (xb, yb + lo - .6, 0), (xb, yb + hi + .6, 0), (xa, ya + hi + .6, 0)],
              pose, f, w, h, SHOULDER)
        _poly(draw, [(xa, ya + lo, 0), (xb, yb + lo, 0), (xb, yb + hi, 0), (xa, ya + hi, 0)],
              pose, f, w, h, ROAD)
        for off, dashed in ((lo, False), (hi, False), (1.75, True)):
            if dashed and int(xa) % 6 >= 3:
                continue
            _poly(draw, [(xa, ya + off - .08, 0), (xb, yb + off - .08, 0),
                         (xb, yb + off + .08, 0), (xa, ya + off + .08, 0)], pose, f, w, h, LINE)

    for a in sorted(actors, key=lambda a: -math.hypot(a["x"] - pose[0], a["y"] - pose[1])):
        base = KIND_COLOR.get(a["kind"], (200, 200, 200))
        for i, face in enumerate(_box_faces(a["x"], a["y"], a["yaw"], *a["size"])):
            shade = 1.0 if i == 0 else 0.7 + 0.1 * (i % 3)
            _poly(draw, face, pose, f, w, h, tuple(int(c * shade) for c in base))
    return np.asarray(img)
