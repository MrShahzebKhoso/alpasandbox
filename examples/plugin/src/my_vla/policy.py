"""Example plugin: a camera-only lane follower in about 50 lines.

Install with `pip install -e examples/plugin`, then `alpasandbox app --model lane-follower`.
It finds the lane paint nearest the image centre on each side of several rows of the front
camera, projects those points onto the ground, fits a curve through the lane centre and
returns it as the plan. Replace `act` with your network.
"""
import math

import numpy as np

from alpasandbox.core import InputSpec, ModelInfo, PolicyOutput
from alpasandbox.models.base import Policy
from alpasandbox.rig import REFERENCE_RIG

FRONT = next(c for c in REFERENCE_RIG if c.cam_id == 1)


class LaneFollower(Policy):
    def __init__(self, speed: float = 8.0):
        self.speed = speed
        self._fit = np.zeros(3)
        self._info = ModelInfo(
            name="lane-follower", display_name="Lane follower (example plugin)",
            drive_spec=InputSpec({1: (0.0,)}, history_steps=2),
            description="Example plugin: fits the lane centre seen by the front camera.")

    @property
    def info(self) -> ModelInfo:
        return self._info

    def act(self, obs) -> PolicyOutput:
        img = obs.frames[1][-1]
        h, w, _ = img.shape
        f = (w / 2) / math.tan(math.radians(FRONT.fov_deg) / 2)
        xs, ys = [], []
        for r in range(int(h * 0.56), int(h * 0.95), max(1, h // 60)):
            paint = np.nonzero(img[r].min(axis=1) > 200)[0]          # white-ish pixels in this row
            left, right = paint[paint < w / 2], paint[paint >= w / 2]
            if not (len(left) and len(right)):
                continue
            dist = FRONT.z * f / (r - h / 2)                         # ground distance of this row
            yl, yr = (-(u - w / 2) / f * dist + FRONT.y for u in (left.max(), right.min()))
            if not 2.5 < yl - yr < 4.5:                               # not one lane wide: skip the row
                continue
            xs.append(FRONT.x + dist)
            ys.append((yl + yr) / 2)                                  # lane centre, left positive
        if len(xs) >= 4:
            self._fit = np.polyfit(xs, ys, 2 if max(xs) > 12 else 1)
        s = self.speed * 0.1 * np.arange(1, 65)
        y = np.polyval(self._fit, np.minimum(s, 25.0))              # do not trust the fit far away
        return PolicyOutput(np.stack([s, y], 1), 0.1, meta={"nominal": True, "points": len(xs)})
