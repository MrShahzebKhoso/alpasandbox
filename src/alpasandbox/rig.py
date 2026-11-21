"""Camera rig, input profiles and temporal buffers.

Camera ids follow NVIDIA's PhysicalAI-AV convention used by the Alpamayo family:
0 cross-left, 1 front-wide, 2 cross-right, 3 rear-left, 4 rear-tele, 5 rear-right, 6 front-tele.

Mount poses are the PhysicalAI-AV fleet medians (rear-axle origin); see docs/calibration.md.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .core import EgoState, InputSpec, Observation

CAMERA_NAMES = {0: "Cross left", 1: "Front wide", 2: "Cross right", 3: "Rear left",
                4: "Rear tele", 5: "Rear right", 6: "Front tele"}


@dataclass(frozen=True)
class CameraSpec:
    cam_id: int
    name: str
    x: float        # mount, vehicle frame (rear-axle origin), metres
    y: float
    z: float
    yaw_deg: float  # relative to vehicle heading, CCW
    fov_deg: float
    pitch_deg: float = 0.0


#: Medians over 100 PhysicalAI-AV clips (calibration/sensor_extrinsics chunk 0000): rear-axle origin at
#: ground level. Yaw rounded to 1 deg, |pitch| <= 1.1 deg set to 0. See docs/calibration.md. The cross
#: cameras look ~67 deg to the side (not 90) and sit low on the front fenders, like the mirror-mounted
#: rear corners. The real 120 deg lenses are f-theta (~67 deg vertical); simulator pinholes are not.
REFERENCE_RIG: tuple[CameraSpec, ...] = (
    CameraSpec(0, "cross_left", 2.52, 0.94, 0.90, 67, 120),
    CameraSpec(1, "front_wide", 1.75, 0.0, 1.43, 0, 120),
    CameraSpec(2, "cross_right", 2.52, -0.94, 0.90, -67, 120),
    CameraSpec(3, "rear_left", 2.00, 1.03, 0.90, 152, 70),
    CameraSpec(4, "rear_tele", 0.43, 0.31, 1.46, 180, 30),
    CameraSpec(5, "rear_right", 2.00, -1.03, 0.90, -152, 70),
    CameraSpec(6, "front_tele", 1.72, 0.09, 1.44, 0, 30),
)

# --- Input profiles (from the upstream loaders) -------------------------------------------
_FRONT = (-2.0, -1.0, -0.5, -0.1, 0.0)
_SIDE = (-1.0, -0.1, 0.0)
_FOUR = (-0.3, -0.2, -0.1, 0.0)

#: Alpamayo 2 Edge driving: 22 images, 20 ego poses (-1.9 .. 0 s).
EDGE_DRIVE_SPEC = InputSpec({0: _SIDE, 1: _FRONT, 2: _SIDE, 3: _SIDE, 5: _SIDE, 6: _FRONT}, 20)
#: Alpamayo 2 Edge free-form VQA: six driving cameras x four frames (-0.3 .. 0 s).
EDGE_VQA_SPEC = InputSpec({c: _FOUR for c in (0, 1, 2, 3, 5, 6)}, 20)
#: Alpamayo 2 Super: the full seven-camera ring x four frames; 16 ego poses (-1.5 .. 0 s).
#: Upstream `select_task_input` then picks the task's six cameras.
SUPER_SPEC = InputSpec({c: _FOUR for c in range(7)}, 16)

# Back-compat aliases used by older code and tests.
CAMERA_FRAME_COUNTS = {c: len(v) for c, v in EDGE_DRIVE_SPEC.cameras.items()}
HISTORY_STEPS = EDGE_DRIVE_SPEC.history_steps
MAX_HISTORY = 32


class FrameBuffer:
    """Keeps the last few seconds of frames for every camera; gathers any InputSpec from them.

    At episode start the oldest available frame is repeated (images cannot be back-filled).
    """

    def __init__(self, dt: float, horizon_s: float = 2.0):
        self.dt = dt
        self._buf: deque[dict[int, np.ndarray]] = deque(maxlen=int(round(horizon_s / dt)) + 1)

    def push(self, images: dict[int, np.ndarray]) -> None:
        self._buf.append(images)

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> dict[int, np.ndarray]:
        return self._buf[-1] if self._buf else {}

    def gather(self, spec: InputSpec) -> dict[int, list[np.ndarray]]:
        if not self._buf:
            raise RuntimeError("frame buffer is empty")
        last = len(self._buf) - 1
        out: dict[int, list[np.ndarray]] = {}
        for cid, offsets in spec.cameras.items():
            if cid not in self._buf[-1]:
                raise KeyError(f"camera {cid} ({CAMERA_NAMES.get(cid, '?')}) is not rendered by the "
                               "backend; add it to the rig")
            out[cid] = [self._buf[max(0, last - int(round(-off / self.dt)))][cid] for off in offsets]
        return out


class EgoHistory:
    """Recent poses at 0.1 s, expressed in the ego frame at the newest pose (model convention)."""

    def __init__(self, maxlen: int = MAX_HISTORY):
        self._h: deque[EgoState] = deque(maxlen=maxlen)

    def push(self, s: EgoState) -> None:
        self._h.append(s)

    def arrays(self, steps: int = HISTORY_STEPS) -> tuple[np.ndarray, np.ndarray]:
        hist = list(self._h)[-steps:]
        if not hist:
            raise RuntimeError("ego history is empty")
        if len(hist) < steps:  # episode start: assume the car was already moving at its current speed
            f = hist[0]
            pad = [EgoState(f.t - 0.1 * k, f.x - f.v * 0.1 * k * math.cos(f.yaw),
                            f.y - f.v * 0.1 * k * math.sin(f.yaw), f.yaw, f.v, f.steer)
                   for k in range(steps - len(hist), 0, -1)]
            hist = pad + hist
        ref = hist[-1]
        c, s = math.cos(ref.yaw), math.sin(ref.yaw)
        xyz = np.zeros((steps, 3))
        rot = np.zeros((steps, 3, 3))
        for i, h in enumerate(hist):
            dx, dy = h.x - ref.x, h.y - ref.y
            xyz[i] = (c * dx + s * dy, -s * dx + c * dy, 0.0)
            d = h.yaw - ref.yaw
            cd, sd = math.cos(d), math.sin(d)
            rot[i] = ((cd, -sd, 0), (sd, cd, 0), (0, 0, 1))
        return xyz.astype(np.float32), rot.astype(np.float32)


def build_observation(t: float, frame_buf: FrameBuffer, ego_hist: EgoHistory, spec: InputSpec,
                      nav_text: str, privileged=None) -> Observation:
    xyz, rot = ego_hist.arrays(spec.history_steps)
    return Observation(t=t, frames=frame_buf.gather(spec), offsets=dict(spec.cameras),
                       ego_xyz=xyz, ego_rot=rot, nav_text=nav_text, privileged=privileged)
