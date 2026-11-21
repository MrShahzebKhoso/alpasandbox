"""Observation <-> JSON, shared by the HTTP model API and the stdio worker protocol.

Images travel as base64 JPEG (quality 95 by default): ~20x smaller than raw RGB, which matters
for 22-28 frames per step over a pipe or a network. Set `ALPASANDBOX_WIRE_QUALITY` to tune it.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image

from .core import Observation

DEFAULT_QUALITY = int(os.environ.get("ALPASANDBOX_WIRE_QUALITY", "95"))


def jpeg_b64(frame: np.ndarray, quality: int = DEFAULT_QUALITY) -> str:
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def unjpeg_b64(s: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


def encode_observation(obs: Observation, quality: int = DEFAULT_QUALITY) -> dict:
    """Serialize everything except `privileged` (ground truth never leaves the sandbox)."""
    cache: dict[int, str] = {}   # the same frame is often requested twice (e.g. drive + VQA)

    def enc(f: np.ndarray) -> str:
        key = id(f)
        if key not in cache:
            cache[key] = jpeg_b64(f, quality)
        return cache[key]

    return {
        "t": obs.t, "nav_text": obs.nav_text,
        "ego_xyz": np.asarray(obs.ego_xyz).tolist(), "ego_rot": np.asarray(obs.ego_rot).tolist(),
        "cameras": {str(c): {"offsets": list(obs.offsets[c]), "frames": [enc(f) for f in obs.frames[c]]}
                    for c in obs.frames},
    }


def decode_observation(d: dict) -> Observation:
    cams = {int(k): v for k, v in d["cameras"].items()}
    return Observation(
        t=float(d["t"]), nav_text=d.get("nav_text", ""),
        frames={c: [unjpeg_b64(s) for s in v["frames"]] for c, v in cams.items()},
        offsets={c: tuple(v["offsets"]) for c, v in cams.items()},
        ego_xyz=np.asarray(d["ego_xyz"], np.float32), ego_rot=np.asarray(d["ego_rot"], np.float32),
    )


def stamps_us(obs: Observation, cid: int) -> list[int]:
    """Capture timestamps in microseconds. The clock is shifted +2.5 s so none is negative
    (the models only use relative time labels)."""
    base = obs.t + 2.5
    return [int(round((base + off) * 1e6)) for off in obs.offsets[cid]]


def t0_us(obs: Observation) -> int:
    return int(round((obs.t + 2.5) * 1e6))


def check_frames(obs: Observation, cameras: dict[int, tuple[float, ...]]) -> tuple[int, int]:
    """Validate that obs holds exactly the frames a profile needs; return the shared (H, W)."""
    shapes = {f.shape for c in cameras for f in obs.frames.get(c, [])}
    if len(shapes) != 1:
        raise ValueError(f"all camera frames must share one HxWx3 shape, got {shapes}")
    for cid, offs in cameras.items():
        if len(obs.frames.get(cid, [])) != len(offs):
            raise ValueError(f"camera {cid}: need {len(offs)} frames, got {len(obs.frames.get(cid, []))}")
    h, w, _ = next(iter(shapes))
    return h, w
