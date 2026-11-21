"""Plain data types shared by backends, models, the engine and the web app.

Frame convention everywhere: x forward, y LEFT, z up, yaw counter-clockwise (radians),
origin at the rear axle. Backends convert into this frame; models receive it unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class EgoState:
    t: float
    x: float
    y: float
    yaw: float
    v: float
    steer: float = 0.0


@dataclass
class Control:
    steer: float = 0.0     # [-1, 1], left positive, 1 = the vehicle's maximum steering angle
    throttle: float = 0.0  # [0, 1]
    brake: float = 0.0     # [0, 1]


@dataclass
class StepResult:
    t: float
    ego: EgoState
    images: dict[int, np.ndarray]          # cam_id -> HxWx3 uint8 RGB
    collision: bool = False
    off_road: bool = False
    lane_offset: float = 0.0               # metres from ego-lane centre, left positive
    done: bool = False
    privileged: dict[str, Any] = field(default_factory=dict)  # ground truth: UI + oracle only


@dataclass(frozen=True)
class InputSpec:
    """Which camera frames (seconds relative to now, <= 0) and how many ego poses a task needs."""

    cameras: dict[int, tuple[float, ...]]
    history_steps: int = 20

    @property
    def camera_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.cameras))

    @property
    def num_images(self) -> int:
        return sum(len(v) for v in self.cameras.values())

    @property
    def horizon(self) -> float:
        return max((-min(v) for v in self.cameras.values() if v), default=0.0)

    def to_dict(self) -> dict:
        return {"cameras": {str(k): list(v) for k, v in self.cameras.items()},
                "history_steps": self.history_steps}

    @classmethod
    def from_dict(cls, d: dict) -> InputSpec:
        return cls({int(k): tuple(float(x) for x in v) for k, v in d["cameras"].items()},
                   int(d.get("history_steps", 20)))


@dataclass
class ModelInfo:
    """Self-description a model publishes so the app can build the right inputs and UI."""

    name: str                                # registry key, e.g. "edge"
    display_name: str                        # e.g. "Alpamayo 2 Edge"
    drive_spec: InputSpec
    vqa_spec: InputSpec | None = None        # None = no free-form question answering
    description: str = ""
    homepage: str = ""
    license: str = ""
    privileged: bool = False                 # reads simulator ground truth (test fixtures only)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def supports_vqa(self) -> bool:
        return self.vqa_spec is not None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["drive_spec"] = self.drive_spec.to_dict()
        d["vqa_spec"] = self.vqa_spec.to_dict() if self.vqa_spec else None
        d["supports_vqa"] = self.supports_vqa
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ModelInfo:
        return cls(
            name=d["name"], display_name=d.get("display_name", d["name"]),
            drive_spec=InputSpec.from_dict(d["drive_spec"]),
            vqa_spec=InputSpec.from_dict(d["vqa_spec"]) if d.get("vqa_spec") else None,
            description=d.get("description", ""), homepage=d.get("homepage", ""),
            license=d.get("license", ""), privileged=bool(d.get("privileged", False)),
            extra=dict(d.get("extra", {})),
        )


@dataclass
class Observation:
    """What a model sees. Built from an InputSpec; mirrors the model's input contract."""

    t: float
    frames: dict[int, list[np.ndarray]]    # cam_id -> frames, oldest -> newest
    offsets: dict[int, tuple[float, ...]]  # cam_id -> seconds relative to t (<= 0)
    ego_xyz: np.ndarray                    # (N, 3) past poses in the ego frame at t, oldest first
    ego_rot: np.ndarray                    # (N, 3, 3)
    nav_text: str
    privileged: dict[str, Any] | None = None   # never sent over the wire


@dataclass
class PolicyOutput:
    waypoints: np.ndarray                  # (N, 2) ego frame: x forward, y left (the plan we drive)
    dt: float = 0.1                        # spacing; waypoint k is at (k + 1) * dt seconds
    reasoning: str = ""                    # empty for routine ("nominal") driving
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
    candidates: list[np.ndarray] = field(default_factory=list)   # extra (N, 2) samples, for display
    candidate_reasoning: list[str] = field(default_factory=list)


@dataclass
class QueryResult:
    answer: str
    latency_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
