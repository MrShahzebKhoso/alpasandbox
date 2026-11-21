from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..core import Control, StepResult
from ..rig import CameraSpec
from ..scenario import Scenario


class Backend(ABC):
    """A simulator adapter. Implement `reset` and `step` (and `close`) to add a new simulator.

    Contract for `StepResult.privileged` (ground truth, used by the viewer's bird's-eye view and
    the oracle fixture, never sent to a model):
      ego     EgoState
      actors  [{id, kind, x, y, yaw, size: (L, W, H), vx, vy}]   world frame (x fwd, y left)
      lines   [{kind: "edge" | "lane" | "center", pts: [[x, y], ...]}] near the ego, world frame
    """

    name = "base"
    dt = 0.1
    wheelbase = 2.8          # metres, used by the pure-pursuit controller
    max_steer_rad = 0.5      # road-wheel angle at Control.steer = 1
    rig: tuple[CameraSpec, ...]
    active_cameras: tuple[int, ...] | None = None

    def set_cameras(self, cam_ids: Iterable[int]) -> None:
        """Only these cameras need rendering (call before `reset`)."""
        self.active_cameras = tuple(sorted(set(cam_ids)))

    def cameras(self) -> tuple[CameraSpec, ...]:
        if self.active_cameras is None:
            return self.rig
        return tuple(c for c in self.rig if c.cam_id in self.active_cameras)

    @abstractmethod
    def reset(self, scenario: Scenario) -> StepResult: ...

    @abstractmethod
    def step(self, control: Control) -> StepResult: ...

    def debug_state(self) -> dict:
        """Backend internals after the last step, for traces (e.g. the control the sim applied)."""
        return {}

    def close(self) -> None:
        pass
