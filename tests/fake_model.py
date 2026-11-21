"""A tiny image-reading 'model' used to test the worker protocol and plugin loading."""
import numpy as np

from alpasandbox.core import ModelInfo, PolicyOutput, QueryResult
from alpasandbox.models.base import Policy
from alpasandbox.rig import EDGE_DRIVE_SPEC, EDGE_VQA_SPEC


class EchoModel(Policy):
    def __init__(self, speed: float = 5.0, candidates: int = 2):
        self.speed, self.k = speed, candidates
        self._info = ModelInfo("echo", "Echo test model", EDGE_DRIVE_SPEC, EDGE_VQA_SPEC)

    @property
    def info(self):
        return self._info

    def act(self, obs):
        n = sum(len(v) for v in obs.frames.values())
        s = self.speed * 0.1 * np.arange(1, 65)
        wp = np.stack([s, np.zeros_like(s)], 1)
        return PolicyOutput(wp, 0.1, f"{n} frames; nav={obs.nav_text}", 1.0, {"nominal": False},
                            candidates=[wp + [0, 0.5 * (i + 1)] for i in range(self.k)])

    def query(self, obs, question):
        return QueryResult(f"{question} | {sum(len(v) for v in obs.frames.values())} frames", 2.0)
