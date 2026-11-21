from __future__ import annotations

from abc import ABC, abstractmethod

from ..core import ModelInfo, Observation, PolicyOutput, QueryResult


class Policy(ABC):
    """A driving model. Implement `info` and `act`; add `query` if it can answer questions.

    `act` receives an Observation built from `info.drive_spec`; `query` one built from
    `info.vqa_spec`. Everything is in the ego frame (x forward, y left, rear-axle origin).
    """

    @property
    @abstractmethod
    def info(self) -> ModelInfo: ...

    @property
    def name(self) -> str:
        return self.info.name

    def reset(self) -> None:
        """Called at the start of every episode."""

    @abstractmethod
    def act(self, obs: Observation) -> PolicyOutput: ...

    def query(self, obs: Observation, question: str) -> QueryResult:
        raise NotImplementedError(f"{self.info.display_name} does not answer free-form questions")

    def close(self) -> None:
        """Release GPU memory, subprocesses, sockets."""


def flatten_text(value) -> list[str]:
    """Upstream returns CoT as nested lists / numpy arrays of strings: flatten to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        import numpy as np
        return [str(x) for x in np.asarray(value, dtype=object).reshape(-1).tolist()]
    except Exception:  # pragma: no cover - defensive
        out: list[str] = []
        for v in value:
            out.extend(flatten_text(v))
        return out


def clean_reasoning(text: str) -> tuple[str, bool]:
    """Return (reasoning, nominal). Edge emits `<NOMINAL>` (or nothing) for routine driving."""
    t = (text or "").strip()
    nominal = (not t) or "NOMINAL" in t.upper()
    return ("" if nominal else t), nominal
