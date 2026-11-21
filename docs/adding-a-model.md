# Adding a model

A model is a `Policy`: it declares which inputs it needs and returns a plan. The app, metrics, traces, HTTP server and every runtime work with it unchanged.

```python
import numpy as np
from alpasandbox.core import InputSpec, ModelInfo, PolicyOutput, QueryResult
from alpasandbox.models.base import Policy

class MyVLA(Policy):
    def __init__(self, checkpoint: str = "my/checkpoint", device: str = "cuda"):
        self.net = load_my_model(checkpoint, device)          # your code
        self._info = ModelInfo(
            name="my-vla", display_name="My VLA",
            # camera id -> frame offsets in seconds (<= 0); plus ego poses at 0.1 s
            drive_spec=InputSpec({0: (-0.5, 0.0), 1: (-0.5, 0.0), 2: (-0.5, 0.0)}, history_steps=10),
            vqa_spec=InputSpec({1: (0.0,)}),                   # or None if it cannot answer questions
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    def act(self, obs) -> PolicyOutput:
        # obs.frames[cam] -> list of HxWx3 uint8 RGB, oldest first, at obs.offsets[cam]
        # obs.ego_xyz (N, 3), obs.ego_rot (N, 3, 3): past poses in the current ego frame
        # obs.nav_text: the live navigation command
        xy = self.net.plan(obs)                                # (T, 2), x forward, y left, metres
        return PolicyOutput(np.asarray(xy), dt=0.1, reasoning="why, if your model says")

    def query(self, obs, question: str) -> QueryResult:        # optional
        return QueryResult(self.net.answer(obs, question))
```

Camera ids: 0 cross-left, 1 front-wide, 2 cross-right, 3 rear-left, 4 rear-tele, 5 rear-right, 6 front-tele (the PhysicalAI-AV convention). The waypoint at index *k* is reached at `(k + 1) * dt` seconds. Extra samples go in `candidates` and are drawn as alternatives.

## Use it

Without packaging:

```bash
alpasandbox app --model my_pkg.my_module:MyVLA --set checkpoint=/data/ckpt
```

As a plugin, so `--model my-vla` works, add this to your package's `pyproject.toml`:

```toml
[project.entry-points."alpasandbox.models"]
my-vla = "my_pkg.my_module:MyVLA"
```

A complete, installable example is in [examples/plugin](../examples/plugin).

## Run it somewhere else

Because the worker loads models by the same names, every runtime works for plugins too: `--runtime subprocess --worker-python ...`, `--runtime wsl ...`, or `alpasandbox serve-model --model my-vla` on a GPU box.

## Checklist for a pull request adding a model

- Inputs are built exactly as the model's upstream loader builds them. Write a pure-numpy builder and test its shapes, as `tests/test_core.py` does for Edge and Super.
- Heavy imports (torch, the model package) go inside `__init__` and methods, so `alpasandbox` imports without them.
- Add a registry entry with `requires` and `install_hint` so `alpasandbox models` and `doctor` can guide users.
