# Contributing

Thanks for helping. The most valuable contributions right now:

1. **Test reports** from real setups (GPU, OS, runtime, CARLA version). Use the "Test report" issue form and paste `alpasandbox doctor`.
2. **Scenarios** (a YAML file and a line in `docs/scenarios.md`).
3. **Model adapters** for other driving VLAs (see `docs/adding-a-model.md`).
4. Fixes to the CARLA backend and the WSL runtime from people who run them daily.

## Development setup

```bash
git clone https://github.com/MrShahzebKhoso/alpasandbox && cd alpasandbox
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                    # about 3 minutes, no GPU needed
ruff check src tests
alpasandbox app --model oracle
```

## Guidelines

- Keep the base install light: numpy, pillow, pyyaml, fastapi, uvicorn, httpx. Import torch, model packages and `carla` lazily, inside the code that needs them.
- Model inputs must match the upstream loader. Add a pure-numpy builder with a shape test.
- Everything inside the package uses x forward, y left, yaw counter-clockwise, rear-axle origin. Convert at the backend boundary.
- The UI is plain HTML/CSS/JS in `src/alpasandbox/web`, with no build step. Keep it that way so anyone can edit it.
- Write UI text in sentence case and plain words. Error messages say what happened and how to fix it.

## Pull requests

Keep them focused, describe how you tested, and add or update a test when behaviour changes. By contributing you agree that your work is released under the Apache-2.0 license.
