# Runtimes: where the model runs

| Runtime | Use when | Protocol |
|---|---|---|
| `local` | The model's packages are importable in the Python running AlpaSandbox | direct call |
| `subprocess` | The model lives in another venv or conda env on this machine | JSON lines over stdin/stdout |
| `wsl` | AlpaSandbox runs on Windows and the model inside WSL2 | the same JSON lines, through `wsl.exe` |
| `remote` | The GPU is another machine | HTTP (`alpasandbox serve-model`) |

All four present the same `Policy` interface to the app, so nothing else changes.

## How the WSL runtime starts the model

`wsl.exe -d <distro> --exec bash --noprofile --norc -c <script>`, where the script does the following:

1. `cd` into `[runtime.wsl] workdir` (`~` is expanded inside WSL).
2. Activates `venv` (relative to workdir, or absolute).
3. Adds the staged AlpaSandbox package to `PYTHONPATH` (`pythonpath = "auto"`; the path is translated with `wslpath`).
4. Exports your `env` table, e.g. `{ HF_HUB_OFFLINE = "1" }`.
5. Runs `exec python -u -m alpasandbox.worker --model edge --settings-b64 ...`.

The worker reserves the real stdout file descriptor for the protocol and redirects fd 1 to stderr, so CUDA or C++ libraries that print to stdout cannot corrupt the JSON stream. Every request has a timeout (`timeout`, default 300 s; `startup_timeout`, default 1800 s for the first weight download), and a hung or crashed worker surfaces as an error in the app instead of freezing it.

## Settings

```toml
[runtime.wsl]
distro = "Ubuntu-24.04"
workdir = "~/alpamayo2-edge"
venv = ".venv"
python = "python"
pythonpath = "auto"     # or "" if alpasandbox is pip-installed in the WSL venv
env = { HF_HUB_OFFLINE = "1" }
timeout = 300
startup_timeout = 1800
```

`[runtime.subprocess]` takes the same keys, plus `python` (path to the interpreter) and `cwd`.
