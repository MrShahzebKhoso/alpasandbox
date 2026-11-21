# Installation

AlpaSandbox itself is a small, pure-Python package (Python 3.10–3.12). The model is the heavy part: Alpamayo 2 Edge needs an NVIDIA GPU with at least 16 GB and bf16 support (Ampere or newer), Python 3.12, PyTorch 2.8 and Transformers 5.17, as pinned by NVIDIA. Pick the setup that matches your machine.

Every setup ends with the same two commands:

```bash
alpasandbox doctor   [same flags as below]   # checks each piece and says how to fix it
alpasandbox app      [same flags as below]   # opens http://localhost:8000
```

Put the flags in `alpasandbox.toml` (`alpasandbox init` writes a commented one) so you can type just `alpasandbox app`.

## Before you start: model access

1. Accept the terms on [nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge).
2. Log in where the model will run: `hf auth login`.
3. The first launch downloads about 5.3 GB of weights. After that you can set `HF_HUB_OFFLINE=1`.

## A. Linux (or everything inside WSL2)

```bash
git clone https://github.com/NVlabs/alpamayo2-edge && cd alpamayo2-edge
uv sync --locked && source .venv/bin/activate        # NVIDIA's environment (Python 3.12, torch 2.8)
pip install git+https://github.com/MrShahzebKhoso/alpasandbox  # into the same venv
alpasandbox app --model edge
```

## B. Windows + WSL2: CARLA on Windows, model in Linux

This is the setup the project was born from. CARLA renders natively on Windows. The model runs in WSL2, where NVIDIA's Linux-only stack works, and AlpaSandbox on Windows drives it through `wsl.exe`. You do not need to install AlpaSandbox inside WSL: it copies its small pure-Python package to `%LOCALAPPDATA%\alpasandbox\worker-src` and points the WSL worker at it.

1. **Inside WSL** (Ubuntu 24.04 recommended), set up the model. Use the NVIDIA Windows driver. Do not install a Linux NVIDIA driver inside WSL.
   ```bash
   bash scripts/setup_wsl.sh            # or follow section A without the last two lines
   ```
2. **On Windows** (PowerShell), install AlpaSandbox and the CARLA client that matches your server version:
   ```powershell
   py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
   pip install "git+https://github.com/MrShahzebKhoso/alpasandbox" carla==0.9.16
   ```
3. Start `CarlaUE4.exe` (0.9.x) and run:
   ```powershell
   alpasandbox app --backend carla --model edge --runtime wsl `
     --wsl-distro Ubuntu-24.04 --wsl-workdir ~/alpamayo2-edge --wsl-venv .venv
   ```

The worker's log, including model loading, appears in the app's **Log** tab and in the terminal.

## C. Model in its own Python environment (any OS)

This keeps NVIDIA's pinned torch and Transformers away from your other tools. Works on Linux, and on native Windows if PyTorch CUDA works there for you.

```bash
alpasandbox app --model edge --runtime subprocess --worker-python /path/to/alpamayo2-edge/.venv/bin/python
```

## D. Remote GPU (lab server, cloud box)

On the GPU machine:

```bash
pip install git+https://github.com/MrShahzebKhoso/alpasandbox   # inside the model's env
ALPASANDBOX_TOKEN=change-me alpasandbox serve-model --model edge --port 8100
```

On your machine:

```bash
alpasandbox app --model edge --runtime remote --remote-url http://gpu-box:8100
# set [runtime.remote] token = "change-me" in alpasandbox.toml
```

Use an SSH tunnel (`ssh -L 8100:localhost:8100 gpu-box`) rather than exposing the port to the internet.

## E. Native Windows, no WSL

AlpaSandbox, the built-in simulator, CARLA and the `oracle` and `cruise` fixtures run natively on Windows. NVIDIA validates Alpamayo on Linux only. If you get its environment working natively on Windows, use `--runtime local` or `subprocess` and please report it in an issue.

## Docker

```bash
docker compose -f docker/compose.yml --profile lite up      # app + oracle, no GPU
docker compose -f docker/compose.yml --profile gpu up       # model server + app (needs nvidia-container-toolkit)
```
