<div align="center">

<img src="docs/assets/logo.svg" width="72" alt="">

# AlpaSandbox

**Drive NVIDIA Alpamayo 2 in closed loop, watch its reasoning, and question it about the scene, all from your browser.**

[![CI](https://github.com/MrShahzebKhoso/alpasandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/MrShahzebKhoso/alpasandbox/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-3776AB)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Platforms](https://img.shields.io/badge/runs%20on-Linux%20%7C%20Windows%20%7C%20Windows%2BWSL2-555)](docs/installation.md)

<img src="assets/demo.gif" alt="AlpaSandbox in action: Alpamayo 2 Edge driving CARLA Town10, with the 3D driving view, six camera feeds, controls and the model's reasoning updating live" width="100%">

<sub>Alpamayo 2 Edge driving CARLA Town10 from its own six-camera input (24 seconds at 5× speed).</sub>

</div>

NVIDIA's [Alpamayo 2 Edge](https://github.com/NVlabs/alpamayo2-edge) is an open driving vision-language-action (VLA) model, and many people are experimenting with it. AlpaSandbox is a live demo around it: put the model behind the wheel, ask it questions while it drives, and see in real time what it thinks, where it plans to go, and the exact steering, throttle and brake it produces.

AlpaSandbox gives a driving VLA model a car. It renders the model's exact multi-camera input from a simulator (CARLA, or a built-in one that needs nothing installed), runs the model, drives with the trajectory it returns, and shows everything live: a 3D driving view, six cameras, the model's Chain-of-Causation reasoning, speed and braking, and closed-loop metrics.

You can steer the model while it drives. Send a **navigation command** ("Turn right in 30m") and it conditions the next plan. Or **ask it a question** about what it sees ("Is it safe to change lanes?"). The simulation waits while the model answers.

## A tour of the app

<p align="center"><img src="assets/alpasanbox_full.png" alt="The full AlpaSandbox app: driving view with HUD and traffic light control, six camera feeds, steering, throttle and brake, a speed chart, and the model's timed reasoning beside the navigation command box" width="100%"></p>

The driving view takes the top left, the cameras and vehicle state sit on the right, and the tabs below hold drive instructions, the conversation, metrics and the log.

### Driving view

A tilted chase camera follows the car. The plan is drawn as a ribbon one car wide with one-second marks, the path driven so far as a cyan line, and other road users as 3D boxes. The HUD shows the steering command, speed, how far the plan reaches, and simulated time. Switch to a flat top-down view with **2D**, and zoom with **+** and **−**.

<p align="center"><img src="assets/3d_view.png" alt="3D driving view: tilted chase camera with the model's plan as a green ribbon, lane lines, and a HUD showing steering, speed, plan length and time" width="100%"></p>
<p align="center"><img src="assets/2d_view.png" alt="Top-down view: lanes and crosswalks, the plan as an amber line with one-second dots, and the HUD" width="100%"></p>

### What the model sees

The six cameras Alpamayo 2 Edge uses, ordered front to back. They are mounted at the medians of NVIDIA's [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) rig calibration (see [docs/calibration.md](docs/calibration.md)). Click a camera, or press 1–6, to enlarge it.

<p align="center"><img src="assets/cameras.png" alt="Six camera feeds: front tele, front wide, cross left, cross right, rear left, rear right" width="70%"></p>

### Drive instructions and reasoning

Send a navigation command, or pick a preset. Each change in the model's reasoning appears as a timed bullet under the command that prompted it, and the latest one is shown in the tab bar.

<p align="center"><img src="assets/drive_reasoning.png" alt="Drive instructions tab: timed reasoning bullets on the left, preset commands and the command box on the right" width="100%"></p>

### Ask the model

Free-form questions about the current scene, answered by the model's VQA head from the cameras alone. The simulation pauses while it answers.

<p align="center"><img src="assets/conversation.png" alt="Conversation tab: questions and the model's answers, with preset questions and an Ask box" width="100%"></p>

### Vehicle state, metrics and traffic lights

Steering, throttle and brake, plan latency, lane offset, and speed over the episode, with braking shaded and a mark wherever the model gave a reason. In CARLA towns you can hold the ego's traffic light at red, yellow or green to test the model.

<p align="center">
  <img src="assets/controls.png" alt="Steer, throttle and brake bars, latency per plan, lane offset, and a speed and braking chart" width="49%">
  <img src="assets/metrics.png" alt="Metrics: distance, mean speed, incidents, collisions, road departures, lane offset, latency" width="49%">
</p>
<p align="center"><img src="assets/traffic_light.png" alt="Traffic light control: current state with R, Y, G and Auto buttons" width="34%"></p>

The **Log** tab lists what the app is doing: backend start, model load time, scenario changes and traffic light overrides.

<p align="center"><img src="assets/logs.png" alt="Log tab: timestamped entries for backend start, model load, scenario start and a traffic light held at green" width="100%"></p>

## Highlights

- **Alpamayo 2 Edge and Super, out of the box.** Adapters follow NVIDIA's own inference code: the 22-image Edge driving profile, the 24-image Edge VQA profile, and the seven-camera Super ring with upstream task selection.
- **Runs where your GPU is.** Same process, a separate Python environment, inside **WSL2 from Windows** (CARLA on Windows, model in Linux), or on a remote GPU server over HTTP.
- **Closed loop, synchronous.** The simulator waits for the model, so a 1.5-second plan on a laptop GPU still drives correctly. The controller tracks the whole plan, not just its first second, and handles the throttle dead band of a car pulling away from rest.
- **Nine scenarios, YAML-defined**: pedestrian crossing, occluded pedestrian, cut-in, lead-vehicle braking, slow lead, oncoming traffic, cone obstruction, S-curves, straight cruise. In CARLA they are built as real roads with OpenDRIVE, or run in any town with optional traffic.
- **Bring your own model.** Implement `act()` (and optionally `query()`), then register it with an entry point or pass `--model my_pkg.mod:MyPolicy`. The app, metrics and HTTP API work unchanged.
- **Reproducible output.** Per-step JSONL traces (pose, controls, plan, reasoning, latency, controller and simulator internals), a headless `run --all` mini-benchmark, and `doctor` to check a setup.

## Getting started: Alpamayo 2 Edge in CARLA

This is the main setup and the one the project was built and tested on: CARLA renders the world (on Windows or Linux) and Alpamayo 2 Edge drives from its cameras in WSL2 or Linux. AlpaSandbox does not include NVIDIA's model code or weights; you install them from NVIDIA in step 2.

### What you need

| | |
|---|---|
| GPU | NVIDIA, **16 GB or more**, with bf16 support (Ampere or newer). CARLA and the model can share one 16 GB laptop GPU. |
| OS | Linux, or Windows 11 with **WSL2** (Ubuntu 24.04 tested). Inside WSL use the normal NVIDIA *Windows* driver; do not install a Linux driver. |
| Python | 3.12, managed by [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh \| sh`) |
| Model access | A Hugging Face account that has accepted the terms of [nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge) |
| Simulator | [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases) (about 20 GB unpacked) |
| Disk | About 5.3 GB for the model weights, plus the Python environment |

### 1. Start CARLA headless

Download the CARLA 0.9.16 package for your OS from the [CARLA releases page](https://github.com/carla-simulator/carla/releases) and unpack it. Start it without a window, which saves GPU memory for the model:

```powershell
# Windows (PowerShell, inside the CARLA folder)
.\CarlaUE4.exe -RenderOffScreen -quality-level=Low -nosound
```

```bash
# Linux
./CarlaUE4.sh -RenderOffScreen -quality-level=Low -nosound
```

CARLA listens on TCP ports 2000 and 2001. On Windows, allow `CarlaUE4` through the firewall when asked (WSL connects to it over the network). Leave it running.

### 2. Install Alpamayo 2 Edge (Linux or WSL)

```bash
git clone https://github.com/NVlabs/alpamayo2-edge && cd alpamayo2-edge
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --locked
source .venv/bin/activate
hf auth login                      # paste a Hugging Face token with access to the model
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported())"   # both must be True
```

This is NVIDIA's own environment (PyTorch 2.8, Transformers 5.17). See their [README](https://github.com/NVlabs/alpamayo2-edge) if anything fails here.

### 3. Install AlpaSandbox into the same environment

With the `alpamayo2-edge` environment still active:

```bash
uv pip install "git+https://github.com/MrShahzebKhoso/alpasandbox" carla==0.9.16
```

The `carla` client must match the server version exactly. To work on AlpaSandbox itself, clone it and use `uv pip install -e ./alpasandbox carla==0.9.16` instead.

**Optional, faster plans:** install FlashAttention 2.8.3 from its prebuilt wheel (this one matches the pinned PyTorch 2.8, CUDA 12, Python 3.12, x86-64):

```bash
uv pip install --no-deps "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

### 4. Check the setup

```bash
# In WSL, CARLA on Windows is reached through the Windows host IP:
CARLA_HOST=$(ip route | awk '/default/ {print $3}')
# On Linux with CARLA on the same machine: CARLA_HOST=localhost

alpasandbox doctor --model edge --backend carla --carla-host $CARLA_HOST
```

`doctor` checks Python, the GPU, the model package, the CARLA connection and the config, and says how to fix anything missing.

### 5. Drive

```bash
alpasandbox app --model edge --backend carla --carla-host $CARLA_HOST --carla-town Town10HD_Opt
```

Open <http://localhost:8000>, pick a scenario and press **Drive**. The first launch downloads the weights (about 5.3 GB); afterwards set `export HF_HUB_OFFLINE=1` to skip the online check.

If you installed FlashAttention, add `--set accel=tuned --set attn_implementation=flash_attention_2`. This uses NVIDIA's acceleration in `alpamayo2_edge.accel`: the first plan compiles CUDA graphs and takes about 15 seconds, and later plans take about 1.5 s on a 16 GB laptop GPU shared with CARLA. If GPU memory runs short, also `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

While it drives you can:

- send a navigation command from **Drive instructions** ("Turn right in 30m.", "Change to the left lane.") and watch the reasoning react,
- ask a question in **Conversation** ("Why did you change lanes?"); the simulation waits while the model answers,
- hold the traffic light at red, yellow or green, switch between the 3D and 2D views, and download the trace.

### 6. Save your settings (optional)

`alpasandbox init` writes a commented `alpasandbox.toml` in the current folder. With your settings in it, `alpasandbox app` is all you need to type:

```toml
[app]
scenario = "straight_cruise"

[sim]
backend = "carla"

[carla]
host = "172.x.x.x"       # the Windows host IP from step 4, or "localhost"
town = "Town10HD_Opt"
spawn_index = 1          # which spawn point of the map the car starts at

[model]
name = "edge"

[model.settings]
accel = "tuned"                           # only with FlashAttention installed
attn_implementation = "flash_attention_2"
```

Useful model settings: `num_traj_sets=3` draws alternative plans, `diffusion_steps=5` trades accuracy for speed, `model_id=/path/to/checkpoint` loads local weights. All settings are in [docs/configuration.md](docs/configuration.md), and CARLA options (other towns, traffic, scenario roads) in [docs/carla.md](docs/carla.md).

### Other ways to run the model

The model can also run in a separate environment from the app. These are implemented but **not tested on real hardware yet**; see [docs/installation.md](docs/installation.md) for each.

| Setup | Command |
|---|---|
| AlpaSandbox on Windows, model in WSL | `alpasandbox app --model edge --runtime wsl --wsl-distro Ubuntu-24.04 --wsl-workdir ~/alpamayo2-edge --wsl-venv .venv --backend carla` |
| Model in its own venv on the same machine | `alpasandbox app --model edge --runtime subprocess --worker-python ~/alpamayo2-edge/.venv/bin/python` |
| Remote GPU | GPU box: `alpasandbox serve-model --model edge`<br>Your machine: `alpasandbox app --model edge --runtime remote --remote-url http://gpu-box:8100` |

Without CARLA, drop the `--backend carla` flags: the built-in simulator draws simple roads and needs nothing installed. Alpamayo was trained on real video, so it drives better from CARLA's imagery.

## Try the app without a GPU

To look around the interface before setting up the model, AlpaSandbox can run on its own with the built-in simulator and `oracle`, a rule-based driver that reads the simulator's ground truth instead of cameras (so it does not reason or answer real questions):

```bash
pip install git+https://github.com/MrShahzebKhoso/alpasandbox
alpasandbox app --model oracle
```

Then open <http://localhost:8000>.

> **Note:** this mode is not tested yet beyond automated checks (the app starts and headless runs complete). It is a test fixture, not a model, and is not how the screenshots above were made. Please report anything that does not work.

## Models

| Model | `--model` | Drives | Answers questions | GPU | Status |
|---|---|---|---|---|---|
| NVIDIA Alpamayo 2 Edge (2B) | `edge` | yes | yes | ≥ 16 GB | tested locally with CARLA (see below) |
| NVIDIA Alpamayo 2 Super (34B) | `super` | yes | yes | ~80 GB | **experimental**, untested; use `--runtime remote` |
| Oracle (ground truth) | `oracle` | yes | templates | none | test fixture |
| Cruise | `cruise` | yes | no | none | blind baseline |
| Your model | `pkg.module:Class` | | | | see [docs/adding-a-model.md](docs/adding-a-model.md) |

## What has been tested

On one machine: Windows 11 with CARLA 0.9.16, the model in WSL2 Ubuntu with the `local` runtime, an RTX 4090 Laptop GPU (16 GB) shared by CARLA and Alpamayo 2 Edge, Town10HD_Opt.

| Check | Result |
|---|---|
| Edge loads, drives, reasons and answers questions | Yes, about 1.5 s per plan with `accel=tuned` and flash-attn |
| Pulls away from a stop line when the light turns green | Yes, within two plans |
| Light switched to red about 25 m out at 50 km/h | Stops before the line |
| Holds at a red light after stopping | **Not reliably.** Edge sometimes decides to go on red ("the light has turned green"). This is the model's decision, not the controller's |
| Lane keeping on Town10 straights | No collisions in 15-second runs, but Edge holds the car up to about 1 m off the lane centre |

Not tested on real hardware yet: the `oracle` no-GPU mode beyond automated checks, the `subprocess`, `wsl` and `remote` runtimes, Alpamayo 2 Super, Linux-native CARLA, and the town placement of scenario actors (pedestrians and vehicles placed along the lane can land on kerbs and fail to spawn; the Log tab warns when they do). Reports from other setups are very welcome.

## How it works

```mermaid
flowchart LR
  subgraph Sim[Simulator backend]
    B[builtin or CARLA]
  end
  B -- 6-7 cameras + ego pose --> R[Rig buffers<br/>frames at model offsets<br/>ego history in ego frame]
  R -- Observation --> M[Model runtime<br/>local / subprocess / WSL / HTTP]
  M -- 64 waypoints + reasoning --> C[Pure pursuit + speed control]
  C -- steer, throttle, brake --> B
  N[Navigation command] --> R
  Q[Question] --> M
  M -. answer .-> UI[Web app]
  B -. cameras, ground truth .-> UI
```

Each model publishes an `InputSpec` that lists which cameras it needs, at which time offsets, and how many ego poses. The rig builds exactly that from a rolling buffer, so Edge gets its 22 images and Super gets its 7 × 4 ring from the same simulator. More detail: [docs/architecture.md](docs/architecture.md).

## Command line

```text
alpasandbox app          web app (alias: demo)
alpasandbox run          headless closed loop; --all runs every scenario, --trace writes JSONL,
                         --signal / --signal-at hold or switch a CARLA traffic light, --steps limits length
alpasandbox serve-model  HTTP model server for a remote GPU
alpasandbox smoke        load the model, time a few plans and one question
alpasandbox doctor       check Python, GPU, model packages, CARLA, WSL and config
alpasandbox models       list models, and how to install the missing ones
alpasandbox scenarios    list scenarios
alpasandbox init         write a commented alpasandbox.toml
```

## What this is, and what it is not

The built-in renderer is schematic, and even CARLA imagery differs from the real-world video Alpamayo was trained on. The simulated cameras use the real rig's mounts but pinhole lenses, while the real wide cameras are f-theta. Treat results as qualitative and comparative. This is **not** safety validation. `Incidents` counts collisions and road departures without judging fault, so it is not comparable to NVIDIA's at-fault figures. Traffic-light states shown in the app come from the simulator; the model only sees the cameras.

## Contributing

Issues and pull requests are welcome, especially new scenarios, model adapters, and test reports from different GPUs and platforms. Please include your `alpasandbox doctor` output. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

If AlpaSandbox helps your research, please cite it (GitHub shows a "Cite this repository" button from [CITATION.cff](CITATION.cff)):

```bibtex
@software{khoso_alpasandbox_2026,
  author  = {Khoso, Shahzeb},
  title   = {AlpaSandbox: an interactive closed-loop sandbox for driving vision-language-action models},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/MrShahzebKhoso/alpasandbox}
}
```

## License

AlpaSandbox code is Apache-2.0. It does not include NVIDIA's code or weights: the Alpamayo model code is Apache-2.0 and the weights are released by NVIDIA under OpenMDW-1.1, both installed separately. CARLA keeps its own licenses. This project is not affiliated with or endorsed by NVIDIA.
