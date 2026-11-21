# Troubleshooting

Start with `alpasandbox doctor` and the same flags you use for `app`. It checks each piece and prints a fix.

**"needs the 'alpamayo2_edge' package in this Python environment"** — the model is not installed where it runs. Install it (see the hint), or point a runtime at the env that has it: `--runtime subprocess --worker-python ...` or `--runtime wsl ...`.

**WSL: worker exits immediately.** Open the app's Log tab, or run the printed `wsl.exe ...` command yourself. Common causes are a wrong `workdir` or `venv`, or a venv created with a Python other than 3.12.

**WSL: `torch.cuda.is_available()` is False.** Update the NVIDIA driver on Windows. Do not install an NVIDIA driver inside WSL. `nvidia-smi` must work inside WSL.

**Out of memory.** Edge needs about 12 GB with six candidates and less with one. Use `--set num_traj_sets=1`, close other GPU programs, or lower `--image-size` (the model resizes to its own pixel budget anyway).

**CARLA "time-out while waiting for the simulator".** Start the server first, match the client version (`pip install carla==<server version>`), and allow ports 2000–2001 through the firewall.

**CARLA runs but the car weaves.** Check that you are on 0.2 or newer: 0.1 steered with the wrong scale. If you changed vehicles, the steering limit is read from the new vehicle's physics automatically.

**The page says "Reconnecting…".** The app process stopped; see the terminal. Reloading the page reconnects to a running app without losing the episode.

**Plans take seconds.** That is fine: the simulation is synchronous and waits. Uncheck **Real time** to stop pacing between fast steps. Latency is shown per plan and in Metrics.
