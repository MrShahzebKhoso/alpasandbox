# Configuration

Precedence: built-in defaults < `alpasandbox.toml` < environment variables < command-line flags.

The file is read from `--config PATH`, else `./alpasandbox.toml`, else `~/.config/alpasandbox/alpasandbox.toml`. `alpasandbox init` writes a commented starter.

| Environment variable | Sets |
|---|---|
| `ALPASANDBOX_MODEL` | `[model] name` |
| `ALPASANDBOX_RUNTIME` | `[model] runtime` |
| `ALPASANDBOX_BACKEND` | `[sim] backend` |
| `ALPASANDBOX_REMOTE_URL`, `ALPASANDBOX_TOKEN` | `[runtime.remote]` |
| `ALPASANDBOX_WSL_DISTRO` | `[runtime.wsl] distro` |
| `ALPASANDBOX_SCENARIOS` | extra scenario directories (path-separator separated) |
| `ALPASANDBOX_WIRE_QUALITY` | JPEG quality of frames sent to a worker or server |

## Model settings

Anything under `[model.settings]` (or `--set key=value`) is passed to the model's constructor.

| Edge setting | Default | |
|---|---|---|
| `model_id` | `nvidia/Alpamayo2-Edge` | Hugging Face id or local directory |
| `revision` | latest | pin a Hugging Face revision |
| `num_traj_sets` | 1 | trajectory candidates; extras are drawn as alternatives |
| `diffusion_steps` | 10 | flow-matching steps |
| `temperature`, `top_p`, `seed` | 0.6, 0.98, 43 | reasoning sampling |
| `attn_implementation` | `sdpa` | `flash_attention_2` if installed |
| `max_vqa_tokens` | 384 | answer length cap |

Super accepts `model_id`, `device_map`, `num_traj_sets`, `diffusion_steps`, `temperature`, `top_p`, `seed`, `max_vqa_tokens`, `vqa_temperature`.
