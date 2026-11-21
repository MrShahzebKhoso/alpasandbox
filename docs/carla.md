# CARLA

AlpaSandbox drives CARLA 0.9.14 or newer through the 0.9.x Python API (0.9.16 is current). The `carla` client package must match the server version exactly: `pip install carla==0.9.16`. CARLA needs TCP ports 2000 and 2001.

```bash
alpasandbox app --backend carla --model edge
```

## Maps

| `--carla-town` | What happens |
|---|---|
| `scenario` (default) | Builds the scenario's road (straight or S-curves) with OpenDRIVE, so scenarios look the same as in the built-in backend. The road is rebuilt only when the scenario's geometry changes. |
| `Town10HD_Opt`, `Town04`, ... | Loads that town. The ego spawns at `[carla] spawn_index`, and scenario actors are placed along the ego's lane, so they follow curves. Add `--traffic 30` for autopilot traffic. |
| `current` | Uses whatever map the server already has loaded. |

## Coordinate frames and vehicle geometry

CARLA is left-handed (y right, yaw clockwise). AlpaSandbox converts to x forward, y left, yaw counter-clockwise, and moves the ego reference from CARLA's actor origin (body centre) to the **rear axle**, measured from the wheel positions. Camera mounts in `rig.py` are given relative to the rear axle and are attached accordingly. The controller reads the vehicle's real wheelbase and maximum steering angle from its physics, so `steer = 1.0` means the same physical angle in the controller and in CARLA.

Road departure means the ego is no longer on a driving lane (`project_to_road=False`). Lane offset is signed (left positive) relative to the nearest lane centre.

## Pedestrians and scripted actors

Pedestrians are moved with `WalkerControl` (direction and speed), vehicles with target velocities. `lateral_start_t` and `y_target` script lane changes (see `cut_in.yaml`).

## CARLA on Windows, AlpaSandbox in WSL

If you run AlpaSandbox itself inside WSL, use the Windows host IP as `--carla-host` (`ip route | awk '/default/ {print $3}'`, or `localhost` with WSL mirrored networking) and allow CarlaUE4 through the Windows firewall. The simpler setup is to run AlpaSandbox on Windows and only the model in WSL ([installation B](installation.md#b-windows--wsl2-carla-on-windows-model-in-linux)).
