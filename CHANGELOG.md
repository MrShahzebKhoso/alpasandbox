# Changelog

## 0.3.0 — 2026-11-21

### Added
- New app layout: the driving view is the headline, with a tilted 3D chase camera (plan ribbon, lane surfaces, 3D boxes for road users) and a HUD for steering, speed, plan reach and time. A 2D top-down view is one click away.
- Six cameras in a two-column wall beside the driving view, with steering, throttle and brake bars and the speed chart below.
- Separate **Drive instructions** and **Conversation** tabs. Reasoning appears as timed bullets next to the commands you send; the conversation holds only questions and answers.
- CARLA traffic light control in the app (hold red, yellow or green, or Auto) and on the command line (`--signal`, `--signal-at STATE@METRES`), plus `--steps` to limit a run.
- Bird's-eye view draws junction road surfaces and crosswalks.
- Traces include controller and simulator internals (applied control, light state, stop-line distance).

### Changed
- Camera rig set to PhysicalAI-AV fleet medians (see `docs/calibration.md`).
- Static files are served with `Cache-Control: no-cache`, so a restarted app never runs a stale UI.

### Fixed
- In CARLA the car could stay parked on green: VLA plans start moving late in the horizon and the vehicle has a throttle dead band from rest. The controller now reads go and stop from the whole plan, launches with enough throttle, and brakes to the plan's stop point.
- Drift into dividers on straights: pure pursuit now looks ahead by time (1.5 s, 5–30 m), so it sees the model's late lateral corrections.

## 0.2.0 — 2026-11-20

First public release.

### Added
- Web app redesign: camera wall with enlarge, live navigation command, **Ask the model** (free-form scene Q&A while the sim waits), reasoning timeline, speed and braking history, bird's-eye plan with alternative candidates, metrics, log, episode summary, trace download, keyboard shortcuts, light and dark themes.
- Alpamayo 2 Edge VQA through upstream `prepare_vqa_inputs` and `generate_vqa_answer` (six cameras × four frames).
- Experimental Alpamayo 2 Super adapter (seven-camera source ring, upstream task selection, VQA).
- Runtimes: `local`, `subprocess`, `wsl`, `remote`, all behind one `Policy` interface. The stdio worker has per-request timeouts, id-matched replies, and fd-level stdout protection.
- Model registry with `alpasandbox.models` entry points and `module:Class` specs, plus an example plugin.
- `alpasandbox doctor`, `init`, `models`, `run --all`, and `alpasandbox.toml` configuration.
- Scenarios: cut-in, occluded pedestrian, oncoming traffic, slow lead; actor lane changes (`lateral_start_t`, `y_target`).
- CARLA: scenario roads generated with OpenDRIVE, actors placed along the lane in any town, optional autopilot traffic, frame-synchronized sensors, spectator follow.
- CI on Linux and Windows, issue templates, citation file.

### Fixed
- CARLA steering scale: the controller now uses the vehicle's real maximum steering angle and wheelbase. Previously one value was assumed, which over-steered by about 2.4× on vehicles with 70° lock.
- CARLA ego pose and camera mounts are now referenced to the rear axle, not the body centre.
- Pedestrians in CARLA move with `WalkerControl`.
- The WSL worker can no longer hang the app (timeouts) or have its protocol corrupted by native libraries writing to stdout.
- Off-road detection in CARLA uses driving-lane membership instead of a 3 m distance threshold.
- Built-in renderer clips polygons at the near plane, so side cameras show the road beside the car.

## 0.1.0

Initial prototype: builtin backend, Edge driving via HTTP or WSL, basic viewer.
