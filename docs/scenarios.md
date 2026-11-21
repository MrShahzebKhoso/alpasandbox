# Scenarios

A scenario is a YAML file. The road runs along +x and the ego starts at x = 0 in the centre of its lane. The oncoming lane is to the left (y = +3.5).

```yaml
name: cut_in
description: A faster car in the left lane cuts in close in front of the ego.
tags: [vehicle, interaction]
duration: 22          # seconds
ego_speed: 9          # m/s at t = 0
nav_text: Continue straight.
road_amplitude: 0     # S-curve amplitude in metres (0 = straight)
road_wavelength: 200
actors:
  - kind: vehicle     # vehicle | pedestrian | cone
    x: 6              # metres along the road
    y: 3.5            # metres left of the ego-lane centre
    vx: 11            # along-road speed; negative = oncoming
    vy: -1.3          # lateral speed, left positive
    start_t: 0        # motion starts
    lateral_start_t: 3.0
    y_target: 0.0     # stop moving sideways here
    decel: 0          # braking along its heading, m/s^2, from decel_start_t
```

Put your own files in a directory and pass `--scenarios-dir DIR` (or set `ALPASANDBOX_SCENARIOS`). Files there override built-ins with the same name. Unknown keys are rejected, so typos fail loudly.

| Built-in | Tests |
|---|---|
| `straight_cruise` | lane keeping, sanity |
| `curvy_road` | lane keeping on S-curves |
| `pedestrian_crossing` | yielding to a crossing pedestrian |
| `occluded_pedestrian` | a pedestrian stepping out from behind a parked car |
| `lead_vehicle_braking` | braking behind a car that stops |
| `slow_lead` | car following |
| `cut_in` | a close cut-in from the left lane |
| `oncoming_traffic` | holding the lane with oncoming cars |
| `cone_obstruction` | a static blocker in the lane |

`alpasandbox run --all --out results.json` runs every scenario and writes the metrics, a quick way to compare models or settings.
