# Camera calibration

`rig.REFERENCE_RIG` places seven cameras relative to the rear axle at ground level (x forward, y left, z up, yaw counter-clockwise, field of view in degrees). The Alpamayo models were trained on one real rig, so the closer the simulated cameras match it, the closer the model's inputs are to what it learned from.

## Where the numbers come from

The values are medians over the 100 clips in chunk 0000 of the [PhysicalAI-AV](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) calibration tables (`calibration/sensor_extrinsics`, `calibration/camera_intrinsics`, `calibration/vehicle_dimensions`). Extrinsics are camera-to-rig quaternions with the OpenCV optical convention (z forward, x right, y down); yaw and pitch below are the optical axis in the rig frame. Fields of view are solved from each camera's f-theta forward polynomial at the image edge (1920 × 1080).

| Camera | x | y | z | Yaw | Horizontal FOV | Vertical FOV |
|---|---|---|---|---|---|---|
| 0 cross left | 2.52 | 0.94 | 0.90 | 66.8° | 121° | 67° |
| 1 front wide | 1.75 | 0.00 | 1.43 | −0.4° | 121° | 67° |
| 2 cross right | 2.52 | −0.93 | 0.90 | −66.6° | 121° | 67° |
| 3 rear left | 2.00 | 1.05 | 0.90 | 152.6° | 71° | 40° |
| 4 rear tele | 0.43 | 0.31 | 1.46 | 179.4° | 30° | 17° |
| 5 rear right | 1.97 | −1.01 | 0.90 | −151.7° | 71° | 40° |
| 6 front tele | 1.72 | 0.09 | 1.44 | −0.2° | 30° | 17° |

All pitches are within ±1.1°. Yaw varies by under 1° between clips for the front and cross cameras and about 2.3° for the rear corners. The vehicle is about 4.87 m long with a 2.85 m wheelbase; the CARLA default ego (Tesla Model 3) is close to that.

`REFERENCE_RIG` uses these medians with yaw rounded to 1° and pitch set to 0. The cross cameras look about 67° to the side, not 90°, and sit low on the front fenders, as the mirror-mounted rear corners do.

## What is not matched yet

- **Lens model.** The real 120° cameras are f-theta lenses: about 67° vertical field of view. A simulator pinhole camera with a 120° horizontal field of view covers about 89° vertically at 16:9, so it shows more of the bonnet and stretches the image edges. The 30° and 70° cameras are close to pinhole, so the mismatch matters mainly for cameras 0, 1 and 2.
- **Vehicle body.** Camera 1 sees the ego's own bonnet. How much depends on the simulated vehicle model, not just the mount.

## Using your own rig

Build a tuple of `CameraSpec` and pass it as `rig=` to the backend, or edit `REFERENCE_RIG` in a fork. Keep the camera ids: the models use them as camera embeddings.
