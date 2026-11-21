# Architecture

```text
src/alpasandbox/
  core.py          data types: EgoState, Observation, InputSpec, ModelInfo, PolicyOutput, QueryResult
  rig.py           camera rig, model input profiles, frame and ego-history buffers
  engine.py        Episode (one tick = observe -> act -> control -> step) and the headless Runner
  controller.py    pure pursuit + speed control, using the vehicle's real steering limit
  metrics.py       closed-loop metrics
  scenario.py      YAML scenarios
  backends/        builtin (kinematic bicycle + PIL renderer), carla_backend, opendrive
  models/          base Policy, registry, fixtures, alpamayo_edge, alpamayo_super, runtimes, protocol
  worker.py        stdio worker used by the subprocess and WSL runtimes
  server.py        HTTP model server
  session.py       interactive session: one sim thread, commands in, events out
  app.py           FastAPI app: static UI, REST, WebSocket
  web/             the UI (vanilla JS, no build step)
  doctor.py        environment checks
```

**One thread owns the world.** The session thread is the only one that touches the simulator and the model. Browser commands (drive, pause, step, navigation command, question) go into a queue and are applied between ticks. A question is answered from the current frame buffer while the simulation waits, the same way synchronous CARLA waits for a slow model.

**Inputs come from specs.** `FrameBuffer` keeps about 2 s of every rendered camera. `build_observation(spec)` selects each camera's frames at the spec's offsets and the last *N* ego poses, expressed in the current ego frame. The backend renders only the cameras that some spec, or the camera wall, needs.

**Frames.** Everything inside AlpaSandbox is x forward, y left, z up, yaw counter-clockwise, with the origin at the rear axle. Backends convert at the boundary.

**Events to the browser.** Frames are coalesced per client (only the newest is sent), while other events (answers, logs, state changes) are queued, so a slow browser never stalls the simulation.
