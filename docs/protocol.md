# Model protocols

## HTTP (`alpasandbox serve-model`)

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/healthz` | | `{ok, model, version}` |
| GET | `/v1/info` | | `ModelInfo` (input specs, `supports_vqa`) |
| POST | `/v1/reset` | | `{ok}` |
| POST | `/v1/drive` | Observation | `{waypoints, dt, reasoning, latency_ms, meta, candidates, candidate_reasoning}` |
| POST | `/v1/query` | `{observation, question}` | `{answer, latency_ms, meta}` |

With `ALPASANDBOX_TOKEN` set on the server, every `/v1` call needs `Authorization: Bearer <token>`.

An **Observation** is:

```json
{
  "t": 4.2,
  "nav_text": "Turn right in 30m.",
  "ego_xyz": [[-15.1, 0.0, 0.0], "... N rows, oldest first, current pose = [0, 0, 0]"],
  "ego_rot": ["... N 3x3 matrices"],
  "cameras": { "1": { "offsets": [-2.0, -1.0, -0.5, -0.1, 0.0], "frames": ["<base64 JPEG>", "..."] } }
}
```

Frames are JPEG quality 95 (`ALPASANDBOX_WIRE_QUALITY`). Simulator ground truth is never serialized.

## Stdio worker (`python -m alpasandbox.worker`)

Used by the `subprocess` and `wsl` runtimes. One JSON object per line.

```text
worker -> {"type":"ready","success":true,"info":{...ModelInfo}}      (or success:false, error, traceback)
client -> {"id":1,"op":"predict","observation":{...}}
worker -> {"id":1,"success":true,"output":{...same as /v1/drive}}
client -> {"id":2,"op":"query","observation":{...},"question":"..."}
worker -> {"id":2,"success":true,"output":{"answer":"..."}}
client -> {"id":3,"op":"reset"} | {"id":4,"op":"ping"} | {"op":"shutdown"}
```

Logs go to stderr. The client matches replies by `id` and drops stale replies to requests that timed out.
