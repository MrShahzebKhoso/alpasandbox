"""NVIDIA Alpamayo 2 Super (34B multi-task VLA): driving + VQA. EXPERIMENTAL in alpasandbox.

Needs a large GPU (NVIDIA measured ~72 GB peak on one H100 80GB), Linux, Python 3.12 and:
    pip install "alpamayo2-super @ git+https://github.com/NVlabs/alpamayo2"
Super is a cloud/teacher model, not a 10 Hz in-vehicle model, so run it with
`--runtime remote` on a GPU server and expect seconds per step.

The adapter builds the upstream seven-camera "source ring" (4 frames each, 16 ego poses) and
lets upstream `select_task_input` pick each task's cameras, so camera selection stays upstream's.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core import ModelInfo, Observation, PolicyOutput, QueryResult
from ..rig import SUPER_SPEC
from ..wire import check_frames, stamps_us, t0_us
from .base import Policy, clean_reasoning, flatten_text

CANONICAL_NAMES = ("camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov",
                   "camera_rear_left_70fov", "camera_rear_tele_30fov", "camera_rear_right_70fov",
                   "camera_front_tele_30fov")

DEFAULTS: dict[str, Any] = dict(
    model_id="nvidia/Alpamayo2-Super", device_map="cuda:0", dtype="bfloat16", seed=42,
    top_p=0.98, temperature=0.6, num_traj_sets=1, num_traj_samples=1, diffusion_steps=10,
    max_vqa_tokens=512, vqa_temperature=0.2,
)

INFO = ModelInfo(
    "super", "Alpamayo 2 Super", SUPER_SPEC, SUPER_SPEC,
    description="34B multi-task driving foundation model (trajectory, CoC reasoning, VQA). "
                "Teacher-scale: needs ~80 GB GPU; experimental adapter.",
    homepage="https://huggingface.co/nvidia/Alpamayo2-Super", license="OpenMDW-1.1 (weights)",
    extra={"experimental": True},
)


def super_source_arrays(obs: Observation) -> dict[str, Any]:
    """Pure-numpy upstream source sample: images (7, 4, 3, H, W), timestamps (7, 4), 16 poses."""
    check_frames(obs, SUPER_SPEC.cameras)
    frames = np.stack([np.stack([f.transpose(2, 0, 1) for f in obs.frames[c]]) for c in range(7)])
    abs_ts = np.asarray([stamps_us(obs, c) for c in range(7)], np.int64)
    tmin, t0 = int(abs_ts.min()), t0_us(obs)
    if obs.ego_xyz.shape != (16, 3):
        raise ValueError(f"Super needs 16 ego poses, got {obs.ego_xyz.shape}")
    return {
        "image_frames": np.ascontiguousarray(frames.astype(np.uint8)),
        "camera_indices": np.arange(7, dtype=np.int64),
        "camera_names": list(CANONICAL_NAMES),
        "absolute_timestamps": abs_ts,
        "relative_timestamps": ((abs_ts - tmin) * 1e-6).astype(np.float32),
        "camera_tmin": tmin,
        "ego_available": np.asarray(True),
        "ego_t0": np.asarray([t0], np.int64),
        "ego_t0_relative": np.asarray([(t0 - tmin) * 1e-6], np.float32),
        "ego_t0_frame_idx": np.asarray([3], np.int64),
        "prediction_start_offset": np.zeros(1, np.float32),
        "ego_history_tvals": (np.arange(-15, 1, dtype=np.float32) * 0.1),
        "ego_history_xyz": obs.ego_xyz[None, None].astype(np.float32),
        "ego_history_rot": obs.ego_rot[None, None].astype(np.float32),
        "nav_text": [obs.nav_text.strip() or "Continue straight."],
        "t0_us": t0, "clip_id": "alpasandbox",
    }


class AlpamayoSuper(Policy):
    def __init__(self, **settings):
        import torch
        from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown Super settings: {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
        self.s = {**DEFAULTS, **settings}
        self.torch = torch
        self.model = Alpamayo2Super.from_pretrained(self.s["model_id"], dtype=getattr(torch, self.s["dtype"]),
                                                    device_map=self.s["device_map"])
        self.device = next(self.model.parameters()).device
        self._info = ModelInfo(**{**INFO.__dict__, "extra": {**INFO.extra, "settings": dict(self.s)}})

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _source(self, obs: Observation) -> dict:
        torch = self.torch
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                for k, v in super_source_arrays(obs).items()}

    def act(self, obs: Observation) -> PolicyOutput:
        from alpamayo2_super import helper
        from alpamayo2_super.input_profiles import select_task_input

        torch, s, t0 = self.torch, self.s, time.perf_counter()
        data = select_task_input(self._source(obs), "trajectory")
        inputs = helper.to_device(helper.prepare_model_inputs(data, self.model.config, self.model.tokenizer),
                                  self.device)
        torch.cuda.manual_seed_all(s["seed"])
        with torch.inference_mode(), torch.autocast("cuda", dtype=getattr(torch, s["dtype"])):
            xyz, _rot, _lp, extra = self.model.sample_trajectories_from_data(
                data=inputs, top_p=s["top_p"], temperature=s["temperature"],
                num_traj_sets=s["num_traj_sets"], num_traj_samples=s["num_traj_samples"],
                diffusion_kwargs={"inference_step": s["diffusion_steps"]}, return_extra=True)
        trajs = xyz[0].reshape(-1, xyz.shape[-2], xyz.shape[-1])[..., :2].float().cpu().numpy()
        cots = flatten_text(extra.get("cot"))
        metas = flatten_text(extra.get("meta_action"))
        reasoning, nominal = clean_reasoning(cots[0] if cots else "")
        return PolicyOutput(
            np.asarray(trajs[0], float), 0.1, reasoning, (time.perf_counter() - t0) * 1000,
            {"nominal": nominal, "meta_action": metas[0] if metas else "", "num_candidates": int(trajs.shape[0])},
            candidates=[np.asarray(t, float) for t in trajs[1:]],
            candidate_reasoning=[clean_reasoning(c)[0] for c in cots[1:]])

    def query(self, obs: Observation, question: str) -> QueryResult:
        from alpamayo2_super import helper
        from alpamayo2_super.input_profiles import select_task_input
        from alpamayo2_super.text_tasks import generate_text, prepare_vqa_inputs

        torch, s, t0 = self.torch, self.s, time.perf_counter()
        data = select_task_input(self._source(obs), "vqa")
        inputs = helper.to_device(prepare_vqa_inputs(data, self.model.config, self.model.tokenizer, question),
                                  self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=getattr(torch, s["dtype"])):
            out = generate_text(self.model, inputs, temperature=s["vqa_temperature"],
                                max_new_tokens=int(s["max_vqa_tokens"]))
        answers = flatten_text(out.get("answer"))
        return QueryResult(answers[0].strip() if answers else "", (time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        self.model = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
