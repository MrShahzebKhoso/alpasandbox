"""NVIDIA Alpamayo 2 Edge (2B reasoning VLA): driving + free-form VQA.

Requires a CUDA GPU (>= 16 GB recommended), Python 3.12 and the upstream package:
    pip install "alpamayo2-edge @ git+https://github.com/NVlabs/alpamayo2-edge"
Call pattern follows NVlabs/alpamayo2-edge (checkpoint.load_model, helper.prepare_model_inputs,
helper.prepare_vqa_inputs, text_tasks.generate_vqa_answer).
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core import ModelInfo, Observation, PolicyOutput, QueryResult
from ..rig import EDGE_DRIVE_SPEC, EDGE_VQA_SPEC
from ..wire import check_frames, stamps_us, t0_us
from .base import Policy, clean_reasoning, flatten_text

DRIVE_PIXEL_BOUNDS = {c: ((163840, 196608) if c in (1, 6) else (81920, 98304)) for c in (0, 1, 2, 3, 5, 6)}
VQA_PIXEL_BOUNDS = {c: (163840, 196608) for c in (0, 1, 2, 3, 5, 6)}

DEFAULTS: dict[str, Any] = dict(
    model_id="nvidia/Alpamayo2-Edge", revision=None, device="cuda", dtype="bfloat16",
    attn_implementation="sdpa", seed=43, top_p=0.98, top_k=None, temperature=0.6,
    num_traj_sets=1, num_traj_samples=1, max_generation_length=256, diffusion_steps=10,
    max_vqa_tokens=384, accel=None,   # accel: None | "compile" | "tuned" (NVIDIA alpamayo2_edge.accel; needs flash-attn)
)

INFO = ModelInfo(
    "edge", "Alpamayo 2 Edge", EDGE_DRIVE_SPEC, EDGE_VQA_SPEC,
    description="2B reasoning VLA for in-vehicle use: 64-waypoint trajectory, adaptive "
                "Chain-of-Causation reasoning, free-form scene Q&A.",
    homepage="https://huggingface.co/nvidia/Alpamayo2-Edge", license="OpenMDW-1.1 (weights)",
)


def _arrays(obs: Observation, cameras: dict[int, tuple[float, ...]]) -> dict[str, np.ndarray]:
    """Camera-major uint8 [N, 1, 3, H, W] images with int64 microsecond timestamps."""
    check_frames(obs, cameras)
    imgs, cam_idx, stamps = [], [], []
    for cid in sorted(cameras):
        for frame, ts in zip(obs.frames[cid], stamps_us(obs, cid), strict=False):
            imgs.append(np.ascontiguousarray(frame.transpose(2, 0, 1))[None])
            cam_idx.append(cid)
            stamps.append(ts)
    return {"image_frames": np.stack(imgs).astype(np.uint8),
            "camera_indices": np.asarray(cam_idx, np.int64),
            "absolute_timestamps": np.asarray(stamps, np.int64)}


def edge_drive_arrays(obs: Observation) -> dict[str, Any]:
    """Pure-numpy version of the upstream driving sample (22 images, 20 poses)."""
    a = _arrays(obs, EDGE_DRIVE_SPEC.cameras)
    if obs.ego_xyz.shape != (20, 3):
        raise ValueError(f"Edge needs 20 ego poses, got {obs.ego_xyz.shape}")
    nav = obs.nav_text.strip() or "Continue straight."
    return {**a, "ego_history_xyz": obs.ego_xyz[None].astype(np.float32),
            "ego_history_rot": obs.ego_rot[None].astype(np.float32),
            "nav_text": [nav], "t0_us": t0_us(obs)}


def edge_vqa_arrays(obs: Observation) -> dict[str, Any]:
    """Pure-numpy version of the upstream VQA sample (6 cameras x 4 frames)."""
    return {**_arrays(obs, EDGE_VQA_SPEC.cameras), "t0_us": t0_us(obs)}


class AlpamayoEdge(Policy):
    def __init__(self, **settings):
        import torch
        from alpamayo2_edge.checkpoint import load_model

        unknown = set(settings) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown Edge settings: {sorted(unknown)}; valid: {sorted(DEFAULTS)}")
        self.s = {**DEFAULTS, **settings}
        self.torch, self.device = torch, self.s["device"]
        kw = {"revision": self.s["revision"]} if self.s["revision"] else {}
        self.model, _ = load_model(self.s["model_id"], device=self.device, dtype=self.s["dtype"],
                                   attn_implementation=self.s["attn_implementation"], **kw)
        self.accel = None
        if self.s["accel"]:
            torch.set_float32_matmul_precision("high")   # TF32 for the few fp32 matmuls inductor emits
            from alpamayo2_edge.accel import AccelConfig, enable_acceleration
            self.accel = enable_acceleration(self.model, AccelConfig(mode=self.s["accel"]))
            print(f"[edge] acceleration '{self.s['accel']}': {self.accel.describe()} "
                  "(the first plan compiles graphs and is slow)", flush=True)
        self._info = ModelInfo(**{**INFO.__dict__, "extra": {"settings": {k: v for k, v in self.s.items()}}})

    @property
    def info(self) -> ModelInfo:
        return self._info

    def _tensors(self, a: dict, pixel_bounds: dict, profile: str) -> dict:
        torch = self.torch
        d = {k: torch.from_numpy(v).contiguous() if isinstance(v, np.ndarray) else v for k, v in a.items()}
        lo, hi = min(b[0] for b in pixel_bounds.values()), max(b[1] for b in pixel_bounds.values())
        d.update(camera_pixel_bounds=dict(pixel_bounds), min_pixels=lo, max_pixels=hi,
                 input_profile=profile, clip_id="alpasandbox",
                 provenance={"dataset": "alpasandbox-simulation", "clip_id": "alpasandbox", "t0_us": a["t0_us"]})
        return d

    def act(self, obs: Observation) -> PolicyOutput:
        from alpamayo2_edge.helper import prepare_model_inputs, to_device
        from alpamayo2_edge.sample import validate_sample

        torch, s, t0 = self.torch, self.s, time.perf_counter()
        data = self._tensors(edge_drive_arrays(obs), DRIVE_PIXEL_BOUNDS, "camera_subsampled")
        data["history_num_steps"] = 20
        validate_sample(data)
        inputs = to_device(prepare_model_inputs(data, self.model.config, self.model.tokenizer), device=self.device)
        torch.manual_seed(s["seed"])
        gen = torch.Generator(device=self.device).manual_seed(s["seed"])
        dev_type = torch.device(self.device).type
        with torch.inference_mode(), torch.autocast(device_type=dev_type, dtype=self.model.dtype,
                                                    enabled=self.model.dtype != torch.float32):
            xyz, _rot, _, extra = self.model.sample_trajectories_from_data(
                inputs, num_traj_sets=s["num_traj_sets"], num_traj_samples=s["num_traj_samples"],
                top_p=s["top_p"], top_k=s["top_k"], temperature=s["temperature"],
                max_generation_length=s["max_generation_length"],
                diffusion_kwargs={"inference_step": s["diffusion_steps"], "generator": gen},
                return_extra=True)
        trajs = xyz[0].reshape(-1, xyz.shape[-2], xyz.shape[-1])[..., :2].float().cpu().numpy()  # (K, 64, 2)
        cots = flatten_text(extra.get("cot"))
        reasons = [clean_reasoning(c)[0] for c in cots] or [""]
        reasoning, nominal = clean_reasoning(cots[0] if cots else "")
        return PolicyOutput(
            np.asarray(trajs[0], float), 0.1, reasoning, (time.perf_counter() - t0) * 1000,
            {"nominal": nominal, "num_candidates": int(trajs.shape[0]), "raw_cot": (cots[0] if cots else "")[:400]},
            candidates=[np.asarray(t, float) for t in trajs[1:]], candidate_reasoning=reasons[1:])

    def query(self, obs: Observation, question: str) -> QueryResult:
        from alpamayo2_edge.helper import prepare_vqa_inputs, to_device
        from alpamayo2_edge.text_tasks import generate_vqa_answer

        t0 = time.perf_counter()
        data = self._tensors(edge_vqa_arrays(obs), VQA_PIXEL_BOUNDS, "vqa_6cam_4f")
        inputs = to_device(prepare_vqa_inputs(data, self.model.config, self.model.tokenizer, question),
                           device=self.device)
        torch = self.torch
        with torch.autocast(device_type=torch.device(self.device).type, dtype=self.model.dtype,
                            enabled=self.model.dtype != torch.float32):
            answer = generate_vqa_answer(self.model, inputs, max_new_tokens=int(self.s["max_vqa_tokens"]))
        return QueryResult(answer, (time.perf_counter() - t0) * 1000)

    def close(self) -> None:
        self.model = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass
