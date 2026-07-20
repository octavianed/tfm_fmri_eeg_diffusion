"""Metrics comparing generated images to the real stimulus (spec §11.3).

CLIP similarity / retrieval are the primary (semantic) metrics; pixel MSE,
SSIM and LPIPS are optional and only computed if the libraries are available.
All return per-sample arrays too, so notebooks can pick best/median/worst cases.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from ..features.clip_model import encode_pil_images
from .retrieval_metrics import compute_retrieval_metrics


def _clip_embeddings(clip_bundle, images: List[Image.Image], device):
    return encode_pil_images(clip_bundle, images, device, normalize=True).numpy()


def clip_pairwise_similarity(real_emb: np.ndarray,
                             gen_emb: np.ndarray) -> np.ndarray:
    real = real_emb / (np.linalg.norm(real_emb, axis=1, keepdims=True) + 1e-8)
    gen = gen_emb / (np.linalg.norm(gen_emb, axis=1, keepdims=True) + 1e-8)
    return np.sum(real * gen, axis=1)


def pixel_mse(real_images, gen_images, size: int = 256) -> np.ndarray:
    out = []
    for r, g in zip(real_images, gen_images):
        a = np.asarray(r.convert("RGB").resize((size, size)), np.float32) / 255.0
        b = np.asarray(g.convert("RGB").resize((size, size)), np.float32) / 255.0
        out.append(float(np.mean((a - b) ** 2)))
    return np.asarray(out, dtype=np.float32)


def _maybe_ssim(real_images, gen_images, size: int = 256):
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return None
    vals = []
    for r, g in zip(real_images, gen_images):
        a = np.asarray(r.convert("RGB").resize((size, size)), np.float32) / 255.0
        b = np.asarray(g.convert("RGB").resize((size, size)), np.float32) / 255.0
        vals.append(float(ssim(a, b, channel_axis=2, data_range=1.0)))
    return np.asarray(vals, dtype=np.float32)


def _maybe_lpips(real_images, gen_images, device, size: int = 256):
    try:
        import lpips
        import torch
    except Exception:
        return None
    net = lpips.LPIPS(net="alex").to(device)

    def to_tensor(img):
        arr = np.asarray(img.convert("RGB").resize((size, size)), np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1) * 2 - 1
        return t.unsqueeze(0).to(device)

    vals = []
    with torch.no_grad():
        for r, g in zip(real_images, gen_images):
            vals.append(float(net(to_tensor(r), to_tensor(g)).item()))
    return np.asarray(vals, dtype=np.float32)


def compute_generation_metrics(real_images: List[Image.Image],
                               gen_images: List[Image.Image],
                               clip_bundle, device, ks=(1, 5),
                               use_ssim: bool = False, use_lpips: bool = False,
                               pixel_size: int = 256) -> dict:
    real_emb = _clip_embeddings(clip_bundle, real_images, device)
    gen_emb = _clip_embeddings(clip_bundle, gen_images, device)

    per_clip = clip_pairwise_similarity(real_emb, gen_emb)
    retrieval, ranks = compute_retrieval_metrics(gen_emb, real_emb, ks=ks)

    metrics = {
        "mean_clip_similarity": float(per_clip.mean()),
        "median_clip_similarity": float(np.median(per_clip)),
        "clip_retrieval": retrieval,
    }
    per_sample = {"clip_similarity": per_clip, "retrieval_rank": ranks}

    per_sample["pixel_mse"] = pixel_mse(real_images, gen_images, pixel_size)
    metrics["mean_pixel_mse"] = float(per_sample["pixel_mse"].mean())
    if use_ssim:
        s = _maybe_ssim(real_images, gen_images, pixel_size)
        if s is not None:
            per_sample["ssim"] = s
            metrics["mean_ssim"] = float(s.mean())
    if use_lpips:
        lp = _maybe_lpips(real_images, gen_images, device, pixel_size)
        if lp is not None:
            per_sample["lpips"] = lp
            metrics["mean_lpips"] = float(lp.mean())

    return {"metrics": metrics, "per_sample": per_sample,
            "real_emb": real_emb, "gen_emb": gen_emb}
