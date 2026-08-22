"""Spatial ControlNet conditions derived from the low-level (VAE-PCA) branch (§6, §8).

Pipeline, identical at training and inference except for where the PCA vector
comes from::

    VAE-PCA vector -> inverse PCA -> VAE latent -> VAE decode -> coarse RGB
                   -> Canny / Depth -> ControlNet condition image

At **inference** the vector is ``low_pred``, the brain prediction of Exp3. At
**training** it is the *ground-truth* PCA vector of the real image
(``training_condition_source: gt_vae_pca_reconstruction``). Training never uses
``GT image -> Canny`` directly: that would hand ControlNet an essentially perfect
outline it will never see at inference, where the structure arrives blurred by
the brain decoder *and* by the PCA bottleneck. Passing the GT through the same
bottleneck shrinks that train/inference gap (§8).

Canny is computed with scikit-image (already a project dependency, used for
SSIM) — no OpenCV/controlnet_aux needed. Thresholds default to **quantiles of
the gradient magnitude** rather than absolute values, because the PCA
reconstruction is much smoother than a natural photo and fixed absolute
thresholds would return an almost empty edge map for it.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from ..features.load_features import (inverse_pca_to_latent, load_pca_bundle,
                                      load_split_features)
from ..utils import controlnet_condition_dir, get_logger

logger = get_logger("controlnet_cond")

CONDITION_TYPES = ("canny", "depth")


# --- condition extractors ---------------------------------------------------
def canny_edges(image: Image.Image, sigma: float = 1.0,
                low_threshold: float = 0.85, high_threshold: float = 0.95,
                use_quantiles: bool = True) -> Image.Image:
    """Canny edge map as a 3-channel white-on-black image (ControlNet's format)."""
    from skimage.feature import canny

    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    edges = canny(gray, sigma=float(sigma), low_threshold=float(low_threshold),
                  high_threshold=float(high_threshold),
                  use_quantiles=bool(use_quantiles))
    arr = (edges.astype(np.uint8) * 255)
    return Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB")


def depth_map(images: Sequence[Image.Image], model_name: str, device=None
              ) -> List[Image.Image]:
    """Monocular depth as a 3-channel image (optional ablation, §6.2)."""
    from transformers import pipeline
    dev = 0 if (device is not None and getattr(device, "type", "") == "cuda") else -1
    estimator = pipeline("depth-estimation", model=model_name, device=dev)
    out = []
    for img in images:
        depth = np.asarray(estimator(img)["depth"], dtype=np.float32)
        lo, hi = float(depth.min()), float(depth.max())
        norm = (depth - lo) / max(1e-6, hi - lo)
        arr = (norm * 255).astype(np.uint8)
        out.append(Image.fromarray(np.stack([arr] * 3, axis=-1), mode="RGB"))
    return out


def build_control_images(cfg, coarse_images: Sequence[Image.Image], device=None
                         ) -> List[Image.Image]:
    """Apply the configured processor to coarse reconstructions."""
    kind = str(cfg.get("generation.controlnet.condition_type", "canny")).lower()
    if kind == "canny":
        return [canny_edges(
            img,
            sigma=float(cfg.get("generation.controlnet.canny.sigma", 1.0)),
            low_threshold=float(cfg.get("generation.controlnet.canny.low_threshold",
                                        0.85)),
            high_threshold=float(cfg.get("generation.controlnet.canny.high_threshold",
                                         0.95)),
            use_quantiles=bool(cfg.get("generation.controlnet.canny.use_quantiles",
                                       True)))
            for img in coarse_images]
    if kind == "depth":
        model = cfg.get("generation.controlnet.depth_model",
                        "Intel/dpt-hybrid-midas")
        return depth_map(coarse_images, str(model), device)
    raise ValueError(f"generation.controlnet.condition_type must be one of "
                     f"{CONDITION_TYPES}, got {kind!r}")


# --- coarse reconstruction --------------------------------------------------
def coarse_images_from_pca(cfg, decode_fn, subject: str,
                           pca_vectors: np.ndarray,
                           bundle_cache: Optional[dict] = None
                           ) -> List[Image.Image]:
    """``PCA vector -> inverse PCA -> VAE decode -> PIL`` for one subject."""
    cache = bundle_cache if bundle_cache is not None else {}
    if subject not in cache:
        cache[subject] = load_pca_bundle(cfg, subject)
    latents = inverse_pca_to_latent(cache[subject], np.asarray(pca_vectors,
                                                               dtype=np.float32))
    return decode_fn(latents)


def control_images_from_lowlevel(cfg, decode_fn, selection: Sequence[dict],
                                 low_vectors: np.ndarray, device=None,
                                 bundle_cache: Optional[dict] = None
                                 ) -> List[Image.Image]:
    """Inference path: ``low_pred`` (per selected sample) -> ControlNet condition."""
    cache = bundle_cache if bundle_cache is not None else {}
    coarse: List[Image.Image] = []
    for i, sample in enumerate(selection):
        coarse.extend(coarse_images_from_pca(cfg, decode_fn, sample["subject"],
                                             low_vectors[i:i + 1], cache))
    return build_control_images(cfg, coarse, device)


# --- training-condition cache ----------------------------------------------
def condition_path(cfg, subject: str, split: str, feat_idx: int) -> Path:
    kind = str(cfg.get("generation.controlnet.condition_type", "canny")).lower()
    return controlnet_condition_dir(cfg, kind, subject, split) / f"{int(feat_idx):06d}.png"


def precompute_controlnet_conditions(cfg, datamodule, splits=("train", "val", "test"),
                                     overwrite: bool = False, device=None,
                                     batch_size: int = 8,
                                     limit: Optional[int] = None) -> dict:
    """Cache the ControlNet condition of every image's **GT** VAE-PCA reconstruction.

    Needed to train the TokenAdapter of the ControlNet architecture without a
    VAE forward per step. Requires ``scripts/04_precompute_vae_pca.py`` to have
    run (it reads the fitted PCA features, not the images).

    Depends only on the images + PCA, never on the brain data: one cache serves
    every EEG preprocessing variant, exactly like the CLIP/VAE features.
    """
    import time

    import torch

    from ..data.image_transforms import tensor_to_pil
    from ..features.precompute_vae_latents import load_vae
    from ..utils import get_device

    device = device or get_device(cfg.get("runtime.device", "auto"))
    # fp16 + slicing: decoding to 512x512 in fp32 with a batch of 16 was pushing
    # the allocator into shared system RAM (~1 image/s instead of tens).
    vae = load_vae(cfg, device,
                   dtype=torch.float16 if device.type == "cuda" else None,
                   slicing=True)

    def decode(latents: np.ndarray) -> List[Image.Image]:
        z = torch.as_tensor(latents, dtype=vae.dtype, device=device)
        with torch.no_grad():
            imgs = vae.decode(z / vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1).float().cpu()
        return [tensor_to_pil(img) for img in imgs]

    results = {}
    bundles: dict = {}
    for subject in datamodule.subjects:
        for split in splits:
            frame = datamodule.subject_split_frame(subject, split)
            if len(frame) == 0:
                continue
            pca = load_split_features(cfg, subject, split, "low")
            if pca is None:
                raise FileNotFoundError(
                    f"No VAE-PCA features for {subject}/{split}. Run "
                    f"scripts/04_precompute_vae_pca.py first — the ControlNet "
                    f"training condition is derived from them (§8).")
            n = len(pca) if not limit else min(int(limit), len(pca))
            out_dir = controlnet_condition_dir(
                cfg, str(cfg.get("generation.controlnet.condition_type", "canny")),
                subject, split)
            # Resume at file granularity: a cache interrupted half-way (this can
            # take hours) must not restart from zero.
            todo = [i for i in range(n)
                    if overwrite or not condition_path(cfg, subject, split, i).exists()]
            if not todo:
                logger.info("[skip] %s/%s already complete (%d conditions): %s",
                            subject, split, n, out_dir)
                results[(subject, split)] = n
                continue
            if len(todo) < n:
                logger.info("[resume] %s/%s: %d/%d already cached, %d to go",
                            subject, split, n - len(todo), n, len(todo))
            t0, done = time.time(), 0
            for start in range(0, len(todo), batch_size):
                idx = todo[start:start + batch_size]
                coarse = coarse_images_from_pca(cfg, decode, subject,
                                                pca[idx], bundles)
                for i, img in zip(idx, build_control_images(cfg, coarse, device)):
                    img.save(condition_path(cfg, subject, split, i))
                done += len(idx)
                if done % max(batch_size, 512) < batch_size or done == len(todo):
                    rate = done / max(1e-6, time.time() - t0)
                    eta = (len(todo) - done) / max(1e-6, rate) / 60
                    logger.info("  %s/%s %d/%d (%.1f img/s, ETA %.1f min)",
                                subject, split, done, len(todo), rate, eta)
            results[(subject, split)] = n
            logger.info("[controlnet] %s/%s -> %d conditions in %s",
                        subject, split, n, out_dir)
    return results


class ControlConditionCache:
    """Reads cached condition PNGs by ``feat_idx`` (adapter training)."""

    def __init__(self, cfg, subject_splits: Sequence[tuple]):
        self.cfg = cfg
        self.pairs = list(subject_splits)
        missing = [(s, sp) for s, sp in self.pairs
                   if not condition_path(cfg, s, sp, 0).exists()]
        if missing:
            raise FileNotFoundError(
                f"No cached ControlNet conditions for {missing}. Run "
                f"scripts/14_precompute_controlnet_conditions.py with this config.")

    def load(self, subject: str, split: str, feat_idx: int) -> Image.Image:
        return Image.open(condition_path(self.cfg, subject, split, feat_idx)).convert("RGB")

    def batch_tensor(self, keys: Sequence[tuple], size: int, device, dtype):
        """``[B, 3, size, size]`` in ``[0, 1]`` — the layout diffusers expects."""
        import torch
        arrs = []
        for subject, split, feat_idx in keys:
            img = self.load(subject, split, int(feat_idx))
            if img.size != (size, size):
                img = img.resize((size, size), Image.NEAREST)
            arrs.append(np.asarray(img, dtype=np.float32) / 255.0)
        arr = np.stack(arrs).transpose(0, 3, 1, 2)
        return torch.from_numpy(arr).to(device=device, dtype=dtype)


def save_condition_qc(images: Sequence[Image.Image], coarse: Sequence[Image.Image],
                      out_path) -> Optional[str]:
    """Side-by-side coarse reconstruction vs extracted condition (visual check)."""
    if not images:
        return None
    from .make_grids import save_comparison_grid
    return save_comparison_grid({"coarse": list(coarse), "condition": list(images),
                                 "image_ids": [str(i) for i in range(len(images))]},
                                out_path, column_order=("coarse", "condition"))
