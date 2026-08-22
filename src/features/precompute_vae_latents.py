"""Precompute frozen Stable Diffusion VAE latents for the low-level branch.

Images are encoded to SD latents ``z = vae.encode(x).mean * scaling_factor``
(the standard SD scaling), flattened and saved as ``[N, C*H*W]`` per
(subject, split). PCA is fitted separately (train only) in
:mod:`src.features.fit_vae_pca` (spec §9.2). A small ``*_vae_meta.json`` records
the latent shape and scaling factor for later reconstruction (generation
Option C).
"""
from __future__ import annotations

import json

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None

from ..data.image_transforms import build_vae_preprocess
from ..utils import autocast, get_device, get_logger, vae_latent_path
from .precompute_clip_embeddings import ImagePathDataset

logger = get_logger("precompute_vae")


def load_vae(cfg, device, dtype=None, slicing: bool = False):
    """Frozen SD VAE.

    ``dtype=None`` keeps the historical float32 load — do NOT change it for the
    *encoding* path (``precompute_vae_latents``), because the cached latents and
    the PCA fitted on them were produced in float32.

    Decoding (64x64 -> 512x512) is a different story: it is ~10x heavier than
    encoding and in float32 it is what made the ControlNet-condition precompute
    thrash. Pass ``dtype=torch.float16`` + ``slicing=True`` there — it also
    matches inference, where the SD pipeline runs its VAE in fp16 on CUDA.
    """
    from diffusers import AutoencoderKL
    name = str(cfg.get("features.vae_model",
                       "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    try:
        vae = AutoencoderKL.from_pretrained(name, subfolder="vae", **kwargs)
    except Exception:
        vae = AutoencoderKL.from_pretrained(name, **kwargs)
    vae.eval().to(device)
    for p in vae.parameters():
        p.requires_grad_(False)
    if slicing:
        # Decode one image at a time internally: peak memory stops depending on
        # the batch size, so a large features.vae_batch_size can no longer push
        # the allocator into system RAM.
        vae.enable_slicing()
    return vae


def _meta_path(cfg, subject):
    return vae_latent_path(cfg, subject, "train").parent / f"{subject}_vae_meta.json"


def precompute_vae_latents(cfg, datamodule, splits=("train", "val", "test"),
                           overwrite: bool = False) -> dict:
    device = get_device(cfg.get("runtime.device", "auto"))
    vae = load_vae(cfg, device)
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    image_size = int(cfg.get("features.vae_image_size", 512))
    transform = build_vae_preprocess(image_size)
    batch_size = int(cfg.get("features.vae_batch_size", 16))
    num_workers = int(cfg.get("training.num_workers", 4))

    results = {}
    for subj in datamodule.subjects:
        latent_shape = None
        for split in splits:
            frame = datamodule.subject_split_frame(subj, split)
            if len(frame) == 0:
                continue
            out_path = vae_latent_path(cfg, subj, split)
            if out_path.exists() and not overwrite:
                logger.info("[skip] exists: %s", out_path)
                results[(subj, split)] = tuple(np.load(out_path, mmap_mode="r").shape)
                continue
            ds = ImagePathDataset(frame["image_path"].tolist(), transform)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=device.type == "cuda")
            latents = []
            with torch.no_grad():
                for imgs in loader:
                    imgs = imgs.to(device, non_blocking=True)
                    with autocast(device, enabled=device.type == "cuda"):
                        z = vae.encode(imgs).latent_dist.mean * scaling
                    if latent_shape is None:
                        latent_shape = tuple(int(s) for s in z.shape[1:])
                    latents.append(z.float().reshape(z.shape[0], -1).cpu().numpy())
            arr = np.concatenate(latents, axis=0).astype(np.float32)
            np.save(out_path, arr)
            results[(subj, split)] = arr.shape
            logger.info("[vae] %s/%s -> %s %s", subj, split, out_path, arr.shape)
        if latent_shape is not None:
            meta = {"scaling_factor": scaling, "latent_shape": list(latent_shape),
                    "image_size": image_size,
                    "vae_model": str(cfg.get("features.vae_model", ""))}
            _meta_path(cfg, subj).write_text(json.dumps(meta, indent=2),
                                             encoding="utf-8")
    return results


def load_vae_meta(cfg, subject) -> dict:
    p = _meta_path(cfg, subject)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    image_size = int(cfg.get("features.vae_image_size", 512))
    return {"scaling_factor": float(cfg.get("features.vae_scaling_factor", 0.18215)),
            "latent_shape": [4, image_size // 8, image_size // 8],
            "image_size": image_size}
