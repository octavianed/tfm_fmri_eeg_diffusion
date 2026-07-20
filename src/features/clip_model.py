"""Frozen CLIP / OpenCLIP loading and image encoding.

The CLIP image encoder produces the semantic *targets* the fMRI decoder learns
to predict, and is reused at evaluation time for generation metrics. It is
always frozen (spec §4). The default ``ViT-L-14 / openai`` yields 768-d image
embeddings, which also match the ``sd-image-variations`` generator used by
default in :mod:`src.generation` — so predicted embeddings can drive generation
without an adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..data.image_transforms import load_image
from ..utils import autocast, get_logger

logger = get_logger("clip")


@dataclass
class ClipBundle:
    model: object
    preprocess: object
    embed_dim: int
    name: str
    pretrained: str
    backend: str  # 'open_clip' or 'transformers'


def load_clip(cfg, device, eval_preprocess: bool = True) -> ClipBundle:
    """Load a frozen CLIP model. Prefers ``open_clip``; falls back to HF."""
    name = str(cfg.get("features.clip_model", "ViT-L-14"))
    pretrained = str(cfg.get("features.clip_pretrained", "openai"))
    try:
        import open_clip
        model, pre_train, pre_eval = open_clip.create_model_and_transforms(
            name, pretrained=pretrained)
        preprocess = pre_eval if eval_preprocess else pre_train
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        embed_dim = int(getattr(model.visual, "output_dim", 0)) or _infer_dim(
            model, preprocess, device)
        logger.info("Loaded open_clip %s/%s (dim=%d)", name, pretrained, embed_dim)
        return ClipBundle(model, preprocess, embed_dim, name, pretrained, "open_clip")
    except Exception as exc:  # pragma: no cover
        logger.warning("open_clip unavailable (%s); trying transformers CLIP", exc)

    from transformers import CLIPModel, CLIPProcessor
    hf_name = cfg.get("features.clip_hf_name", "openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained(hf_name).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    processor = CLIPProcessor.from_pretrained(hf_name)
    embed_dim = int(model.config.projection_dim)

    def preprocess(img):
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    logger.info("Loaded transformers CLIP %s (dim=%d)", hf_name, embed_dim)
    return ClipBundle(model, preprocess, embed_dim, hf_name, "hf", "transformers")


def _infer_dim(model, preprocess, device) -> int:  # pragma: no cover
    from PIL import Image
    dummy = Image.new("RGB", (224, 224))
    with torch.no_grad():
        emb = encode_image_tensor(model, preprocess(dummy).unsqueeze(0).to(device),
                                  backend="open_clip")
    return emb.shape[-1]


def encode_image_tensor(model, pixel_values, backend: str = "open_clip"):
    if backend == "open_clip":
        return model.encode_image(pixel_values)
    return model.get_image_features(pixel_values=pixel_values)


def encode_pil_images(bundle: ClipBundle, images: List, device,
                      batch_size: int = 32, normalize: bool = True):
    """Encode a list of PIL images to a [N, D] tensor (used by gen metrics)."""
    feats = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            pixels = torch.stack([bundle.preprocess(im) for im in batch]).to(device)
            with autocast(device, enabled=device.type == "cuda"):
                emb = encode_image_tensor(bundle.model, pixels, bundle.backend)
            emb = emb.float()
            if normalize:
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            feats.append(emb.cpu())
    return torch.cat(feats, dim=0)


def encode_image_paths(bundle: ClipBundle, paths: List[str], device,
                       batch_size: int = 32, normalize: bool = True):
    images = [load_image(p) for p in paths]
    return encode_pil_images(bundle, images, device, batch_size, normalize)
