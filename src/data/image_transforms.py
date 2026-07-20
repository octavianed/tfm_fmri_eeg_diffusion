"""Image loading and transforms for CLIP targets and the Stable Diffusion VAE.

CLIP embeddings are best computed with the exact preprocessing shipped by the
CLIP implementation (``open_clip``), so :func:`build_clip_preprocess` prefers
that and only falls back to a manual pipeline. The VAE transform maps an image
to the ``[-1, 1]`` range Stable Diffusion's VAE expects.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

try:
    import torch
    from torchvision import transforms as T
except Exception:  # pragma: no cover
    torch = None
    T = None

# OpenAI CLIP normalization constants.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_clip_preprocess(image_size: int = 224):
    """Manual CLIP preprocessing (fallback when open_clip's is unavailable)."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def build_vae_preprocess(image_size: int = 512):
    """Preprocess for the SD VAE encoder: RGB -> tensor in [-1, 1]."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def build_display_transform(image_size: int = 256):
    """Plain resize+crop+tensor in [0, 1] for visualization/grids."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])


def denormalize_vae(tensor):
    """Map a VAE-space tensor in [-1, 1] back to [0, 1] for display."""
    return (tensor * 0.5 + 0.5).clamp(0.0, 1.0)


def tensor_to_pil(tensor) -> Image.Image:
    """Convert a CHW tensor in [0, 1] to a PIL image."""
    arr = (tensor.detach().cpu().clamp(0, 1) * 255).round().to(torch.uint8)
    return Image.fromarray(arr.permute(1, 2, 0).numpy())
