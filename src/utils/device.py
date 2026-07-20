"""Device selection and mixed-precision helpers."""
from __future__ import annotations

import contextlib

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def get_device(prefer: str = "auto"):
    """Return a torch.device honoring the preference ('auto'|'cuda'|'cpu'|'mps')."""
    if torch is None:  # pragma: no cover
        return "cpu"
    if prefer not in ("auto", "cuda", "cpu", "mps"):
        prefer = "auto"
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if prefer in ("auto", "mps") and mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled: bool, dtype=None):
    """Context manager for mixed precision; no-op when disabled/unsupported."""
    if torch is None or not enabled:
        return contextlib.nullcontext()
    device_type = device.type if hasattr(device, "type") else str(device)
    if device_type not in ("cuda", "cpu"):
        return contextlib.nullcontext()
    if dtype is None:
        dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)


def amp_dtype(device, name: str = "float16"):
    if torch is None:  # pragma: no cover
        return None
    return {"float16": torch.float16, "fp16": torch.float16,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float32": torch.float32}.get(str(name).lower(), torch.float16)


def make_grad_scaler(enabled: bool):
    """Create a GradScaler across torch versions (new ``torch.amp`` vs old)."""
    if torch is None:  # pragma: no cover
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def count_parameters(model, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def cuda_mem_summary() -> str:
    if torch is None or not torch.cuda.is_available():  # pragma: no cover
        return "cuda: unavailable"
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    reserved = torch.cuda.memory_reserved() / 1024 ** 3
    return f"cuda mem: {alloc:.2f} GiB allocated / {reserved:.2f} GiB reserved"
