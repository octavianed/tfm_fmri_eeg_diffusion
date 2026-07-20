"""Precompute frozen-CLIP image embeddings (the semantic targets).

For every (subject, split) we iterate images in ``feat_idx`` order and save a
``[N, clip_dim]`` array aligned to that order, so the dataset can look targets
up by position (spec §7.3, §13.2). Embeddings are stored *unnormalized*;
normalization happens in the loss / retrieval code.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # pragma: no cover
        pass

from ..data.image_transforms import load_image
from ..utils import autocast, clip_feature_path, get_device, get_logger
from .clip_model import encode_image_tensor, load_clip

logger = get_logger("precompute_clip")


class ImagePathDataset(Dataset):
    """Applies a preprocessing transform to images given by path (order kept)."""

    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(load_image(self.paths[i]))


def precompute_clip(cfg, datamodule, splits=("train", "val", "test"),
                    overwrite: bool = False) -> dict:
    device = get_device(cfg.get("runtime.device", "auto"))
    bundle = load_clip(cfg, device)
    batch_size = int(cfg.get("features.precompute_batch_size", 64))
    num_workers = int(cfg.get("training.num_workers", 4))
    results = {}
    for subj in datamodule.subjects:
        for split in splits:
            frame = datamodule.subject_split_frame(subj, split)
            if len(frame) == 0:
                continue
            out_path = clip_feature_path(cfg, subj, split)
            if out_path.exists() and not overwrite:
                logger.info("[skip] exists: %s", out_path)
                results[(subj, split)] = tuple(np.load(out_path, mmap_mode="r").shape)
                continue
            paths = frame["image_path"].tolist()
            ds = ImagePathDataset(paths, bundle.preprocess)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=device.type == "cuda")
            embs = []
            with torch.no_grad():
                for imgs in loader:
                    imgs = imgs.to(device, non_blocking=True)
                    with autocast(device, enabled=device.type == "cuda"):
                        emb = encode_image_tensor(bundle.model, imgs, bundle.backend)
                    embs.append(emb.float().cpu().numpy())
            arr = np.concatenate(embs, axis=0).astype(np.float32)
            np.save(out_path, arr)
            results[(subj, split)] = arr.shape
            logger.info("[clip] %s/%s -> %s %s", subj, split, out_path, arr.shape)
    return results
