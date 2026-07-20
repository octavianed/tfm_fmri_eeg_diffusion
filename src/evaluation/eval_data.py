"""Assemble aligned (fMRI, CLIP, low-level) matrices for a (subject, split).

Rows follow ``feat_idx`` order so fMRI rows line up with the precomputed
feature arrays. fMRI is returned already normalized with the train-fitted
scaler. Used by the baselines and the ablation evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..features.load_features import load_split_features


@dataclass
class SubjectMatrices:
    subject: str
    split: str
    fmri: Optional[np.ndarray]
    clip: Optional[np.ndarray]
    low: Optional[np.ndarray]
    image_paths: List[str]
    image_ids: List[str]

    def __len__(self):
        for arr in (self.clip, self.fmri, self.low):
            if arr is not None:
                return arr.shape[0]
        return len(self.image_paths)


def load_subject_matrices(cfg, datamodule, subject: str, split: str,
                          want: Sequence[str] = ("fmri", "clip", "low"),
                          fmri_dtype=np.float32) -> SubjectMatrices:
    frame = datamodule.subject_split_frame(subject, split)
    image_paths = frame["image_path"].astype(str).tolist()
    image_ids = frame["image_id"].astype(str).tolist()

    fmri = clip = low = None
    if "fmri" in want and len(frame):
        reader = datamodule.subject_reader(subject)
        norm = datamodule.normalizer(subject)
        sources = frame["source"].astype(str).to_numpy() \
            if "source" in frame.columns else ["train"] * len(frame)
        rows = []
        for li, src in zip(frame["local_index"].to_numpy(), sources):
            vec = reader.get_fmri(int(li), str(src))
            if norm is not None:
                vec = norm.transform(vec)
            rows.append(np.asarray(vec, dtype=fmri_dtype))
        fmri = np.stack(rows, axis=0) if rows else None
    if "clip" in want:
        clip = load_split_features(cfg, subject, split, "clip")
    if "low" in want:
        low = load_split_features(cfg, subject, split, "low")

    return SubjectMatrices(subject, split, fmri, clip, low, image_paths, image_ids)
