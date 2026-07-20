"""EEG (THINGS-EEG2) data orchestration — the modality-parallel of
:class:`~src.data.datamodule.FmriDataModule`.

It exposes the SAME public surface the training/eval/generation stack relies on
(``prepare``, ``subjects``, ``voxel_counts``, ``get_frame``,
``subject_split_frame``, ``subject_reader``, ``normalizer``, ``input_dim``,
``load_feature_arrays``, ``build_dataset``, ``build_dataloader``) so nothing
downstream needs to know the modality.

Two-frame design (this is what lets the shared feature/eval code work unchanged):

* ``subject_split_frame(subject, split)`` → **one row per unique image**
  (``feat_idx`` order). Used by ``precompute_clip``/``precompute_vae_latents``
  and by ``load_subject_matrices`` (via ``reader.get_fmri`` = per-image mean over
  repetitions). Feature arrays are therefore per-image and align by identity.
* ``get_frame(split)`` → **trial-aware rows** per ``dataset.trial_aggregation``:
  ``none`` = one row per (image, repetition) [training augmentation]; ``mean`` =
  one row per image (``rep = -1``) [clean, high-SNR val/test retrieval — the
  candidate set is unique images]. Both carry ``feat_idx`` = the image's rank in
  its (subject, split) group, so ``arr[feat_idx]`` is the correct target.

Normalization is per-channel, fitted on train trials only, and cached to disk
namespaced by channel count (spec §7.4).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None

from ..utils import (clip_feature_path, get_logger, metadata_path, seed_worker,
                     vae_pca_feature_path)
from ..utils.paths import processed_dir
from .datamodule import SubjectHomogeneousBatchSampler
from .eeg_normalization import EegNormalizer
from .eeg_things_dataset import (EegDataset, EegSubjectData, eeg_selection_tag,
                                 resolve_eeg_subjects)

logger = get_logger("eeg_datamodule")

_DEFAULT_AGG = {"train": "none", "val": "mean", "test": "mean"}


def _subject_num(subject: str) -> int:
    m = re.search(r"(\d+)", subject)
    return int(m.group(1)) if m else 0


class EegDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = cfg.get("dataset.root_dir")
        self.selection = cfg.get("dataset.subject_selection", "sub-01")
        self.channels = int(cfg.get("dataset.channels", 17))
        tw = cfg.get("dataset.time_window_ms", None)
        self.time_window_ms = list(tw) if tw else None
        self.val_ratio = float(cfg.get("dataset.val_ratio", 0.1))
        self.split_seed = int(cfg.get("dataset.split_seed",
                                      cfg.get("project.seed", 42)))
        self.norm_eps = float(cfg.get("dataset.norm_eps", 1e-6))
        agg = dict(_DEFAULT_AGG)
        agg.update(cfg.get("dataset.trial_aggregation", {}) or {})
        self.trial_aggregation = {k: str(v) for k, v in agg.items()}
        self.batch_size = int(cfg.get("training.batch_size", 256))
        self.num_workers = int(cfg.get("training.num_workers", 0))

        self.subjects: List[str] = resolve_eeg_subjects(
            self.root_dir, self.selection, self.channels)
        self.tag = eeg_selection_tag(self.selection)
        self._subject_data: Dict[str, EegSubjectData] = {
            s: EegSubjectData(s, self.root_dir, channels=self.channels,
                              time_window_ms=self.time_window_ms)
            for s in self.subjects}

        self.image_meta: Optional[pd.DataFrame] = None
        self.normalizers: Dict[str, EegNormalizer] = {}
        self.voxel_counts: Dict[str, Tuple[int, int]] = {}
        self.signal_shape: Dict[str, Tuple[int, int]] = {}
        self._frame_cache: Dict[str, pd.DataFrame] = {}

    # -- aggregation --------------------------------------------------------
    def _agg_for(self, split: str) -> str:
        return self.trial_aggregation.get(split, "none" if split == "train" else "mean")

    # -- preparation --------------------------------------------------------
    def prepare(self, force_rebuild: bool = False) -> "EegDataModule":
        self.image_meta = self._load_or_build_metadata(force_rebuild)
        self._load_or_fit_normalization(force_rebuild)
        for s in self.subjects:
            sd = self._subject_data[s]
            self.signal_shape[s] = (sd.channels, sd.n_times)
            self.voxel_counts[s] = (sd.channels, sd.n_times)
        logger.info("Prepared %d EEG subject(s) %s | channels=%d | signal=%s | "
                    "agg=%s | images/split: %s", len(self.subjects), self.subjects,
                    self.channels, self.signal_shape, self.trial_aggregation,
                    dict(self.image_meta["split"].value_counts()))
        return self

    def _split_signature(self) -> dict:
        return {"subjects": self.subjects, "val_ratio": self.val_ratio,
                "split_seed": self.split_seed, "channels": self.channels}

    def _load_or_build_metadata(self, force: bool) -> pd.DataFrame:
        meta_path = metadata_path(self.cfg, f"eeg_{self.tag}_{self.channels}ch")
        sig_path = meta_path.with_suffix(".split.json")
        if not force and meta_path.exists() and sig_path.exists():
            saved = json.loads(sig_path.read_text(encoding="utf-8"))
            if saved == self._split_signature():
                logger.info("Loading cached EEG split metadata: %s", meta_path)
                return pd.read_csv(meta_path)
            logger.warning("EEG split config changed; rebuilding metadata.")
        df = self._build_metadata()
        df.to_csv(meta_path, index=False)
        sig_path.write_text(json.dumps(self._split_signature(), indent=2),
                            encoding="utf-8")
        logger.info("Built EEG split metadata (%d images): %s", len(df), meta_path)
        return df

    def _build_metadata(self) -> pd.DataFrame:
        rows = []
        for subj in self.subjects:
            sd = self._subject_data[subj]
            n_tr = sd.num_images("train")
            idx = np.arange(n_tr)
            rng = np.random.default_rng(self.split_seed + _subject_num(subj))
            rng.shuffle(idx)
            n_val = int(round(self.val_ratio * n_tr))
            assign = {int(i): ("val" if r < n_val else "train")
                      for r, i in enumerate(idx)}
            rows.extend(self._image_rows(subj, sd, "train", assign))
            if sd.has_test:
                n_te = sd.num_images("test")
                rows.extend(self._image_rows(subj, sd, "test",
                                             {i: "test" for i in range(n_te)}))
        df = pd.DataFrame(rows)
        df.insert(0, "global_index", np.arange(len(df)))
        return df

    def _image_rows(self, subj, sd, source, assign) -> list:
        """One row per unique image; ``feat_idx`` = rank within (subject, split)."""
        per_split = defaultdict(list)
        for i, split in assign.items():
            per_split[split].append(i)
        feat_idx = {}
        for split, imgs in per_split.items():
            for rank, ii in enumerate(sorted(imgs)):
                feat_idx[ii] = rank
        out = []
        for i, split in assign.items():
            out.append({"subject_id": subj, "source": source, "split": split,
                        "img_index": int(i), "feat_idx": feat_idx[i],
                        "image_id": sd.image_id(source, i),
                        "image_path": sd.image_path(source, i)})
        return out

    # -- normalization ------------------------------------------------------
    def _norm_path(self, subject: str) -> Path:
        d = processed_dir(self.cfg) / "normalization"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{subject}_eeg{self.channels}ch_norm.npz"

    def _load_or_fit_normalization(self, force: bool) -> None:
        for subj in self.subjects:
            path = self._norm_path(subj)
            if not force and path.exists():
                self.normalizers[subj] = EegNormalizer.load(path)
                continue
            norm = self._fit_normalizer(subj)
            norm.save(path)
            self.normalizers[subj] = norm
            logger.info("Fitted EEG normalization for %s (C=%d) -> %s",
                        subj, norm.mean.shape[0], path)

    def _fit_normalizer(self, subj: str) -> EegNormalizer:
        sd = self._subject_data[subj]
        train_imgs = self.image_meta[(self.image_meta.subject_id == subj) &
                                     (self.image_meta.split == "train") &
                                     (self.image_meta.source == "train")]["img_index"].to_numpy()
        arr = sd._load("train")                      # [n_img, n_rep, C, T]
        sub = arr[np.sort(train_imgs)]               # [n_tr, n_rep, C, T]
        flat = sub.reshape(-1, sub.shape[2], sub.shape[3])   # [N, C, T]
        return EegNormalizer(eps=self.norm_eps).fit(flat)

    # -- frames -------------------------------------------------------------
    def get_frame(self, split: str) -> pd.DataFrame:
        """Trial-aware rows for a split (all subjects), per ``trial_aggregation``."""
        assert self.image_meta is not None, "call prepare() first"
        if split in self._frame_cache:
            return self._frame_cache[split]
        agg = self._agg_for(split)
        imgs = self.image_meta[self.image_meta.split == split]
        rows = []
        for r in imgs.itertuples(index=False):
            base = {"subject_id": r.subject_id, "source": r.source, "split": split,
                    "img_index": int(r.img_index), "feat_idx": int(r.feat_idx),
                    "image_id": r.image_id, "image_path": r.image_path}
            if agg == "mean":
                rows.append({**base, "rep": -1})
            else:  # per-trial
                n_rep = self._subject_data[r.subject_id].n_reps(r.source)
                for rep in range(n_rep):
                    rows.append({**base, "rep": rep})
        frame = pd.DataFrame(rows).reset_index(drop=True)
        self._frame_cache[split] = frame
        return frame

    def subject_split_frame(self, subject: str, split: str) -> pd.DataFrame:
        """One row per unique image for (subject, split), ordered by ``feat_idx``.

        Carries ``local_index`` (= ``img_index``) and ``source`` so the shared
        ``load_subject_matrices`` reads the per-image mean via ``reader.get_fmri``.
        """
        assert self.image_meta is not None, "call prepare() first"
        f = self.image_meta[(self.image_meta.subject_id == subject) &
                            (self.image_meta.split == split)].copy()
        f = f.sort_values("feat_idx").reset_index(drop=True)
        f["local_index"] = f["img_index"]
        return f

    def subject_reader(self, subject: str) -> EegSubjectData:
        return self._subject_data[subject]

    def normalizer(self, subject: str) -> Optional[EegNormalizer]:
        return self.normalizers.get(subject)

    def input_dim(self, subject: Optional[str] = None):
        if subject is not None:
            return self.signal_shape[subject]
        vals = set(self.signal_shape.values())
        return next(iter(vals)) if len(vals) == 1 else dict(self.signal_shape)

    # -- features -----------------------------------------------------------
    def load_feature_arrays(self, split: str,
                            kinds=("clip", "low")) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for subj in self.subjects:
            entry: Dict[str, np.ndarray] = {}
            if "clip" in kinds:
                p = clip_feature_path(self.cfg, subj, split)
                if p.exists():
                    entry["clip"] = np.load(p)
            if "low" in kinds:
                p = vae_pca_feature_path(self.cfg, subj, split)
                if p.exists():
                    entry["low"] = np.load(p)
            if entry:
                out[subj] = entry
        return out

    # -- datasets / loaders -------------------------------------------------
    def build_dataset(self, split: str, return_image: bool = False,
                      load_features: bool = True, kinds=("clip", "low"),
                      image_transform=None, return_fmri: bool = True) -> EegDataset:
        frame = self.get_frame(split)
        features = self.load_feature_arrays(split, kinds) if load_features else None
        return EegDataset(
            frame=frame, subjects=self._subject_data,
            normalizers=self.normalizers, features=features,
            image_transform=image_transform, return_fmri=return_fmri,
            return_image=return_image)

    def build_dataloader(self, split: str, batch_size: Optional[int] = None,
                        shuffle: Optional[bool] = None,
                        num_workers: Optional[int] = None,
                        drop_last: Optional[bool] = None,
                        return_image: bool = False, load_features: bool = True,
                        kinds=("clip", "low"), image_transform=None,
                        return_fmri: bool = True):
        dataset = self.build_dataset(split, return_image=return_image,
                                     load_features=load_features, kinds=kinds,
                                     image_transform=image_transform,
                                     return_fmri=return_fmri)
        if shuffle is None:
            shuffle = split == "train"
        if drop_last is None:
            drop_last = split == "train"
        batch_size = batch_size or self.batch_size
        num_workers = self.num_workers if num_workers is None else num_workers
        sampler = SubjectHomogeneousBatchSampler(
            dataset._subject, batch_size=batch_size, shuffle=shuffle,
            drop_last=drop_last, seed=self.split_seed)
        pin = bool(torch is not None and torch.cuda.is_available())
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=pin, worker_init_fn=seed_worker)
