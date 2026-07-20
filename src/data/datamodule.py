"""Data orchestration: subject resolution, reproducible splits, train-only
normalization, feature loading and DataLoaders.

The split is created once from the *labeled* training data and cached to
``data/processed/metadata_<tag>.csv`` (with a sidecar recording the split
parameters, so a config change triggers a rebuild rather than silently reusing
a stale split). Normalization is fitted per subject on train rows only and
saved to disk (spec §3.5, §13.1).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except Exception:  # pragma: no cover
    torch = None
    DataLoader = None

from ..utils import (clip_feature_path, get_logger, metadata_path,
                     normalization_path, seed_worker, vae_pca_feature_path)
from .algonauts_dataset import AlgonautsDataset, SubjectData
from .fmri_normalization import FmriNormalizer
from .subject_selection import resolve_subjects, selection_tag

logger = get_logger("datamodule")


class SubjectHomogeneousBatchSampler:
    """Yield batches whose samples all come from a single subject.

    Different subjects have different voxel counts, so batching them together
    would break tensor stacking (and within-batch contrastive losses). Grouping
    by subject keeps every batch shape-consistent; for a single subject this is
    just ordinary (optionally shuffled) batching.
    """

    def __init__(self, subject_ids: Sequence[str], batch_size: int,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0):
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for idx, s in enumerate(subject_ids):
            self.groups[str(s)].append(idx)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def _num_batches(self, n: int) -> int:
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return sum(self._num_batches(len(v)) for v in self.groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch) if self.shuffle else None
        batches: List[List[int]] = []
        for _, idxs in self.groups.items():
            order = list(idxs)
            if self.shuffle:
                rng.shuffle(order)
            for i in range(0, len(order), self.batch_size):
                batch = order[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def _subject_num(subject: str) -> int:
    m = re.search(r"(\d+)", subject)
    return int(m.group(1)) if m else 0


class FmriDataModule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root_dir = cfg.get("dataset.root_dir")
        self.selection = cfg.get("dataset.subject_selection", "subj01")
        self.hemispheres = list(cfg.get("dataset.use_hemispheres", ["lh", "rh"]))
        self.val_ratio = float(cfg.get("dataset.val_ratio", 0.1))
        self.test_ratio = float(cfg.get("dataset.test_ratio", 0.1))
        self.split_seed = int(cfg.get("dataset.split_seed",
                                      cfg.get("project.seed", 42)))
        self.norm_eps = float(cfg.get("dataset.norm_eps", 1e-6))
        self.batch_size = int(cfg.get("training.batch_size", 64))
        self.num_workers = int(cfg.get("training.num_workers", 4))

        self.train_root, self.test_root = self._resolve_roots()
        # 'official' uses the released test_data as the test split; it requires a
        # test_root. 'internal' carves test from the labeled training data.
        requested_mode = str(cfg.get("dataset.test_split", "internal"))
        if requested_mode == "official" and self.test_root is None:
            logger.warning("dataset.test_split='official' but no test_data found "
                           "under %s; falling back to 'internal'.", self.root_dir)
            requested_mode = "internal"
        self.test_split_mode = requested_mode

        self.subjects: List[str] = resolve_subjects(self.train_root, self.selection)
        self.tag = selection_tag(self.selection)
        self._subject_data: Dict[str, SubjectData] = {
            s: SubjectData(s, self.train_root,
                           self.test_root if self.test_split_mode == "official" else None,
                           self.hemispheres)
            for s in self.subjects}
        if self.test_split_mode == "official":
            missing = [s for s in self.subjects if not self._subject_data[s].has_test]
            if missing:
                raise FileNotFoundError(
                    f"test_split='official' but test fMRI missing for {missing} "
                    f"under {self.test_root}. Use dataset.test_split=internal instead.")
        self.metadata: Optional[pd.DataFrame] = None
        self.normalizers: Dict[str, FmriNormalizer] = {}
        self.voxel_counts: Dict[str, int] = {}

    def _resolve_roots(self):
        """Detect the split_dirs layout (train_data/ + test_data/) or flat."""
        root = Path(self.root_dir)
        train_name = str(self.cfg.get("dataset.train_dirname", "train_data"))
        test_name = str(self.cfg.get("dataset.test_dirname", "test_data"))
        train_root = root / train_name if (root / train_name).exists() else root
        test_root = root / test_name if (root / test_name).exists() else None
        return train_root, test_root

    # -- preparation --------------------------------------------------------
    def prepare(self, force_rebuild: bool = False) -> "FmriDataModule":
        self.metadata = self._load_or_build_metadata(force_rebuild)
        self._load_or_fit_normalization(force_rebuild)
        self.voxel_counts = {s: self._subject_data[s].num_voxels("train")
                             for s in self.subjects}
        logger.info("Prepared %d subject(s) %s | test_split=%s | samples: %s | voxels: %s",
                    len(self.subjects), self.subjects, self.test_split_mode,
                    dict(self.metadata["split"].value_counts()), self.voxel_counts)
        return self

    def _split_signature(self) -> dict:
        return {"subjects": self.subjects, "val_ratio": self.val_ratio,
                "test_ratio": self.test_ratio, "test_split": self.test_split_mode,
                "split_seed": self.split_seed, "hemispheres": self.hemispheres,
                "train_root": str(self.train_root),
                "test_root": str(self.test_root) if self.test_root else None}

    def _load_or_build_metadata(self, force: bool) -> pd.DataFrame:
        meta_path = metadata_path(self.cfg, self.tag)
        sig_path = meta_path.with_suffix(".split.json")
        if not force and meta_path.exists() and sig_path.exists():
            saved_sig = json.loads(sig_path.read_text(encoding="utf-8"))
            if saved_sig == self._split_signature():
                logger.info("Loading cached split metadata: %s", meta_path)
                return pd.read_csv(meta_path)
            logger.warning("Split config changed; rebuilding metadata.")
        df = self._build_metadata()
        df.to_csv(meta_path, index=False)
        sig_path.write_text(json.dumps(self._split_signature(), indent=2),
                            encoding="utf-8")
        logger.info("Built split metadata (%d rows): %s", len(df), meta_path)
        return df

    def _build_metadata(self) -> pd.DataFrame:
        rows = []
        for subj in self.subjects:
            sd = self._subject_data[subj]
            n = sd.num_samples("train")
            idx = np.arange(n)
            rng = np.random.default_rng(self.split_seed + _subject_num(subj))
            rng.shuffle(idx)

            if self.test_split_mode == "official":
                # train/val come from the training data; test is the official set.
                n_val = int(round(self.val_ratio * n))
                assign = {int(i): ("val" if r < n_val else "train")
                          for r, i in enumerate(idx)}
            else:
                n_val = int(round(self.val_ratio * n))
                n_test = int(round(self.test_ratio * n))
                assign = {}
                for r, i in enumerate(idx):
                    assign[int(i)] = ("val" if r < n_val else
                                      "test" if r < n_val + n_test else "train")
            rows.extend(self._rows_for_source(subj, sd, "train", assign))

            if self.test_split_mode == "official":
                n_test = sd.num_samples("test")
                test_assign = {i: "test" for i in range(n_test)}
                rows.extend(self._rows_for_source(subj, sd, "test", test_assign))

        df = pd.DataFrame(rows)
        df.insert(0, "global_index", np.arange(len(df)))
        return df

    def _rows_for_source(self, subj, sd, source, assign) -> list:
        """Build rows for local indices in ``assign`` (index -> split), assigning
        a per-split ``feat_idx`` by sorted local index."""
        per_split = defaultdict(list)
        for i, split in assign.items():
            per_split[split].append(i)
        feat_idx = {}
        for split, locs in per_split.items():
            for rank, li in enumerate(sorted(locs)):
                feat_idx[li] = rank
        out = []
        for i, split in assign.items():
            out.append({"subject_id": subj, "source": source, "local_index": i,
                        "image_id": sd.image_id(i, source),
                        "image_path": sd.get_image_path(i, source),
                        "split": split, "feat_idx": feat_idx[i]})
        return out

    def _load_or_fit_normalization(self, force: bool) -> None:
        for subj in self.subjects:
            path = normalization_path(self.cfg, subj)
            if not force and path.exists():
                self.normalizers[subj] = FmriNormalizer.load(path)
                continue
            train_locs = self.metadata[(self.metadata.subject_id == subj) &
                                       (self.metadata.split == "train") &
                                       (self.metadata.source == "train")]["local_index"]
            norm = self._fit_normalizer(subj, train_locs.to_numpy())
            norm.save(path)
            self.normalizers[subj] = norm
            logger.info("Fitted fMRI normalization for %s (V=%d) -> %s",
                        subj, norm.mean.shape[0], path)

    def _fit_normalizer(self, subj: str, train_locs: np.ndarray,
                        chunk: int = 2048) -> FmriNormalizer:
        sd = self._subject_data[subj]
        rows = np.sort(np.asarray(train_locs, dtype=np.int64))
        means, stds = [], []
        for hemi in self.hemispheres:
            arr = sd.hemi_array(hemi, "train")
            V = arr.shape[1]
            s = np.zeros(V, np.float64)
            ss = np.zeros(V, np.float64)
            for start in range(0, len(rows), chunk):
                block = rows[start:start + chunk]
                x = np.asarray(arr[block], dtype=np.float64)
                s += x.sum(0)
                ss += (x * x).sum(0)
            n = max(1, len(rows))
            mean = s / n
            var = np.clip(ss / n - mean * mean, 0.0, None)
            means.append(mean.astype(np.float32))
            stds.append(np.sqrt(var).astype(np.float32))
        mean = np.concatenate(means)
        std = np.concatenate(stds)
        return FmriNormalizer(mean=mean, std=std, eps=self.norm_eps)

    # -- frames -------------------------------------------------------------
    def get_frame(self, split: str) -> pd.DataFrame:
        assert self.metadata is not None, "call prepare() first"
        return self.metadata[self.metadata.split == split].reset_index(drop=True)

    def subject_split_frame(self, subject: str, split: str) -> pd.DataFrame:
        """Rows for one (subject, split), ordered by ``feat_idx`` — the order
        precomputed feature arrays must follow."""
        assert self.metadata is not None, "call prepare() first"
        f = self.metadata[(self.metadata.subject_id == subject) &
                          (self.metadata.split == split)]
        return f.sort_values("feat_idx").reset_index(drop=True)

    def subject_reader(self, subject: str) -> SubjectData:
        return self._subject_data[subject]

    def normalizer(self, subject: str) -> Optional[FmriNormalizer]:
        return self.normalizers.get(subject)

    def input_dim(self, subject: Optional[str] = None):
        if subject is not None:
            return self.voxel_counts[subject]
        vals = set(self.voxel_counts.values())
        return next(iter(vals)) if len(vals) == 1 else dict(self.voxel_counts)

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
                      image_transform=None, return_fmri: bool = True) -> AlgonautsDataset:
        frame = self.get_frame(split)
        features = self.load_feature_arrays(split, kinds) if load_features else None
        return AlgonautsDataset(
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
        # The sampler is reachable as ``loader.batch_sampler`` — call
        # ``loader.batch_sampler.set_epoch(e)`` each epoch for reproducible shuffles.
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=pin, worker_init_fn=seed_worker)
