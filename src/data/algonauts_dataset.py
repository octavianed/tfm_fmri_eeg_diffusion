"""NSD Algonauts 2023 dataset objects.

Two on-disk layouts are supported (auto-detected by the datamodule):

1. **split_dirs** (the official download): training and test live under
   separate top-level folders, and the test split includes the released fMRI::

       <root>/train_data/subj01/
           training_split/training_fmri/{lh,rh}_training_fmri.npy   # [N_train, V]
           training_split/training_images/train-XXXX_nsd-YYYYY.png
           roi_masks/...
       <root>/test_data/subj01/
           test_split/test_fmri/{lh,rh}_test_fmri.npy               # [N_test, V]
           test_split/test_images/test-XXXX_nsd-YYYYY.png
           test_split/noise_ceiling/{lh,rh}_noise_ceiling.npy

2. **flat**: ``<root>/subj01/training_split/...`` with everything under one
   subject folder (older/repackaged layouts). Here the official test may lack
   fMRI, so an internal test split is used instead.

``SubjectData`` memory-maps the fMRI arrays (``mmap_mode='r'``) so several
subjects never sit fully in RAM (spec §4). fMRI is read per ``source``
('train' vs 'test') because those live in different arrays with the same
number of vertices ``V`` per subject.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None

    class Dataset:  # minimal stand-in so the module imports without torch
        pass

from .fmri_normalization import FmriNormalizer
from .image_transforms import load_image
from .subject_selection import subject_dir

_IMG_EXT = (".png", ".jpg", ".jpeg")


def parse_image_id(path) -> str:
    """Use the filename stem as a stable image id (e.g. train-0001_nsd-00013)."""
    return Path(path).stem


@dataclass
class SubjectData:
    """Reader for one subject's images and (memory-mapped) fMRI.

    ``train_root``/``test_root`` are the folders that *contain* the ``subjNN``
    directories for the training and (optional) official test data.
    """

    subject_id: str
    train_root: str
    test_root: Optional[str] = None
    hemispheres: Sequence[str] = ("lh", "rh")
    _fmri: Dict = field(default_factory=dict, init=False, repr=False)
    _images: Dict = field(default_factory=dict, init=False, repr=False)
    _counts: Dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        train_dir = subject_dir(self.train_root, self.subject_id) / "training_split"
        self._paths = {
            "train": {"fmri_dir": train_dir / "training_fmri",
                      "fmri_suffix": "training_fmri",
                      "image_dir": train_dir / "training_images"},
        }
        if self.test_root is not None:
            test_dir = subject_dir(self.test_root, self.subject_id) / "test_split"
            self._paths["test"] = {"fmri_dir": test_dir / "test_fmri",
                                   "fmri_suffix": "test_fmri",
                                   "image_dir": test_dir / "test_images"}

    # -- availability -------------------------------------------------------
    @property
    def has_test(self) -> bool:
        if "test" not in self._paths:
            return False
        d = self._paths["test"]["fmri_dir"]
        return all((d / f"{h}_test_fmri.npy").exists() for h in self.hemispheres)

    def _check_source(self, source: str):
        if source not in self._paths:
            raise ValueError(f"Unknown source '{source}' for {self.subject_id}")

    # -- images -------------------------------------------------------------
    def image_paths(self, source: str = "train") -> List[str]:
        self._check_source(source)
        if source not in self._images:
            image_dir = self._paths[source]["image_dir"]
            if not image_dir.exists():
                raise FileNotFoundError(f"Missing images dir: {image_dir}")
            paths = sorted(str(p) for p in image_dir.iterdir()
                           if p.suffix.lower() in _IMG_EXT)
            if not paths:
                raise FileNotFoundError(f"No images found in {image_dir}")
            self._images[source] = paths
        return self._images[source]

    # -- fMRI ---------------------------------------------------------------
    def hemi_array(self, hemi: str, source: str = "train") -> np.ndarray:
        self._check_source(source)
        key = (source, hemi)
        if key not in self._fmri:
            suffix = self._paths[source]["fmri_suffix"]
            path = self._paths[source]["fmri_dir"] / f"{hemi}_{suffix}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing fMRI file: {path}")
            self._fmri[key] = np.load(path, mmap_mode="r")
        return self._fmri[key]

    def num_samples(self, source: str = "train") -> int:
        if source not in self._counts:
            n_img = len(self.image_paths(source))
            n_fmri = self.hemi_array(self.hemispheres[0], source).shape[0]
            if n_img != n_fmri:
                raise ValueError(
                    f"{self.subject_id}/{source}: image/fMRI count mismatch "
                    f"({n_img} images vs {n_fmri} fMRI rows)")
            self._counts[source] = int(n_fmri)
        return self._counts[source]

    def num_voxels(self, source: str = "train") -> int:
        return int(sum(self.hemi_array(h, source).shape[1]
                       for h in self.hemispheres))

    def get_fmri(self, local_index: int, source: str = "train") -> np.ndarray:
        parts = [np.asarray(self.hemi_array(h, source)[local_index], dtype=np.float32)
                 for h in self.hemispheres]
        return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]

    def get_image_path(self, local_index: int, source: str = "train") -> str:
        return self.image_paths(source)[local_index]

    def image_id(self, local_index: int, source: str = "train") -> str:
        return parse_image_id(self.image_paths(source)[local_index])


class AlgonautsDataset(Dataset):
    """Split-scoped dataset backed by a metadata frame + per-subject readers.

    Each row carries its ``source`` ('train'/'test') and ``local_index`` into
    that source's arrays, plus ``feat_idx`` (position within the (subject,
    split) group) so precomputed feature arrays stay aligned with the data.
    """

    def __init__(self, frame, subjects: Dict[str, SubjectData],
                 normalizers: Optional[Dict[str, FmriNormalizer]] = None,
                 features: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
                 image_transform=None, return_fmri: bool = True,
                 return_image: bool = False, return_paths: bool = True):
        self.frame = frame.reset_index(drop=True)
        self.subjects = subjects
        self.normalizers = normalizers or {}
        self.features = features or {}
        self.image_transform = image_transform
        self.return_fmri = return_fmri
        self.return_image = return_image
        self.return_paths = return_paths
        self._subject = self.frame["subject_id"].to_numpy()
        self._source = self.frame["source"].astype(str).to_numpy()
        self._local = self.frame["local_index"].to_numpy()
        self._feat = self.frame["feat_idx"].to_numpy()
        self._image_id = self.frame["image_id"].astype(str).to_numpy()
        self._image_path = self.frame["image_path"].astype(str).to_numpy()

    def __len__(self) -> int:
        return len(self.frame)

    def subject_of(self, i: int) -> str:
        return str(self._subject[i])

    def __getitem__(self, i: int) -> dict:
        subj = str(self._subject[i])
        source = str(self._source[i])
        local_index = int(self._local[i])
        feat_idx = int(self._feat[i])
        sample = {"index": i, "subject_id": subj,
                  "image_id": str(self._image_id[i])}

        if self.return_fmri:
            fmri = self.subjects[subj].get_fmri(local_index, source)
            norm = self.normalizers.get(subj)
            if norm is not None:
                fmri = norm.transform(fmri)
            sample["fmri"] = torch.from_numpy(np.ascontiguousarray(fmri)).float()

        for kind in ("clip", "low"):
            arr = self.features.get(subj, {}).get(kind)
            if arr is not None:
                vec = np.asarray(arr[feat_idx], dtype=np.float32)
                sample[f"{kind}_target"] = torch.from_numpy(vec).float()

        if self.return_image:
            img = load_image(self._image_path[i])
            if self.image_transform is not None:
                img = self.image_transform(img)
            sample["image"] = img
        if self.return_paths:
            sample["image_path"] = str(self._image_path[i])
        return sample
