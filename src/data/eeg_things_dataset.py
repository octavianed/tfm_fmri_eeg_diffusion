"""THINGS-EEG2 readers and torch Dataset.

Layout (see the downloaded dataset)::

    <root>/image_set/image_metadata.npy            # dict of train/test file lists
    <root>/image_set/training_images/<concept>/<file>.jpg
    <root>/image_set/test_images/<concept>/<file>.jpg
    <root>/preprocessed_data/<sub>/preprocessed_eeg_{training,test}.npy      # 17ch
    <root>/preprocessed_data/<sub>__63_channels/preprocessed_eeg_*.npy       # 63ch

Each ``preprocessed_eeg_*.npy`` is a pickled dict with keys
``preprocessed_eeg_data`` (shape ``[n_images, n_repetitions, n_channels,
n_times]``), ``ch_names`` and ``times``. The 63-channel variant carries a
trailing ``stim`` channel that is dropped here (→ 63 real EEG channels).

``EegSubjectData`` mirrors the role of
:class:`~src.data.algonauts_dataset.SubjectData` for fMRI; ``EegDataset`` mirrors
:class:`~src.data.algonauts_dataset.AlgonautsDataset` and emits the SAME batch
keys (``"fmri"`` carries the ``[C, T]`` brain tensor, ``"clip_target"``,
``"low_target"``, ``"subject_id"``, ``"image_id"``, ``"image_path"``,
``"index"``) so the training/eval/generation stack is reused unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object

from .algonauts_dataset import parse_image_id
from .image_transforms import load_image

_TRAIN, _TEST = "train", "test"


def eeg_subject_dir(root_dir, subject: str, channels: int = 17) -> Path:
    """Resolve the preprocessed folder for a subject and channel variant.

    ``channels==63`` uses the ``<sub>__63_channels`` folder; anything else uses
    the default ``<sub>`` (17-channel) folder.
    """
    base = Path(root_dir) / "preprocessed_data"
    name = f"{subject}__63_channels" if int(channels) == 63 else subject
    return base / name


def discover_eeg_subjects(root_dir, channels: int = 17) -> List[str]:
    """List ``sub-NN`` folders present for the requested channel variant."""
    base = Path(root_dir) / "preprocessed_data"
    if not base.exists():
        raise FileNotFoundError(f"THINGS-EEG2 preprocessed_data not found: {base}")
    suffix = "__63_channels" if int(channels) == 63 else ""
    subs = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if int(channels) == 63:
            if name.endswith("__63_channels"):
                subs.append(name[: -len("__63_channels")])
        elif not name.endswith("__63_channels") and name.startswith("sub-"):
            subs.append(name)
    return sorted(set(subs))


def resolve_eeg_subjects(root_dir, selection, channels: int = 17) -> List[str]:
    available = discover_eeg_subjects(root_dir, channels)
    if selection == "all":
        return available
    requested = [selection] if isinstance(selection, str) else list(selection)
    missing = [s for s in requested if s not in available]
    if missing:
        raise ValueError(
            f"EEG subjects {missing} not found for channels={channels} under "
            f"{Path(root_dir) / 'preprocessed_data'} (available: {available}).")
    return sorted(set(requested))


def eeg_selection_tag(selection) -> str:
    if selection == "all":
        return "all"
    if isinstance(selection, str):
        return selection
    subs = list(selection)
    return "-".join(subs) if subs else "none"


class EegSubjectData:
    """Per-subject THINGS-EEG2 reader.

    Loads the preprocessed train/test arrays lazily, drops the ``stim`` channel,
    applies an optional time window and exposes per-(image, repetition) access.
    """

    def __init__(self, subject_id: str, root_dir, channels: int = 17,
                 time_window_ms: Optional[List[float]] = None):
        self.subject_id = subject_id
        self.root_dir = Path(root_dir)
        self.channels_variant = int(channels)
        self.time_window_ms = time_window_ms
        self.subj_dir = eeg_subject_dir(root_dir, subject_id, channels)
        self.image_set = self.root_dir / "image_set"
        self._data: Dict[str, np.ndarray] = {}   # source -> [n_img, n_rep, C, T]
        self._ch_names: Optional[List[str]] = None
        self._times: Optional[np.ndarray] = None
        self._time_idx: Optional[np.ndarray] = None
        self._meta: Optional[dict] = None

    # -- paths --------------------------------------------------------------
    def _npy_path(self, source: str) -> Path:
        fname = "preprocessed_eeg_training.npy" if source == _TRAIN \
            else "preprocessed_eeg_test.npy"
        return self.subj_dir / fname

    @property
    def has_test(self) -> bool:
        return self._npy_path(_TEST).exists()

    # -- metadata (image file lists) ---------------------------------------
    def _metadata(self) -> dict:
        if self._meta is None:
            p = self.image_set / "image_metadata.npy"
            self._meta = np.load(p, allow_pickle=True).item()
        return self._meta

    def _img_files(self, source: str):
        m = self._metadata()
        key = "train_img_files" if source == _TRAIN else "test_img_files"
        return [str(x) for x in m[key]]

    def _img_concepts(self, source: str):
        m = self._metadata()
        key = "train_img_concepts" if source == _TRAIN else "test_img_concepts"
        return [str(x) for x in m[key]]

    # -- signal arrays ------------------------------------------------------
    def _load(self, source: str) -> np.ndarray:
        if source not in self._data:
            path = self._npy_path(source)
            if not path.exists():
                raise FileNotFoundError(f"EEG data not found: {path}")
            d = np.load(path, allow_pickle=True).item()
            arr = np.asarray(d["preprocessed_eeg_data"])  # [n_img, n_rep, C, T]
            ch_names = [str(c) for c in np.asarray(d["ch_names"]).tolist()]
            times = np.asarray(d["times"], dtype=np.float64)
            # Drop trailing 'stim' channel (present in the 63-channel variant).
            keep = [i for i, c in enumerate(ch_names) if c.lower() != "stim"]
            if len(keep) != len(ch_names):
                arr = arr[:, :, keep, :]
                ch_names = [ch_names[i] for i in keep]
            # Optional time window (ms). times are in seconds.
            if self._time_idx is None:
                if self.time_window_ms is not None:
                    lo, hi = self.time_window_ms
                    self._time_idx = np.where((times >= lo / 1000.0) &
                                              (times <= hi / 1000.0))[0]
                else:
                    self._time_idx = np.arange(len(times))
                self._times = times[self._time_idx]
                self._ch_names = ch_names
            arr = arr[:, :, :, self._time_idx].astype(np.float32)
            self._data[source] = np.ascontiguousarray(arr)
        return self._data[source]

    # -- shape accessors ----------------------------------------------------
    @property
    def ch_names(self) -> List[str]:
        if self._ch_names is None:
            self._load(_TRAIN)
        return self._ch_names

    @property
    def channels(self) -> int:
        return len(self.ch_names)

    @property
    def times(self) -> np.ndarray:
        if self._times is None:
            self._load(_TRAIN)
        return self._times

    @property
    def n_times(self) -> int:
        return len(self.times)

    @property
    def feature_size(self) -> int:
        """Flat descriptor C*T (used as the EEG analogue of voxel count)."""
        return self.channels * self.n_times

    def num_images(self, source: str = _TRAIN) -> int:
        return int(self._load(source).shape[0])

    def n_reps(self, source: str = _TRAIN) -> int:
        return int(self._load(source).shape[1])

    # -- image alignment ----------------------------------------------------
    def image_path(self, source: str, img_index: int) -> str:
        sub = "training_images" if source == _TRAIN else "test_images"
        concept = self._img_concepts(source)[img_index]
        fname = self._img_files(source)[img_index]
        return str(self.image_set / sub / concept / fname)

    def image_id(self, source: str, img_index: int) -> str:
        return parse_image_id(self._img_files(source)[img_index])

    # -- signal access ------------------------------------------------------
    def get_trial(self, source: str, img_index: int, rep: int) -> np.ndarray:
        return self._load(source)[img_index, rep]           # [C, T]

    def get_mean(self, source: str, img_index: int) -> np.ndarray:
        return self._load(source)[img_index].mean(axis=0)   # [C, T]

    def get_signal(self, source: str, img_index: int, rep: int) -> np.ndarray:
        """``rep == -1`` returns the mean over repetitions; else that trial."""
        if int(rep) < 0:
            return self.get_mean(source, img_index)
        return self.get_trial(source, img_index, int(rep))

    def get_fmri(self, local_index: int, source: str = _TRAIN) -> np.ndarray:
        """Per-image mean signal ``[C, T]`` (name/signature mirror
        :meth:`SubjectData.get_fmri` so the shared ``load_subject_matrices`` and
        image-level evaluation work unchanged). ``local_index`` is the image
        index; evaluation always operates at the image level for EEG.
        """
        return self.get_mean(source, int(local_index))


class EegDataset(Dataset):
    """Split-scoped EEG dataset backed by a metadata frame + per-subject readers.

    Each frame row carries ``source`` ('train'/'test'), ``img_index`` and ``rep``
    (``rep == -1`` → mean over repetitions) plus ``feat_idx`` (the unique-image
    rank within the (subject, split) group) so precomputed CLIP/PCA arrays stay
    aligned: ``arr[feat_idx]`` is the target for the row's image.
    """

    def __init__(self, frame, subjects: Dict[str, EegSubjectData],
                 normalizers=None, features=None, image_transform=None,
                 return_fmri: bool = True, return_image: bool = False,
                 return_paths: bool = True):
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
        self._img = self.frame["img_index"].to_numpy()
        self._rep = self.frame["rep"].to_numpy()
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
        img_index = int(self._img[i])
        rep = int(self._rep[i])
        feat_idx = int(self._feat[i])
        sample = {"index": i, "subject_id": subj,
                  "image_id": str(self._image_id[i])}

        if self.return_fmri:
            sig = self.subjects[subj].get_signal(source, img_index, rep)
            norm = self.normalizers.get(subj)
            if norm is not None:
                sig = norm.transform(sig)
            sample["fmri"] = torch.from_numpy(np.ascontiguousarray(sig)).float()

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
