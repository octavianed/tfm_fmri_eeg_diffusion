"""Textual captions per stimulus image, aligned to ``feat_idx`` (multimodal §9-§10).

The generation stack can optionally condition Stable Diffusion on a *weak* or
*oracle* text prompt built from a caption of the image the subject actually saw.
This module is the single place that answers three questions:

1. **Where do the captions live and which column is "the caption"?**
   The CSVs were produced by the two notebooks in ``notebooks/{fMRI,EEG}/``:

   * fMRI / NSD-Algonauts — ``<root>/auxiliar/generated_captions/<subj>_{train,test}_captions.csv``
     with ``prompt_categories`` (weak: ``"a chair, couch and laptop"``) and
     ``primary_caption`` (oracle: the first COCO caption).
   * EEG / THINGS-EEG2 — ``<root>/image_set/generated_captions/thingseeg2_{training,test}_image_captions.csv``
     with ``primary_caption`` = the THINGS concept name (``"Aardvark"``). There is
     no more detailed caption, so **oracle == weak** for EEG (§10.3).

2. **How is a caption tied to a brain sample?** By ``image_id`` (the image
   filename stem), which is exactly the key the datamodules already put in their
   frames. The lookup is then re-indexed to ``feat_idx`` order, the same
   convention CLIP/VAE features follow, and validated to be exhaustive — a
   missing caption raises instead of silently shifting the alignment.

3. **What is a "permuted caption"?** A derangement of the *same caption family*
   within the *same (subject, split)* (§16). Computed here with the same
   :func:`~src.utils.permutation.derangement` used for the brain permutation, so
   it can never leak across splits.

Why derive the permutation instead of reading ``permuted_caption_seed42`` from
the CSVs: those columns are valid derangements of ``primary_caption``, but they
were shuffled over the *whole* THINGS/NSD training set, which this project then
carves into train **and** val — so a val sample could receive a train caption.
There is also no ``permuted_prompt_categories_seed42`` column, i.e. no permuted
counterpart for the fMRI *weak* family at all. Deriving it covers every family
uniformly and stays inside the split. Set
``generation.text.permutation_source: column`` to use the CSV column instead
(only allowed for ``primary_caption``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils import get_logger
from ..utils.permutation import derangement

logger = get_logger("captions")

#: Caption column resolved from ``generation.text.mode`` per modality (§13).
CAPTION_FIELDS: Dict[str, Dict[str, str]] = {
    "fmri": {"weak": "prompt_categories", "oracle": "primary_caption"},
    "eeg": {"weak": "primary_caption", "oracle": "primary_caption"},
}

#: CSV column holding the precomputed permutation, per caption family. Only
#: ``primary_caption`` has one (see the module docstring).
PERMUTED_COLUMNS: Dict[str, str] = {"primary_caption": "permuted_caption_seed42"}

TEXT_MODES = ("none", "weak", "oracle")
DEFAULT_TEMPLATE = "Image of {caption}"
#: Neutral prompt for the "generic-text brain-only" control of §18.2 — a control
#: that is deliberately NOT the same as having no text at all.
DEFAULT_GENERIC_CAPTION = "something"


# --- config resolution ------------------------------------------------------
def modality(cfg) -> str:
    return str(cfg.get("dataset.modality", "fmri")).lower()


def text_mode(cfg) -> str:
    """``none`` | ``weak`` | ``oracle``, honouring ``generation.text.enabled``."""
    mode = str(cfg.get("generation.text.mode", "none")).lower()
    if mode not in TEXT_MODES:
        raise ValueError(f"generation.text.mode must be one of {TEXT_MODES}, "
                         f"got {mode!r}")
    enabled = cfg.get("generation.text.enabled", None)
    if enabled is not None and not bool(enabled):
        return "none"
    return mode


def resolve_caption_field(cfg) -> Optional[str]:
    """Column name backing the configured text mode (``None`` when disabled)."""
    mode = text_mode(cfg)
    if mode == "none":
        return None
    field = cfg.get("generation.text.caption_field", "auto")
    if field in (None, "auto", "Auto", "AUTO"):
        mod = modality(cfg)
        try:
            return CAPTION_FIELDS[mod][mode]
        except KeyError as exc:  # pragma: no cover - guarded by text_mode()
            raise ValueError(f"No caption field for modality={mod!r} "
                             f"mode={mode!r}") from exc
    return str(field)


def resolve_template(cfg) -> str:
    template = str(cfg.get("generation.text.template", DEFAULT_TEMPLATE))
    if "{caption}" not in template:
        raise ValueError("generation.text.template must contain '{caption}', "
                         f"got {template!r}")
    return template


def generic_prompt(cfg) -> str:
    """The §18.2 control prompt, built with the SAME template as the real ones."""
    caption = str(cfg.get("generation.text.generic_caption",
                          DEFAULT_GENERIC_CAPTION))
    return build_prompt(resolve_template(cfg), caption)


def captions_dir(cfg) -> Path:
    """Folder with the caption CSVs (``dataset.captions_dir`` or the default)."""
    explicit = cfg.get("dataset.captions_dir", None)
    if explicit:
        return Path(str(explicit))
    root = Path(str(cfg.get("dataset.root_dir", ".")))
    if modality(cfg) == "eeg":
        return root / "image_set" / "generated_captions"
    return root / "auxiliar" / "generated_captions"


# --- prompt building --------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize_caption(caption: str) -> str:
    """Whitespace/punctuation tidy-up ONLY — the caption is never reworded (§10.2)."""
    text = _WS.sub(" ", str(caption)).strip()
    # A trailing period would give "Image of a man riding a bike.." with some
    # templates; drop only that, never interior punctuation.
    return text.rstrip(" .") or text


def build_prompt(template: str, caption: str, normalize: bool = True) -> str:
    caption = normalize_caption(caption) if normalize else str(caption)
    return template.format(caption=caption)


# --- caption tables ---------------------------------------------------------
def _caption_csv(cfg, subject: str, source: str) -> Path:
    """CSV holding the captions for one (subject, dataset source)."""
    directory = captions_dir(cfg)
    if modality(cfg) == "eeg":
        # Shared across subjects: THINGS-EEG2 shows every subject the same images.
        stem = "training" if source == "train" else "test"
        return directory / f"thingseeg2_{stem}_image_captions.csv"
    return directory / f"{subject}_{source}_captions.csv"


def _read_caption_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Caption file not found: {path}\nGenerate it with the notebooks in "
            f"notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb or "
            f"notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb, or point "
            f"dataset.captions_dir at the folder that holds it.")
    df = pd.read_csv(path)
    if "image_id" in df.columns:
        key = df["image_id"].astype(str)
    elif "filename" in df.columns:
        key = df["filename"].astype(str).map(lambda p: Path(p).stem)
    else:
        raise ValueError(f"{path} has neither 'image_id' nor 'filename'")
    df = df.copy()
    df["_key"] = key
    return df


def load_caption_metadata(cfg, datamodule, subject: str, split: str,
                          fields: Optional[List[str]] = None) -> pd.DataFrame:
    """Captions for one ``(subject, split)`` in ``feat_idx`` order.

    Returns a frame with ``feat_idx``, ``image_id`` and the requested caption
    columns, with exactly one row per brain sample of the split. Raises if any
    image lacks a caption — silently dropping rows would shift ``feat_idx`` and
    mis-pair captions with brain data for every later sample.
    """
    frame = datamodule.subject_split_frame(subject, split)
    if len(frame) == 0:
        return pd.DataFrame(columns=["feat_idx", "image_id"] + list(fields or []))

    sources = (frame["source"].astype(str).unique().tolist()
               if "source" in frame.columns else ["train"])
    tables = [_read_caption_csv(_caption_csv(cfg, subject, src)) for src in sources]
    table = pd.concat(tables, ignore_index=True).drop_duplicates("_key", keep="first")
    lookup = table.set_index("_key")

    ids = frame["image_id"].astype(str)
    missing = [i for i in ids if i not in lookup.index]
    if missing:
        raise KeyError(
            f"No caption for {len(missing)} image(s) of {subject}/{split} "
            f"(e.g. {missing[:3]}). Caption folder: {captions_dir(cfg)}")

    picked = lookup.loc[ids.tolist()]
    out = pd.DataFrame({"feat_idx": frame["feat_idx"].to_numpy(dtype=int),
                        "image_id": ids.to_numpy()})
    for col in (fields or []):
        if col not in picked.columns:
            raise KeyError(
                f"Caption column {col!r} not in {_caption_csv(cfg, subject, sources[0])}. "
                f"Available: {sorted(c for c in picked.columns if not c.startswith('_'))}")
        out[col] = picked[col].astype(str).to_numpy()
    return out.sort_values("feat_idx").reset_index(drop=True)


def validate_caption_alignment(cfg, datamodule, subject: str, split: str,
                               field: str) -> dict:
    """Check the captions cover the split 1:1 and report a few examples (Test 4)."""
    frame = load_caption_metadata(cfg, datamodule, subject, split, [field])
    expected = len(datamodule.subject_split_frame(subject, split))
    ok = len(frame) == expected and frame["feat_idx"].is_monotonic_increasing \
        and frame["feat_idx"].tolist() == list(range(expected))
    return {"subject": subject, "split": split, "field": field,
            "num_rows": len(frame), "num_expected": expected, "aligned": bool(ok),
            "num_unique_captions": int(frame[field].nunique()) if len(frame) else 0,
            "examples": frame.head(3).to_dict("records")}


# --- prompts ----------------------------------------------------------------
def build_split_prompts(cfg, datamodule, subject: str, split: str,
                        field: Optional[str] = None,
                        template: Optional[str] = None) -> List[str]:
    """Final prompt strings for ``(subject, split)``, in ``feat_idx`` order."""
    field = field or resolve_caption_field(cfg)
    if field is None:
        return []
    template = template or resolve_template(cfg)
    normalize = bool(cfg.get("generation.text.normalize_caption", True))
    frame = load_caption_metadata(cfg, datamodule, subject, split, [field])
    return [build_prompt(template, c, normalize) for c in frame[field].tolist()]


def caption_permutation(n: int, seed: int, subject: str = "", split: str = "",
                        field: str = "") -> np.ndarray:
    """Within-split derangement of ``range(n)`` for the permuted-text control."""
    return derangement(n, seed, salt=f"text|{subject}|{split}|{field}")


def permuted_prompts(cfg, datamodule, subject: str, split: str,
                     field: Optional[str] = None,
                     template: Optional[str] = None) -> List[str]:
    """Permuted-caption prompts (§16): same family, same split, seed 42."""
    field = field or resolve_caption_field(cfg)
    if field is None:
        return []
    seed = int(cfg.get("generation.text.permutation_seed", 42))
    source = str(cfg.get("generation.text.permutation_source", "derived")).lower()
    if source == "column":
        column = PERMUTED_COLUMNS.get(field)
        if column is None:
            raise ValueError(
                f"generation.text.permutation_source='column' but there is no "
                f"precomputed permutation for caption field {field!r} "
                f"(only {sorted(PERMUTED_COLUMNS)}). Use 'derived'.")
        template = template or resolve_template(cfg)
        normalize = bool(cfg.get("generation.text.normalize_caption", True))
        frame = load_caption_metadata(cfg, datamodule, subject, split, [column])
        logger.warning(
            "Using the precomputed %s column: it was shuffled over the WHOLE "
            "dataset split, so a val sample may receive a train caption "
            "(the 'derived' default stays inside the split).", column)
        return [build_prompt(template, c, normalize) for c in frame[column].tolist()]
    if source != "derived":
        raise ValueError("generation.text.permutation_source must be "
                         f"'derived' or 'column', got {source!r}")
    prompts = build_split_prompts(cfg, datamodule, subject, split, field, template)
    perm = caption_permutation(len(prompts), seed, subject, split, field)
    return [prompts[i] for i in perm]
