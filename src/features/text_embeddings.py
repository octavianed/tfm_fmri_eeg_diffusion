"""Precompute the frozen SD text-encoder embeddings of the caption prompts (§11).

Why precompute at all — the text encoder loads fine with the current stack
(verified: transformers 5.6.2 + diffusers 0.37.1 load SD-1.5's ``CLIPTextModel``
without the historical ``ignore_mismatched_sizes`` failure), so this is no longer
a *workaround*. It is kept because it is simply better:

* the adapter's training loop never has to hold a second frozen encoder in VRAM
  next to the UNet (and the ControlNet);
* prompts are encoded **once** instead of once per epoch;
* the cache is keyed by a hash of ``(template, caption_field, tokenizer,
  text_encoder, max_length, ...)`` so an incompatible embedding can never be
  reused by accident (§11.1);
* generation stays reproducible even if the local diffusers/transformers stack
  changes underneath.

Storage design — captions repeat a lot (THINGS-EEG2 has 16 540 training images
but only 1 654 concepts; NSD's ``prompt_categories`` repeat across images too).
So the cache stores the **unique** prompts once (``unique_embeds.npy``,
``[U, 77, D]``) plus one small ``int32`` index array per ``(subject, split)``.
That turns a 2 GB dump into ~200 MB for EEG, and makes the permuted-text control
just another index array.

Alignment follows the project convention exactly: row ``k`` of an index array is
``feat_idx == k`` within ``(subject, split)``, i.e. the same row as the CLIP and
VAE-PCA features.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.captions import (build_split_prompts, generic_prompt, modality,
                             permuted_prompts, resolve_caption_field,
                             resolve_template, text_mode)
from ..utils import get_logger, save_json
from ..utils.paths import text_embedding_dir

logger = get_logger("text_embeddings")

EMPTY_PROMPT = ""


# --- cache identity ---------------------------------------------------------
def text_cache_signature(cfg) -> dict:
    """Everything that changes the embeddings — hashed into the cache path."""
    return {
        "modality": modality(cfg),
        "text_mode": text_mode(cfg),
        "caption_field": resolve_caption_field(cfg),
        "template": resolve_template(cfg),
        "text_encoder": str(cfg.get("generation.sd_model", "")),
        "tokenizer": str(cfg.get("generation.text.tokenizer", "auto")),
        "max_length": cfg.get("generation.text.max_length", None),
        "normalize_caption": bool(cfg.get("generation.text.normalize_caption", True)),
        "permutation_seed": int(cfg.get("generation.text.permutation_seed", 42)),
        "permutation_source": str(cfg.get("generation.text.permutation_source",
                                          "derived")),
        "generic_prompt": generic_prompt(cfg),
    }


def text_cache_hash(cfg) -> str:
    payload = json.dumps(text_cache_signature(cfg), sort_keys=True,
                         ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def text_cache_dir(cfg) -> Path:
    field = resolve_caption_field(cfg) or "none"
    return text_embedding_dir(cfg, modality(cfg),
                              str(cfg.get("generation.sd_model", "")),
                              field, text_cache_hash(cfg))


# --- frozen text encoder ----------------------------------------------------
@dataclass
class TextEncoderBundle:
    tokenizer: object
    model: object
    max_length: int
    embed_dim: int


def load_text_encoder(cfg, device=None) -> TextEncoderBundle:
    """SD's tokenizer + text encoder, frozen (never trained — spec §7.1)."""
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    from ..utils import get_device
    device = device or get_device(cfg.get("runtime.device", "auto"))
    model_name = str(cfg.get("generation.sd_model",
                             "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    tok_name = cfg.get("generation.text.tokenizer", "auto")
    tok_name = model_name if tok_name in (None, "auto") else str(tok_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(tok_name, subfolder="tokenizer")
    model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder",
                                          torch_dtype=dtype).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    max_length = int(cfg.get("generation.text.max_length", None)
                     or tokenizer.model_max_length)
    return TextEncoderBundle(tokenizer, model,
                             max_length=max_length,
                             embed_dim=int(model.config.hidden_size))


def encode_prompts(bundle: TextEncoderBundle, prompts: Sequence[str],
                   device=None, batch_size: int = 64) -> np.ndarray:
    """``[N, max_length, D]`` float16 embeddings of the given prompt strings."""
    import torch
    device = device or next(bundle.model.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            chunk = list(prompts[start:start + batch_size])
            ids = bundle.tokenizer(
                chunk, padding="max_length", max_length=bundle.max_length,
                truncation=True, return_tensors="pt").input_ids.to(device)
            emb = bundle.model(ids)[0]
            out.append(emb.to(torch.float16).cpu().numpy())
    if not out:
        return np.zeros((0, bundle.max_length, bundle.embed_dim), np.float16)
    return np.concatenate(out, axis=0)


# --- precompute -------------------------------------------------------------
def precompute_text_embeddings(cfg, datamodule,
                               splits: Sequence[str] = ("train", "val", "test"),
                               overwrite: bool = False, device=None) -> dict:
    """Build (or reuse) the text-embedding cache for every (subject, split).

    Encodes the union of all prompts ONCE: the correct prompts, the permuted-text
    control prompts, the empty prompt (CFG negative branch) and the generic
    control prompt of §18.2.
    """
    if text_mode(cfg) == "none":
        logger.info("generation.text.mode=none — nothing to precompute.")
        return {"enabled": False}

    out_dir = text_cache_dir(cfg)
    meta_path = out_dir / "meta.json"
    if meta_path.exists() and not overwrite:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        have = set(tuple(x) for x in meta.get("pairs", []))
        need = {(s, sp) for s in datamodule.subjects for sp in splits}
        if need.issubset(have):
            logger.info("[skip] text embeddings already cached: %s", out_dir)
            return {"enabled": True, "dir": str(out_dir), "meta": meta,
                    "skipped": True}
        logger.info("Cache exists but misses %s; rebuilding.", sorted(need - have))

    field = resolve_caption_field(cfg)
    template = resolve_template(cfg)
    logger.info("Precomputing text embeddings | mode=%s field=%s template=%r",
                text_mode(cfg), field, template)

    # 1) collect the prompts of every (subject, split), correct and permuted
    per_pair: Dict[tuple, Dict[str, List[str]]] = {}
    for subject in datamodule.subjects:
        for split in splits:
            if len(datamodule.subject_split_frame(subject, split)) == 0:
                continue
            per_pair[(subject, split)] = {
                "correct": build_split_prompts(cfg, datamodule, subject, split,
                                               field, template),
                "permuted": permuted_prompts(cfg, datamodule, subject, split,
                                             field, template)}

    # 2) unique vocabulary (sorted -> deterministic row order across runs)
    specials = [EMPTY_PROMPT, generic_prompt(cfg)]
    vocab = sorted({p for pair in per_pair.values() for lst in pair.values()
                    for p in lst} | set(specials))
    row_of = {p: i for i, p in enumerate(vocab)}
    logger.info("%d unique prompt(s) over %d (subject, split) pair(s)",
                len(vocab), len(per_pair))

    # 3) encode once
    bundle = load_text_encoder(cfg, device)
    embeds = encode_prompts(bundle, vocab, batch_size=int(
        cfg.get("generation.text.encode_batch_size", 64)))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "unique_embeds.npy", embeds)
    (out_dir / "prompts.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=0), encoding="utf-8")
    pairs = []
    for (subject, split), prompts in per_pair.items():
        for kind, suffix in (("correct", ""), ("permuted", "_permuted")):
            idx = np.asarray([row_of[p] for p in prompts[kind]], dtype=np.int32)
            np.save(out_dir / f"{subject}_{split}{suffix}.npy", idx)
        pairs.append([subject, split])
        logger.info("[text] %s/%s -> %d prompts (%d unique)", subject, split,
                    len(prompts["correct"]), len(set(prompts["correct"])))

    meta = dict(text_cache_signature(cfg))
    meta.update({"hash": text_cache_hash(cfg), "num_unique": len(vocab),
                 "seq_len": int(embeds.shape[1]), "embed_dim": int(embeds.shape[2]),
                 "special_rows": {"empty": row_of[EMPTY_PROMPT],
                                  "generic": row_of[generic_prompt(cfg)]},
                 "pairs": pairs})
    save_json(meta, meta_path)
    logger.info("Text embeddings -> %s %s", out_dir, tuple(embeds.shape))
    return {"enabled": True, "dir": str(out_dir), "meta": meta, "skipped": False}


# --- load -------------------------------------------------------------------
class TextEmbeddingCache:
    """Read-only accessor over a precomputed text-embedding cache directory."""

    def __init__(self, directory, meta: dict):
        self.dir = Path(directory)
        self.meta = meta
        self._embeds: Optional[np.ndarray] = None
        self._prompts: Optional[List[str]] = None
        self._index: Dict[tuple, np.ndarray] = {}

    # -- lazily loaded arrays ------------------------------------------------
    @property
    def embeds(self) -> np.ndarray:
        if self._embeds is None:
            self._embeds = np.load(self.dir / "unique_embeds.npy")
        return self._embeds

    @property
    def prompts(self) -> List[str]:
        if self._prompts is None:
            self._prompts = json.loads(
                (self.dir / "prompts.json").read_text(encoding="utf-8"))
        return self._prompts

    @property
    def seq_len(self) -> int:
        return int(self.meta["seq_len"])

    @property
    def embed_dim(self) -> int:
        return int(self.meta["embed_dim"])

    def index(self, subject: str, split: str, permuted: bool = False) -> np.ndarray:
        key = (subject, split, bool(permuted))
        if key not in self._index:
            suffix = "_permuted" if permuted else ""
            path = self.dir / f"{subject}_{split}{suffix}.npy"
            if not path.exists():
                raise FileNotFoundError(
                    f"No text embeddings for {subject}/{split} at {path}. Run "
                    f"scripts/13_precompute_text_embeddings.py with this config.")
            self._index[key] = np.load(path)
        return self._index[key]

    # -- lookups -------------------------------------------------------------
    def rows(self, subject: str, split: str, feat_idx, permuted: bool = False
             ) -> np.ndarray:
        """``[B, L, D]`` embeddings for the given ``feat_idx`` values."""
        idx = self.index(subject, split, permuted)
        return self.embeds[idx[np.asarray(feat_idx, dtype=int)]]

    def prompt(self, subject: str, split: str, feat_idx: int,
               permuted: bool = False) -> str:
        return self.prompts[int(self.index(subject, split, permuted)[int(feat_idx)])]

    def special(self, name: str) -> np.ndarray:
        """``[1, L, D]`` embedding of a special prompt (``empty`` | ``generic``)."""
        row = int(self.meta["special_rows"][name])
        return self.embeds[row:row + 1]


def load_text_cache(cfg, required: bool = True) -> Optional[TextEmbeddingCache]:
    """Open the cache for the current config, validating its hash (§11.1)."""
    if text_mode(cfg) == "none":
        return None
    out_dir = text_cache_dir(cfg)
    meta_path = out_dir / "meta.json"
    if not meta_path.exists():
        if not required:
            return None
        raise FileNotFoundError(
            f"No text-embedding cache at {out_dir}.\nRun:\n"
            f"  python scripts/13_precompute_text_embeddings.py --config <this config>")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = text_cache_hash(cfg)
    if str(meta.get("hash")) != expected:
        raise ValueError(
            f"Text-embedding cache {out_dir} was built for hash "
            f"{meta.get('hash')} but this config hashes to {expected}. "
            f"Re-run scripts/13_precompute_text_embeddings.py (the template, "
            f"caption field, tokenizer or max_length changed).")
    return TextEmbeddingCache(out_dir, meta)
