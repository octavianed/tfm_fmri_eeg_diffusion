"""Centralized filesystem layout for data, features and experiment outputs.

All paths derive from the ``paths`` block of the config (with sensible
defaults) so the whole project uses a single source of truth for locations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentPaths:
    """Standard output layout for a single experiment (see spec §12)."""

    root: Path
    checkpoints: Path = field(init=False)
    logs: Path = field(init=False)
    metrics: Path = field(init=False)
    figures: Path = field(init=False)
    embeddings: Path = field(init=False)
    lowlevel: Path = field(init=False)
    generated: Path = field(init=False)
    grids: Path = field(init=False)
    metadata: Path = field(init=False)
    report: Path = field(init=False)

    def __post_init__(self):
        self.root = Path(self.root)
        self.checkpoints = self.root / "checkpoints"
        self.logs = self.root / "logs"
        self.metrics = self.root / "metrics"
        self.figures = self.root / "figures"
        self.embeddings = self.root / "embeddings"
        self.lowlevel = self.root / "lowlevel"
        self.generated = self.root / "generated"
        self.grids = self.root / "grids"
        self.metadata = self.root / "metadata"
        self.report = self.root / "report"

    def ensure(self) -> "ExperimentPaths":
        for p in (self.root, self.checkpoints, self.logs, self.metrics,
                  self.figures, self.embeddings, self.lowlevel, self.generated,
                  self.grids, self.metadata, self.report):
            p.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def last_ckpt(self) -> Path:
        return self.checkpoints / "last.pt"

    @property
    def best_ckpt(self) -> Path:
        return self.checkpoints / "best.pt"

    @property
    def resume_history(self) -> Path:
        return self.logs / "resume_history.jsonl"

    @property
    def train_log(self) -> Path:
        return self.logs / "train_log.csv"


def experiment_dir(cfg) -> Path:
    output_dir = cfg.get("paths.output_dir", "outputs")
    name = cfg.get("experiment.name") or cfg.get("project.name", "experiment")
    return Path(output_dir) / name


def get_experiment_paths(cfg, ensure: bool = True) -> ExperimentPaths:
    ep = ExperimentPaths(experiment_dir(cfg))
    if ensure:
        ep.ensure()
    return ep


# --- data / feature locations ---------------------------------------------
def data_dir(cfg) -> Path:
    return Path(cfg.get("paths.data_dir", "data"))


def processed_dir(cfg) -> Path:
    return Path(cfg.get("paths.processed_dir", "data/processed"))


def features_dir(cfg) -> Path:
    return Path(cfg.get("paths.features_dir", "data/features"))


def splits_dir(cfg) -> Path:
    return Path(cfg.get("paths.splits_dir", "data/splits"))


def _safe(name: str) -> str:
    return str(name).replace("/", "-").replace(" ", "_")


def normalization_path(cfg, subject: str) -> Path:
    d = processed_dir(cfg) / "normalization"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_fmri_norm.npz"


def metadata_path(cfg, selection_tag: str) -> Path:
    processed_dir(cfg).mkdir(parents=True, exist_ok=True)
    return processed_dir(cfg) / f"metadata_{selection_tag}.csv"


def eeg_preproc_dir(cfg, variant: str) -> Path:
    """Cache root of an own-preprocessing EEG variant built from the raw data.

    Layout: ``data/processed/eeg_preproc/<variant>/<subject>/`` holding
    ``preprocessed_eeg_{training,test}.npy`` (same contract as the official
    derivatives) plus ``metadata.json`` and ``qc/``.
    """
    d = processed_dir(cfg) / "eeg_preproc" / _safe(variant)
    d.mkdir(parents=True, exist_ok=True)
    return d


def clip_feature_path(cfg, subject: str, split: str) -> Path:
    model = _safe(cfg.get("features.clip_model", "ViT-L-14"))
    d = features_dir(cfg) / "clip" / model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_{split}.npy"


def vae_latent_path(cfg, subject: str, split: str) -> Path:
    model = _safe(cfg.get("features.vae_model", "sd-v1-5"))
    d = features_dir(cfg) / "vae" / model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_{split}_latents.npy"


def vae_pca_feature_path(cfg, subject: str, split: str) -> Path:
    model = _safe(cfg.get("features.vae_model", "sd-v1-5"))
    d = features_dir(cfg) / "vae" / model
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_{split}_pca.npy"


def text_embedding_dir(cfg, modality: str, text_encoder: str,
                       caption_field: str, cache_hash: str) -> Path:
    """Cache root of the SD text-encoder embeddings of the caption prompts.

    Layout: ``data/features/text/<modality>/<text_encoder>/<caption_field>/<hash>/``.
    The hash covers template/tokenizer/max_length/... (see
    :func:`src.features.text_embeddings.text_cache_signature`), so two
    incompatible prompt setups can never share a directory (§11.1).

    Deliberately independent of the EEG preprocessing variant: prompts depend
    only on the images and the split, exactly like the CLIP/VAE features, so all
    raw-preprocessing ablations share one cache.
    """
    return (features_dir(cfg) / "text" / _safe(modality) / _safe(text_encoder)
            / _safe(caption_field) / _safe(cache_hash))


def controlnet_condition_dir(cfg, condition_type: str, subject: str,
                             split: str) -> Path:
    """Cache of the ControlNet spatial conditions derived from the GT VAE-PCA.

    Namespaced by ``features.vae_model`` and ``features.pca_dim`` because the
    condition is computed *after* the PCA bottleneck — a different VAE or a
    different number of components produces a different coarse reconstruction,
    hence different edges.
    """
    model = _safe(cfg.get("features.vae_model", "sd-v1-5"))
    pca_dim = int(cfg.get("features.pca_dim", 512))
    d = (features_dir(cfg) / "controlnet" / _safe(condition_type)
         / f"{model}_pca{pca_dim}" / _safe(subject) / _safe(split))
    d.mkdir(parents=True, exist_ok=True)
    return d


def vae_pca_model_path(cfg, subject: str) -> Path:
    # Namespaced by features.vae_model (like vae_latent_path/vae_pca_feature_path)
    # so fitting PCA against a different VAE/SD backbone (e.g. switching from
    # SD-1.5 to SD-2.1) never overwrites a previously fitted PCA model.
    model = _safe(cfg.get("features.vae_model", "sd-v1-5"))
    d = processed_dir(cfg) / "pca"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{subject}_{model}_vae_pca_model.pkl"
