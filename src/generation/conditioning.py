"""Multimodal conditioning: how text, neural tokens and ControlNet combine (§3-§5).

Three architectures, selected by ``generation.conditioning_architecture``:

``legacy_adapter``
    The historical behaviour. ``CLIP_pred -> TokenAdapter -> prompt_embeds``,
    no caption, no tokenizer, no text encoder, no ControlNet. Existing adapter
    checkpoints keep working unchanged — this is the backwards-compatibility
    path of the whole system.

``text_adapter_concat``
    ``[text tokens ; neural tokens]`` concatenated along the sequence axis and
    handed to the frozen UNet through ``prompt_embeds``. Cross-attention is
    length-agnostic, so ``[B, L_text + K, D]`` is a valid condition.

``text_adapter_concat_controlnet``
    Exactly the above **plus** a frozen pretrained ControlNet fed with a spatial
    condition derived from the low-level (VAE-PCA) prediction. ControlNet does
    not replace the concatenation: it adds residuals inside the UNet while text
    and neural tokens keep arriving through cross-attention (§5.1).

An experimental *condition* is a triple (text state, semantic brain state,
structural brain state) — :class:`ConditionSpec`. That is what lets §35's matrix
("¿aporta el CLIP cerebral?" vs "¿aporta la predicción VAE-PCA cerebral?") be
expressed without duplicating generation code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..data.captions import resolve_caption_field, resolve_template, text_mode
from ..utils import get_logger

logger = get_logger("conditioning")

LEGACY = "legacy_adapter"
TEXT_CONCAT = "text_adapter_concat"
TEXT_CONCAT_CONTROLNET = "text_adapter_concat_controlnet"
ARCHITECTURES = (LEGACY, TEXT_CONCAT, TEXT_CONCAT_CONTROLNET)

#: Brain states available for each branch. ``zero`` keeps the historical Exp5
#: semantics on the semantic branch (a null BRAIN vector fed to the decoder,
#: not a null CLIP embedding) and means "no ControlNet residuals" on the
#: structural branch (§22.3).
BRAIN_STATES = ("correct", "permuted", "zero")
TEXT_STATES = ("none", "correct", "permuted", "generic")


@dataclass(frozen=True)
class ConditionSpec:
    """One experimental condition of the generation ablation."""

    name: str
    text: str = "none"          # none | correct | permuted | generic
    semantic: str = "correct"   # correct | permuted | zero  (CLIP_pred branch)
    structural: str = "none"    # correct | permuted | zero | none (low_pred branch)

    def __post_init__(self):
        if self.text not in TEXT_STATES:
            raise ValueError(f"{self.name}: text={self.text!r} not in {TEXT_STATES}")
        if self.semantic not in BRAIN_STATES:
            raise ValueError(f"{self.name}: semantic={self.semantic!r}")
        if self.structural not in BRAIN_STATES + ("none",):
            raise ValueError(f"{self.name}: structural={self.structural!r}")

    def to_dict(self) -> dict:
        return asdict(self)


# --- architecture resolution ------------------------------------------------
def resolve_architecture(cfg) -> str:
    arch = str(cfg.get("generation.conditioning_architecture", LEGACY))
    if arch not in ARCHITECTURES:
        raise ValueError(f"generation.conditioning_architecture must be one of "
                         f"{ARCHITECTURES}, got {arch!r}")
    return arch


def uses_text(cfg) -> bool:
    """True when the architecture concatenates text embeddings."""
    return resolve_architecture(cfg) != LEGACY and text_mode(cfg) != "none"


def uses_controlnet(cfg) -> bool:
    arch = resolve_architecture(cfg)
    if arch != TEXT_CONCAT_CONTROLNET:
        return False
    return bool(cfg.get("generation.controlnet.enabled", True))


def fusion_type(cfg) -> str:
    kind = str(cfg.get("generation.fusion.type", "concat")).lower()
    if kind != "concat":
        raise ValueError(f"Only fusion.type=concat is implemented, got {kind!r}")
    return kind


def _dual_key(cfg, nested: str, flat: str, default):
    """Read a setting written either nested (``generation.adapter.x``, the shape
    proposed by the requirements) or flat (``generation.adapter_x``, the shape
    the existing configs/sweeps use). Raises when both are set and disagree —
    silently picking one would make a run's conditioning strength ambiguous."""
    n = cfg.get(nested, None)
    f = cfg.get(flat, None)
    if n is not None and f is not None and type(n)(f) != n:
        raise ValueError(f"Conflicting config: {nested}={n!r} vs {flat}={f!r}. "
                         f"Set only one of them.")
    if n is not None:
        return n
    return default if f is None else f


def num_neural_tokens(cfg) -> int:
    """Number of pseudo-tokens the TokenAdapter emits (K)."""
    return int(_dual_key(cfg, "generation.adapter.num_tokens",
                         "generation.num_tokens", 77))


def adapter_normalize_input(cfg) -> bool:
    return bool(_dual_key(cfg, "generation.adapter.normalize_input",
                          "generation.adapter_normalize_input", False))


def adapter_input_scale(cfg) -> float:
    return float(_dual_key(cfg, "generation.adapter.input_scale",
                           "generation.adapter_input_scale", 1.0))


def controlnet_settings(cfg) -> dict:
    """Resolved ControlNet block (never hard-codes a checkpoint id — §6.3)."""
    enabled = uses_controlnet(cfg)
    model = cfg.get("generation.controlnet.model", None)
    if enabled and not model:
        raise ValueError(
            "generation.controlnet.enabled=true but generation.controlnet.model "
            "is empty. Set an SD-compatible ControlNet checkpoint explicitly, "
            "e.g. 'lllyasviel/sd-controlnet-canny' for SD-1.5.")
    return {
        "enabled": enabled,
        "model": str(model or ""),
        "condition_type": str(cfg.get("generation.controlnet.condition_type",
                                      "canny")).lower(),
        "conditioning_scale": float(cfg.get("generation.controlnet.conditioning_scale",
                                            0.5)),
        "guidance_start": float(cfg.get("generation.controlnet.guidance_start", 0.0)),
        "guidance_end": float(cfg.get("generation.controlnet.guidance_end", 1.0)),
        "training_condition_source": str(
            cfg.get("generation.controlnet.training_condition_source",
                    "gt_vae_pca_reconstruction")),
    }


# --- condition sets ---------------------------------------------------------
def _default_specs(cfg) -> List[ConditionSpec]:
    """The condition matrix implied by the architecture (§34, §35)."""
    arch = resolve_architecture(cfg)
    text = "correct" if uses_text(cfg) else "none"
    if arch == LEGACY:
        return [ConditionSpec("correct", text, "correct", "none"),
                ConditionSpec("permuted", text, "permuted", "none"),
                ConditionSpec("zero", text, "zero", "none")]
    if arch == TEXT_CONCAT:
        specs = [ConditionSpec("correct", text, "correct", "none"),
                 ConditionSpec("permuted", text, "permuted", "none"),
                 ConditionSpec("zero", text, "zero", "none")]
        if text != "none":
            specs.append(ConditionSpec("permuted_text", "permuted", "correct", "none"))
        return specs
    # ControlNet: joint conditions + the two single-branch ablations of §23.
    specs = [ConditionSpec("correct", text, "correct", "correct"),
             ConditionSpec("permuted", text, "permuted", "permuted"),
             ConditionSpec("zero", text, "zero", "zero"),
             ConditionSpec("semantic_permuted", text, "permuted", "correct"),
             ConditionSpec("semantic_zero", text, "zero", "correct"),
             ConditionSpec("lowlevel_permuted", text, "correct", "permuted"),
             ConditionSpec("lowlevel_zero", text, "correct", "zero")]
    if text != "none":
        specs.append(ConditionSpec("permuted_text", "permuted", "correct", "correct"))
    return specs


def _optional_specs(cfg) -> List[ConditionSpec]:
    """Extra named conditions a config may request explicitly.

    ``generic_*`` implement the "generic-text brain-only" control of §18.2: the
    prompt is a fixed, uninformative ``"Image of something"``. It is deliberately
    a DIFFERENT experiment from having no text at all (the sequence the UNet sees
    is not even the same length), so the two must never be aggregated into one
    result — request them by name when you want that comparison.
    """
    if not uses_text(cfg):
        return []
    structural = "correct" if uses_controlnet(cfg) else "none"
    return [
        ConditionSpec("generic_correct", "generic", "correct", structural),
        ConditionSpec("generic_permuted", "generic", "permuted",
                      "permuted" if structural != "none" else "none"),
        ConditionSpec("generic_zero", "generic", "zero",
                      "zero" if structural != "none" else "none"),
    ]


def _spec_from_entry(entry, cfg) -> ConditionSpec:
    """Accept both a plain name (``"correct"``) and an explicit mapping."""
    if isinstance(entry, str):
        table = {s.name: s for s in _default_specs(cfg) + _optional_specs(cfg)}
        if entry in table:
            return table[entry]
        # A bare brain state that the current architecture doesn't predefine
        # (e.g. 'noise' from an older config) — treat it as a semantic state.
        if entry in BRAIN_STATES:
            text = "correct" if uses_text(cfg) else "none"
            structural = entry if uses_controlnet(cfg) else "none"
            return ConditionSpec(entry, text, entry, structural)
        raise ValueError(
            f"Unknown generation condition {entry!r}. Known for architecture "
            f"{resolve_architecture(cfg)}: {sorted(table)}")
    data = dict(entry)
    name = str(data.pop("name"))
    return ConditionSpec(name, **{k: str(v) for k, v in data.items()})


def resolve_conditions(cfg, requested: Optional[Sequence] = None
                       ) -> List[ConditionSpec]:
    """Condition specs to generate: ``generation.conditions`` or the defaults."""
    entries = requested if requested is not None else cfg.get("generation.conditions",
                                                              None)
    if not entries:
        return _default_specs(cfg)
    return [_spec_from_entry(e, cfg) for e in entries]


def required_brain_conditions(specs: Sequence[ConditionSpec]) -> List[str]:
    """Decoder forwards actually needed (``correct``/``permuted``/``zero``).

    ``structural='zero'`` needs no forward: it is implemented by switching the
    ControlNet residuals off, not by decoding a null brain vector (§22.3).
    """
    needed = {s.semantic for s in specs}
    needed |= {s.structural for s in specs if s.structural in ("correct", "permuted")}
    order = {c: i for i, c in enumerate(BRAIN_STATES)}
    return sorted(needed, key=lambda c: order.get(c, 99))


# --- tensor fusion ----------------------------------------------------------
def concat_condition(text_embeds, neural_tokens):
    """``[B, L, D] (+) [B, K, D] -> [B, L+K, D]`` (Test 2)."""
    import torch
    if text_embeds is None:
        return neural_tokens
    if neural_tokens is None:
        return text_embeds
    if text_embeds.shape[0] != neural_tokens.shape[0]:
        raise ValueError(f"Batch mismatch: text {tuple(text_embeds.shape)} vs "
                         f"neural {tuple(neural_tokens.shape)}")
    if text_embeds.shape[-1] != neural_tokens.shape[-1]:
        raise ValueError(
            f"Cross-attention dim mismatch: text D={text_embeds.shape[-1]} vs "
            f"neural D={neural_tokens.shape[-1]}. The text encoder and the UNet "
            f"must come from the same Stable Diffusion checkpoint.")
    return torch.cat([text_embeds, neural_tokens.to(text_embeds.dtype)], dim=1)


def build_uncond_condition(text_embeds_empty, batch: int, num_tokens: int,
                           cross_dim: int, device, dtype):
    """CFG negative branch: ``[empty-text tokens ; zero-brain tokens]`` (§12).

    Must match the positive branch length exactly, otherwise diffusers cannot
    batch the two CFG halves. The neural half is literal zeros — the cheapest
    reproducible "no neural information" token block, and the same thing the
    legacy path already used as its unconditional embedding.
    """
    import torch
    zeros = torch.zeros(batch, num_tokens, cross_dim, device=device, dtype=dtype)
    if text_embeds_empty is None:
        return zeros
    empty = text_embeds_empty.to(device=device, dtype=dtype)
    if empty.shape[0] == 1 and batch > 1:
        empty = empty.expand(batch, -1, -1)
    return torch.cat([empty, zeros], dim=1)


# --- provenance -------------------------------------------------------------
def conditioning_metadata(cfg) -> dict:
    """The identity of a conditioning setup — stored in adapter checkpoints and
    in every generation's metadata so incompatible combinations are detectable
    (§14, §40)."""
    arch = resolve_architecture(cfg)
    cn = controlnet_settings(cfg)
    meta = {
        "conditioning_architecture": arch,
        "text_mode": text_mode(cfg) if arch != LEGACY else "none",
        "caption_field": resolve_caption_field(cfg) if uses_text(cfg) else None,
        "template": resolve_template(cfg) if uses_text(cfg) else None,
        "num_neural_tokens": num_neural_tokens(cfg),
        "adapter_normalize_input": adapter_normalize_input(cfg),
        "adapter_input_scale": adapter_input_scale(cfg),
        "fusion_type": fusion_type(cfg) if arch != LEGACY else None,
        "controlnet_enabled": cn["enabled"],
        "controlnet_model": cn["model"] if cn["enabled"] else None,
        "controlnet_condition_type": cn["condition_type"] if cn["enabled"] else None,
        "controlnet_training_condition_source": (cn["training_condition_source"]
                                                 if cn["enabled"] else None),
    }
    return meta


#: Keys that MUST match between an adapter checkpoint and the config using it.
#: ``adapter_input_scale`` is deliberately absent: it is an inference-time knob
#: designed to be swept without retraining (§15).
COMPAT_KEYS = ("conditioning_architecture", "text_mode", "caption_field",
               "template", "num_neural_tokens", "adapter_normalize_input",
               "fusion_type", "controlnet_enabled", "controlnet_model",
               "controlnet_condition_type")


def assert_adapter_compatible(state: dict, cfg, allow: Optional[bool] = None,
                              source: str = "") -> dict:
    """Fail loudly when a checkpoint's conditioning differs from the config (§14, Test 8).

    A TokenAdapter trained with a weak prompt is a different model from one
    trained with an oracle prompt, and one trained without ControlNet saw a
    different denoising problem than one trained with it. Loading either as the
    other silently would invalidate the experiment, so it is an error.

    ``generation.allow_incompatible_adapter: true`` downgrades it to a warning —
    only for the explicitly-allowed smoke test of §31, never for final results.
    """
    stored = state.get("conditioning") if isinstance(state, dict) else None
    wanted = conditioning_metadata(cfg)
    if allow is None:
        allow = bool(cfg.get("generation.allow_incompatible_adapter", False))
    keys = COMPAT_KEYS
    if not stored:
        # Pre-multimodal checkpoint: it can only be a legacy adapter. Its
        # `num_neural_tokens` is not recorded either, but a wrong K makes
        # load_state_dict fail on a shape mismatch, so nothing slips through.
        stored = {"conditioning_architecture": LEGACY, "text_mode": "none",
                  "caption_field": None, "template": None,
                  "num_neural_tokens": int(state.get("num_tokens",
                                                     wanted["num_neural_tokens"])),
                  "adapter_normalize_input": bool(state.get("normalize_input", False)),
                  "fusion_type": None, "controlnet_enabled": False,
                  "controlnet_model": None, "controlnet_condition_type": None}
        # `normalize_input` predates this check and has its own, more forgiving
        # handling in FrozenSDGenerator.load_adapter (warn + honour the
        # checkpoint). Escalating it to an error here would break every adapter
        # trained before Option B existed, so leave that key to load_adapter.
        keys = tuple(k for k in COMPAT_KEYS if k != "adapter_normalize_input")
    mismatches = {k: (stored.get(k), wanted.get(k)) for k in keys
                  if stored.get(k) != wanted.get(k)}
    if mismatches:
        detail = "; ".join(f"{k}: checkpoint={a!r} != config={b!r}"
                           for k, (a, b) in mismatches.items())
        msg = (f"Incompatible token adapter{f' ({source})' if source else ''}: "
               f"{detail}. Train a separate adapter for this configuration "
               f"(§14) or set generation.allow_incompatible_adapter=true for a "
               f"deliberate smoke test only (§31).")
        if not allow:
            raise ValueError(msg)
        logger.warning("%s [allowed by generation.allow_incompatible_adapter]", msg)
    return stored


def describe_conditions(specs: Sequence[ConditionSpec]) -> str:
    return ", ".join(f"{s.name}(text={s.text},sem={s.semantic},"
                     f"struct={s.structural})" for s in specs)


def condition_dir_name(spec: ConditionSpec, cfg) -> str:
    """Relative folder under ``outputs/<exp>/generated/`` for one condition.

    ``flat`` (default) keeps the historical ``generated/<condition>/`` layout, so
    Experiment 5 and every existing notebook keep working — architecture and text
    mode are already fixed per ``experiment.name``. ``nested`` produces the
    ``<architecture>/<text_mode>/<condition>/`` tree sketched in §39.
    """
    layout = str(cfg.get("generation.output_layout", "flat")).lower()
    if layout == "flat":
        return spec.name
    if layout != "nested":
        raise ValueError("generation.output_layout must be 'flat' or 'nested'")
    arch = resolve_architecture(cfg).replace("legacy_adapter", "legacy")
    mode = text_mode(cfg) if uses_text(cfg) else "brain_only"
    return f"{arch}/{mode}/{spec.name}"


def condition_layout(specs: Sequence[ConditionSpec], cfg) -> Dict[str, str]:
    return {s.name: condition_dir_name(s, cfg) for s in specs}


def zero_brain_vector(like: np.ndarray) -> np.ndarray:
    """Null brain input — kept as a named helper so the meaning of the ``zero``
    control stays defined in exactly one place (§21)."""
    return np.zeros_like(like)
