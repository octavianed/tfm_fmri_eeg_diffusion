"""Generate images from brain-predicted representations, condition by condition.

The SAME sample set and generation seeds are used for every condition so the
comparison is paired and only the experimental factor changes (spec §10.4,
§11.2; multimodal §24). Brain permutation is a full-split Sattolo derangement
(via the ablation utilities); ``zero`` feeds a null brain vector to the decoder.

A condition is a triple (see :class:`~src.generation.conditioning.ConditionSpec`):

* **text** — ``none`` / ``correct`` / ``permuted`` / ``generic``;
* **semantic** — which decoder forward supplies ``CLIP_pred`` for the TokenAdapter;
* **structural** — which decoder forward supplies ``low_pred`` for ControlNet
  (``zero`` = ControlNet residuals switched off, §22.3).

Both brain branches of a permuted condition read the SAME permuted forward, so
``CLIP_pred[perm[i]]`` and ``low_pred[perm[i]]`` always use one and the same
permutation (§22.2, Test 7).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..data import build_datamodule, load_image
from ..evaluation.ablation_eval import make_condition_input
from ..evaluation.eval_data import load_subject_matrices
from ..features.load_features import (clip_norm_reference,
                                      inverse_pca_to_latent, load_pca_bundle)
from ..models import build_model_from_checkpoint
from ..utils import (get_device, get_experiment_paths, get_logger, save_config,
                     save_json)
from ..utils.permutation import condition_seed
from .conditioning import (ConditionSpec, condition_layout, conditioning_metadata,
                           controlnet_settings, describe_conditions,
                           required_brain_conditions, resolve_conditions,
                           uses_controlnet, uses_text)
from .sd_pipeline import FrozenSDGenerator

logger = get_logger("generate")


def load_decoder(cfg, datamodule, checkpoint_path, device):
    model, state = build_model_from_checkpoint(
        cfg, checkpoint_path, device, voxel_counts=datamodule.voxel_counts)
    return model, {"clip_dim": state.get("clip_dim"),
                   "low_dim": state.get("low_dim"),
                   "use_lowlevel": bool(state.get("use_lowlevel", False))}


def select_samples(datamodule, split: str, n: Optional[int], seed: int) -> List[dict]:
    """Sample ``n`` stimuli of a split (``n=None`` -> the whole split).

    Using the whole split is what gives Experiment 5's paired tests real
    statistical power; it is affordable now that generation is chunked
    (``generation.batch_size``).
    """
    frame = datamodule.get_frame(split)
    # One row per image: EEG training frames carry one row per repetition, and
    # generating the same image several times would waste GPU and bias metrics.
    frame = frame.drop_duplicates(subset=["subject_id", "feat_idx"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if n is None or n >= len(frame):
        idx = np.arange(len(frame))
    else:
        idx = np.sort(rng.choice(len(frame), size=n, replace=False))
    rows = frame.iloc[idx]
    return [{"subject": r.subject_id, "feat_idx": int(r.feat_idx),
             "image_path": str(r.image_path), "image_id": str(r.image_id)}
            for r in rows.itertuples()]


def _forward_full(model, fmri_np, subject, device, batch_size: int = 256):
    use_subject = getattr(model, "adapters", None) is not None
    clip, low = [], []
    with torch.no_grad():
        for start in range(0, len(fmri_np), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(fmri_np[start:start + batch_size])
            ).float().to(device)
            out = model(batch, subject=subject if use_subject else None)
            clip.append(out["clip"].float().cpu().numpy())
            if out.get("low") is not None:
                low.append(out["low"].float().cpu().numpy())
    return (np.concatenate(clip, 0),
            np.concatenate(low, 0) if low else None)


def predict_condition_embeddings(model, cfg, datamodule, selection, conditions,
                                 split, device, seed):
    """``clip_pred`` and ``low_pred`` per brain condition, for the selected samples.

    One decoder forward per (subject, brain condition); both outputs of a given
    forward stay paired, which is exactly what §22.2 requires of the permuted
    condition.
    """
    subjects = sorted({s["subject"] for s in selection})
    noise_std = float(cfg.get("evaluation.noise_std", 1.0))
    cache = {}
    for subj in subjects:
        mats = load_subject_matrices(cfg, datamodule, subj, split, want=("fmri",))
        for cond in conditions:
            # Reproducible across processes (see src.utils.permutation), and the
            # SAME draw for every branch of a condition: clip_pred and low_pred
            # come out of one forward, so they share the permutation (§22.2).
            rng = np.random.default_rng(condition_seed(seed, cond))
            fin = make_condition_input(mats.fmri, cond, rng, noise_std)
            cache[(subj, cond)] = _forward_full(model, fin, subj, device)

    clip_by = {c: [] for c in conditions}
    low_by = {c: [] for c in conditions}
    for s in selection:
        for cond in conditions:
            clip_full, low_full = cache[(s["subject"], cond)]
            clip_by[cond].append(clip_full[s["feat_idx"]])
            low_by[cond].append(None if low_full is None else low_full[s["feat_idx"]])
    clip_out = {c: np.stack(v).astype(np.float32) for c, v in clip_by.items()}
    low_out = {}
    for c, v in low_by.items():
        low_out[c] = np.stack(v).astype(np.float32) if all(
            x is not None for x in v) else None
    return clip_out, low_out


def lowlevel_init_images(cfg, generator, selection, low_vectors):
    bundles = {}
    latents = []
    for i, s in enumerate(selection):
        subj = s["subject"]
        if subj not in bundles:
            bundles[subj] = load_pca_bundle(cfg, subj)
        lat = inverse_pca_to_latent(bundles[subj], low_vectors[i:i + 1])
        latents.append(lat[0])
    return generator.decode_latents_to_pil(np.stack(latents))


def save_condition_images(images, out_dir, image_ids):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for img, iid in zip(images, image_ids):
        p = out_dir / f"{iid}.png"
        img.save(p)
        paths.append(str(p))
    return paths


def resolve_adapter_checkpoint(cfg, paths, explicit=None) -> Optional[Path]:
    """CLI argument -> ``generation.adapter_checkpoint`` -> the experiment's own.

    ``cfg.get(key, default)`` returns ``None`` when the key exists *with* a null
    value (which several shipped configs use to mean "resolve it for me"), so a
    plain ``cfg.get(..., default)`` silently skipped the fallback and generation
    ran with NO adapter — producing unconditional images that look like a failed
    decoder. Resolve it in one place instead.
    """
    for candidate in (explicit, cfg.get("generation.adapter_checkpoint", None),
                      paths.checkpoints / "adapter_best.pt"):
        if candidate:
            return Path(candidate)
    return None


def resolve_clip_rescale(cfg, datamodule, generator=None) -> Optional[float]:
    """Resolve ``generation.rescale_clip_pred`` into a target norm (or ``None``).

    ``none`` (default) keeps the historical behaviour. ``train_median`` projects
    every predicted CLIP embedding onto the sphere whose radius is the median
    norm of the **real** train-split embeddings — the regime the TokenAdapter was
    trained in. A float value is used verbatim.
    """
    mode = cfg.get("generation.rescale_clip_pred", "none")
    target = None
    if mode is None or str(mode).lower() in ("none", "off", "false", ""):
        target = None
    elif isinstance(mode, (int, float)) and not isinstance(mode, bool):
        target = float(mode)
    elif str(mode).lower() in ("train_median", "train_mean"):
        stat = "mean" if str(mode).lower().endswith("mean") else "median"
        target = clip_norm_reference(cfg, datamodule.subjects, "train", stat)
        if target is None:
            logger.warning("rescale_clip_pred=%s but no train CLIP features "
                           "found; skipping the rescaling.", mode)
    else:
        raise ValueError(f"Unknown generation.rescale_clip_pred: {mode!r}")

    if generator is not None:
        generator.rescale_to_norm = target
    if target is not None:
        logger.info("Calibrating predicted CLIP norm to %.3f (%s) before the adapter",
                    target, mode)
    return target


# --- per-condition inputs ---------------------------------------------------
def build_text_inputs(cfg, selection, split) -> Optional[dict]:
    """Text embeddings (and their prompt strings) per text state.

    Returns ``None`` for the brain-only architectures. ``generic`` reuses the
    single control prompt of §18.2 broadcast over the batch — deliberately a
    *different* experiment from "no text at all".
    """
    if not uses_text(cfg):
        return None
    from ..features.text_embeddings import load_text_cache
    cache = load_text_cache(cfg)
    out = {}
    for state, permuted in (("correct", False), ("permuted", True)):
        embeds, prompts = [], []
        for s in selection:
            embeds.append(cache.rows(s["subject"], split, [s["feat_idx"]],
                                     permuted=permuted)[0])
            prompts.append(cache.prompt(s["subject"], split, s["feat_idx"],
                                        permuted=permuted))
        out[state] = {"embeds": np.stack(embeds), "prompts": prompts}
    generic = cache.special("generic")
    out["generic"] = {"embeds": np.repeat(generic, len(selection), axis=0),
                      "prompts": [cache.prompts[cache.meta["special_rows"]["generic"]]]
                      * len(selection)}
    out["none"] = None
    logger.info("Text conditioning ready | e.g. correct=%r permuted=%r",
                out["correct"]["prompts"][0], out["permuted"]["prompts"][0])
    return out


def build_control_inputs(cfg, generator, selection, low_by, specs, device):
    """ControlNet condition images per structural state.

    ``zero`` gets black images *and* ``controlnet_conditioning_scale=0`` — the
    scale is what mathematically removes the branch (§22.3); the black image just
    makes the intent visible in the saved artefacts.
    """
    if not uses_controlnet(cfg):
        return None
    from PIL import Image

    from .controlnet_condition import control_images_from_lowlevel
    size = int(cfg.get("features.vae_image_size", 512))
    states = {s.structural for s in specs}
    out: Dict[str, Optional[list]] = {}
    bundles: dict = {}
    for state in states:
        if state in ("none", "zero"):
            out[state] = [Image.new("RGB", (size, size), (0, 0, 0))
                          for _ in selection]
            continue
        low = low_by.get(state)
        if low is None:
            raise ValueError(
                f"The ControlNet architecture needs low-level predictions for "
                f"state {state!r}, but the decoder produced none. Use an Exp3 "
                f"(multitask) checkpoint and model.use_lowlevel=true.")
        out[state] = control_images_from_lowlevel(
            cfg, generator.decode_latents_to_pil, selection, low, device, bundles)
    return out


def _condition_scale(cfg, spec: ConditionSpec) -> Optional[float]:
    if not uses_controlnet(cfg):
        return None
    if spec.structural in ("none", "zero"):
        return 0.0
    return float(controlnet_settings(cfg)["conditioning_scale"])


# --- main entry point -------------------------------------------------------
def generate_images(cfg, decoder_checkpoint, adapter_checkpoint=None,
                    conditions: Optional[Sequence] = None,
                    device=None, split: Optional[str] = None) -> dict:
    device = device or get_device(cfg.get("runtime.device", "auto"))
    split = split or str(cfg.get("generation.split", "test"))
    paths = get_experiment_paths(cfg, ensure=True)
    save_config(cfg, paths.root / "config.yaml")
    dm = build_datamodule(cfg).prepare()

    specs = resolve_conditions(cfg, conditions)
    brain_conditions = required_brain_conditions(specs)
    logger.info("Conditions: %s", describe_conditions(specs))

    model, meta = load_decoder(cfg, dm, decoder_checkpoint, device)
    # null/0 => the whole split (see select_samples)
    n = cfg.get("generation.num_samples", 16)
    n = None if not n else int(n)
    sample_seed = int(cfg.get("generation.sample_seed",
                              cfg.get("project.seed", 42)))
    selection = select_samples(dm, split, n, sample_seed)
    image_ids = [s["image_id"] for s in selection]
    logger.info("Generating %d sample(s) of split '%s' per condition "
                "(batch_size=%s)", len(selection), split,
                cfg.get("generation.batch_size", None) or len(selection))

    clip_by, low_by = predict_condition_embeddings(
        model, cfg, dm, selection, brain_conditions, split, device, sample_seed)

    generator = FrozenSDGenerator(cfg, device=device)
    clip_rescale = resolve_clip_rescale(cfg, dm, generator)
    adapter_used, adapter_loaded = None, False
    if generator.mode in ("adapter", "adapter_lowlevel"):
        adapter_ck = resolve_adapter_checkpoint(cfg, paths, adapter_checkpoint)
        adapter_used = adapter_ck
        if adapter_ck and Path(adapter_ck).exists():
            generator.load_adapter(int(meta["clip_dim"]), adapter_ck)
            adapter_loaded = True
            logger.info("Token adapter loaded from %s", adapter_ck)
        else:
            logger.warning("No token adapter at %s; falling back to "
                           "unconditional prompt (rely on img2img init).",
                           adapter_ck)

    text_inputs = build_text_inputs(cfg, selection, split)
    control_inputs = build_control_inputs(cfg, generator, selection, low_by,
                                          specs, device)

    gen_seed = int(cfg.get("generation.seed", 123))
    gs = float(cfg.get("generation.guidance_scale", 3.0))
    steps = int(cfg.get("generation.num_inference_steps", 50))
    strength = float(cfg.get("generation.strength", 0.8))

    outputs = {"image_ids": image_ids, "subjects": [s["subject"] for s in selection]}
    # real reference images resized to the SD resolution
    size = int(cfg.get("features.vae_image_size", 512))
    reals = [load_image(s["image_path"]).resize((size, size)) for s in selection]
    outputs["real"] = reals
    save_condition_images(reals, paths.generated / "real", image_ids)

    layout = condition_layout(specs, cfg)
    if control_inputs is not None:
        for state, imgs in control_inputs.items():
            if state not in ("none", "zero"):
                save_condition_images(imgs, paths.generated / f"_control_{state}",
                                      image_ids)

    per_condition_meta = []
    for spec in specs:
        init_images = None
        if generator.use_img2img and low_by.get(spec.semantic) is not None:
            # Option C: the img2img seed follows the STRUCTURAL branch when there
            # is one, else the semantic forward (legacy behaviour).
            src = spec.structural if spec.structural in low_by else spec.semantic
            init_images = lowlevel_init_images(cfg, generator, selection, low_by[src])
        text = None if text_inputs is None or text_inputs.get(spec.text) is None \
            else text_inputs[spec.text]["embeds"]
        controls = None if control_inputs is None else control_inputs[spec.structural]
        images = generator.generate(
            clip_by[spec.semantic], seed=gen_seed, guidance_scale=gs,
            num_inference_steps=steps, init_images=init_images, strength=strength,
            text_embeds=text, control_images=controls,
            controlnet_scale=_condition_scale(cfg, spec))
        out_dir = paths.generated / layout[spec.name]
        save_condition_images(images, out_dir, image_ids)
        outputs[spec.name] = images
        per_condition_meta.append({
            **spec.to_dict(), "directory": layout[spec.name],
            "controlnet_scale": _condition_scale(cfg, spec),
            "prompts": (text_inputs[spec.text]["prompts"]
                        if text_inputs and text_inputs.get(spec.text) else None)})
        logger.info("Generated %d images for condition '%s' -> %s", len(images),
                    spec.name, out_dir)

    save_json(_generation_params(
        cfg, generator, specs, layout, split, selection, image_ids, gen_seed,
        sample_seed, gs, steps, strength, clip_rescale, decoder_checkpoint,
        adapter_used, adapter_loaded, per_condition_meta),
        paths.metadata / "generation_params.json")
    save_json(_per_sample_metadata(cfg, selection, split, specs, per_condition_meta,
                                   gen_seed, sample_seed, adapter_used),
              paths.metadata / "generation_samples.json")
    return outputs


def _generation_params(cfg, generator, specs, layout, split, selection, image_ids,
                       gen_seed, sample_seed, gs, steps, strength, clip_rescale,
                       decoder_checkpoint, adapter_used, adapter_loaded,
                       per_condition_meta) -> dict:
    cn = controlnet_settings(cfg)
    return {
        "mode": generator.mode,
        # Kept as plain names so Experiment 5 and older notebooks keep working.
        "conditions": [s.name for s in specs],
        "condition_specs": [s.to_dict() for s in specs],
        "condition_dirs": layout,
        "per_condition": per_condition_meta,
        "split": split, "num_samples": len(selection),
        "generation_seed": gen_seed, "sample_seed": sample_seed,
        "guidance_scale": gs, "num_inference_steps": steps, "strength": strength,
        "scheduler": type(generator.pipe.scheduler).__name__,
        "rescale_clip_pred": cfg.get("generation.rescale_clip_pred", "none"),
        "clip_pred_target_norm": clip_rescale,
        "sd_model": str(cfg.get("generation.sd_model", "")),
        "prompt_mode": str(cfg.get("generation.prompt_mode", "empty")),
        "decoder_checkpoint": str(decoder_checkpoint),
        # The RESOLVED adapter (CLI arg -> config -> experiment default), plus
        # whether it was actually loaded: recording the raw argument would log an
        # empty string whenever the path came from the config (spec §10.4).
        "adapter_checkpoint": str(adapter_used or ""),
        "adapter_loaded": bool(adapter_loaded),
        "brain_permutation_seed": sample_seed,
        "text_permutation_seed": int(cfg.get("generation.text.permutation_seed", 42)),
        "controlnet": cn,
        **conditioning_metadata(cfg),
        "image_ids": image_ids,
    }


def _per_sample_metadata(cfg, selection, split, specs, per_condition_meta,
                         gen_seed, sample_seed, adapter_used) -> list:
    """One record per (sample, condition) — enough to rebuild any image (§40)."""
    from .conditioning import adapter_input_scale, num_neural_tokens
    cn = controlnet_settings(cfg)
    base = conditioning_metadata(cfg)
    prompts_by_cond = {m["name"]: m.get("prompts") for m in per_condition_meta}
    scales_by_cond = {m["name"]: m.get("controlnet_scale") for m in per_condition_meta}
    rows = []
    for i, sample in enumerate(selection):
        for spec in specs:
            prompts = prompts_by_cond.get(spec.name)
            rows.append({
                "dataset": str(cfg.get("dataset.name", "")),
                "modality": str(cfg.get("dataset.modality", "fmri")),
                "subject": sample["subject"], "split": split,
                "image_id": sample["image_id"], "feat_idx": sample["feat_idx"],
                "architecture": base["conditioning_architecture"],
                "caption_mode": base["text_mode"],
                "caption_field": base["caption_field"],
                "text_condition": spec.text,
                "resolved_prompt": prompts[i] if prompts else None,
                "text_permutation_seed": int(
                    cfg.get("generation.text.permutation_seed", 42)),
                "brain_condition": spec.semantic,
                "lowlevel_condition": spec.structural,
                "brain_permutation_seed": sample_seed,
                "adapter_checkpoint": str(adapter_used or ""),
                "adapter_input_scale": adapter_input_scale(cfg),
                "num_neural_tokens": num_neural_tokens(cfg),
                "controlnet_enabled": cn["enabled"],
                "controlnet_model": cn["model"] or None,
                "controlnet_condition_type": cn["condition_type"] if cn["enabled"] else None,
                "controlnet_scale": scales_by_cond.get(spec.name),
                "controlnet_condition_state": spec.structural,
                "diffusion_seed": gen_seed,
                "num_inference_steps": int(cfg.get("generation.num_inference_steps", 50)),
                "guidance_scale": float(cfg.get("generation.guidance_scale", 3.0)),
                "sd_model": str(cfg.get("generation.sd_model", "")),
            })
    return rows
