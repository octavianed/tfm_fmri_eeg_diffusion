#!/usr/bin/env python
"""Minimum tests of the multimodal extension (§41) — CPU, no Stable Diffusion.

Runs the eight required checks against a real config so a broken alignment or an
incompatible checkpoint is caught in seconds instead of after hours of GPU:

1. retro-compatibility — ``legacy_adapter`` needs no caption/text/ControlNet;
2. shapes — ``[B,L,D] (+) [B,K,D] = [B,L+K,D]``;
3. CFG — conditional and unconditional have the same length;
4. caption alignment — every ``feat_idx`` gets the caption of its own image;
5. caption permutation — deterministic, within-split, seed 42, same family;
6. ControlNet disabled — ``conditioning_scale=0`` removes the branch;
7. brain permutation — CLIP_pred and low_pred share ONE permutation;
8. checkpoint compatibility — weak cannot be loaded as oracle, nor text as controlnet.

    python scripts/15_validate_multimodal.py --config configs/fMRI/exp04_generation_text_weak.yaml

``--skip-data`` runs only the pure-logic tests (no dataset needed).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.data.captions import (build_prompt, caption_permutation,  # noqa: E402
                               resolve_caption_field, resolve_template,
                               text_mode)
from src.generation.conditioning import (ConditionSpec,  # noqa: E402
                                         assert_adapter_compatible,
                                         build_uncond_condition,
                                         concat_condition,
                                         conditioning_metadata,
                                         num_neural_tokens,
                                         resolve_architecture,
                                         resolve_conditions, uses_controlnet,
                                         uses_text)
from src.utils import ExtendOverrides, get_logger, load_config  # noqa: E402
from src.utils.permutation import condition_seed  # noqa: E402

log = get_logger("validate_multimodal")
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    log.info("%s %-46s %s", "PASS" if condition else "FAIL", name, detail)
    return bool(condition)


# --- Test 1 -----------------------------------------------------------------
def test_retrocompatibility(cfg_path, overrides):
    cfg = load_config(cfg_path, list(overrides or []) +
                      ["generation.conditioning_architecture=legacy_adapter",
                       "generation.text.enabled=false",
                       "generation.text.mode=none",
                       "generation.controlnet.enabled=false",
                       "generation.conditions=null"])
    meta = conditioning_metadata(cfg)
    specs = resolve_conditions(cfg)
    ok = (not uses_text(cfg) and not uses_controlnet(cfg)
          and meta["text_mode"] == "none" and meta["caption_field"] is None
          and [s.name for s in specs] == ["correct", "permuted", "zero"]
          and all(s.text == "none" and s.structural == "none" for s in specs))
    check("1. legacy_adapter needs no text/ControlNet", ok,
          f"conditions={[s.name for s in specs]}")


# --- Test 2 & 3 -------------------------------------------------------------
def test_shapes_and_cfg(cfg_path, overrides):
    import torch
    cfg = load_config(cfg_path, overrides)
    B, L, D = 3, 77, 768
    K = num_neural_tokens(cfg)
    text = torch.randn(B, L, D)
    brain = torch.randn(B, K, D)
    fused = concat_condition(text, brain)
    check("2. concat shapes [B,L,D]+[B,K,D]=[B,L+K,D]",
          tuple(fused.shape) == (B, L + K, D), f"-> {tuple(fused.shape)}")

    uncond = build_uncond_condition(torch.zeros(1, L, D), B, K, D,
                                    torch.device("cpu"), torch.float32)
    check("3. CFG branches have identical shape",
          tuple(uncond.shape) == tuple(fused.shape),
          f"{tuple(uncond.shape)} vs {tuple(fused.shape)}")
    # The legacy line has no text half: the condition IS the neural block.
    legacy = build_uncond_condition(None, B, K, D, torch.device("cpu"),
                                    torch.float32)
    check("3b. legacy CFG negative is [B,K,D]",
          tuple(legacy.shape) == (B, K, D), f"-> {tuple(legacy.shape)}")


# --- Test 4 -----------------------------------------------------------------
def test_caption_alignment(cfg, dm):
    from src.data.captions import load_caption_metadata
    field = resolve_caption_field(cfg)
    if field is None:
        check("4. caption alignment", True, "skipped (text.mode=none)")
        return
    ok, detail = True, []
    for subject in dm.subjects:
        for split in ("train", "val", "test"):
            frame = dm.subject_split_frame(subject, split)
            if len(frame) == 0:
                continue
            caps = load_caption_metadata(cfg, dm, subject, split, [field])
            same_ids = (caps["image_id"].tolist() ==
                        frame["image_id"].astype(str).tolist())
            same_idx = caps["feat_idx"].tolist() == list(range(len(frame)))
            ok = ok and same_ids and same_idx
            detail.append(f"{subject}/{split}:{len(caps)}")
    check("4. caption follows feat_idx of the same image", ok, " ".join(detail))

    # And the caption really belongs to that image, not to a neighbour.
    subject = dm.subjects[0]
    split = "test" if len(dm.subject_split_frame(dm.subjects[0], "test")) else "train"
    caps = load_caption_metadata(cfg, dm, subject, split, [field])
    frame = dm.subject_split_frame(subject, split)
    template = resolve_template(cfg)
    log.info("   e.g. feat_idx=0 image_id=%s -> %r", frame["image_id"].iloc[0],
             build_prompt(template, caps[field].iloc[0]))


# --- Test 5 -----------------------------------------------------------------
def test_caption_permutation(cfg, dm):
    if text_mode(cfg) == "none":
        check("5. caption permutation", True, "skipped (text.mode=none)")
        return
    from src.data.captions import build_split_prompts, permuted_prompts
    field = resolve_caption_field(cfg)
    subject = dm.subjects[0]
    split = "test" if len(dm.subject_split_frame(subject, "test")) else "train"
    correct = build_split_prompts(cfg, dm, subject, split, field)
    perm_a = permuted_prompts(cfg, dm, subject, split, field)
    perm_b = permuted_prompts(cfg, dm, subject, split, field)

    deterministic = perm_a == perm_b
    same_family = sorted(perm_a) == sorted(correct)
    seed = int(cfg.get("generation.text.permutation_seed", 42))
    idx = caption_permutation(len(correct), seed, subject, split, field)
    no_fixed = not np.any(idx == np.arange(len(idx))) if len(idx) > 1 else True
    # Distinct captions actually move (repeated captions may map to an equal
    # string even under a derangement — that is expected, not a fixed point).
    moved = sum(1 for a, b in zip(correct, perm_a) if a != b)
    check("5. permutation deterministic / same family / in split",
          deterministic and same_family and no_fixed,
          f"n={len(correct)} moved={moved} seed={seed} "
          f"source={cfg.get('generation.text.permutation_source', 'derived')}")


# --- Test 6 -----------------------------------------------------------------
def test_controlnet_disabled(cfg):
    from src.generation.generate_from_fmri import _condition_scale
    if not uses_controlnet(cfg):
        check("6. controlnet zero-state disables the branch", True,
              "skipped (controlnet disabled)")
        return
    zero = _condition_scale(cfg, ConditionSpec("z", "none", "correct", "zero"))
    on = _condition_scale(cfg, ConditionSpec("c", "none", "correct", "correct"))
    check("6. structural='zero' -> conditioning_scale=0",
          zero == 0.0 and on > 0.0, f"zero={zero} correct={on}")


# --- Test 7 -----------------------------------------------------------------
def test_brain_permutation_shared(cfg):
    """CLIP_pred and low_pred must come from ONE permuted forward (§22.2)."""
    from src.evaluation.ablation_eval import make_condition_input
    seed = int(cfg.get("generation.sample_seed", cfg.get("project.seed", 42)))
    brain = np.arange(40, dtype=np.float32).reshape(20, 2)
    rng_a = np.random.default_rng(condition_seed(seed, "permuted"))
    rng_b = np.random.default_rng(condition_seed(seed, "permuted"))
    a = make_condition_input(brain, "permuted", rng_a)
    b = make_condition_input(brain, "permuted", rng_b)
    reproducible = np.array_equal(a, b)
    # A single permuted input feeds both heads -> the permutation is shared by
    # construction; assert that the ablation really deranges (no fixed rows).
    fixed = int(np.sum(np.all(a == brain, axis=1)))

    specs = resolve_conditions(cfg)
    joint = [s for s in specs if s.name == "permuted"]
    shared = (not uses_controlnet(cfg)) or (
        joint and joint[0].semantic == "permuted" and joint[0].structural == "permuted")
    check("7. brain permutation reproducible and shared by both branches",
          reproducible and fixed == 0 and bool(shared),
          f"fixed_points={fixed} joint_spec={joint[0].to_dict() if joint else None}")


# --- Test 8 -----------------------------------------------------------------
def test_checkpoint_compatibility(cfg_path, overrides):
    base = list(overrides or [])
    weak = load_config(cfg_path, base + ["generation.conditioning_architecture="
                                         "text_adapter_concat",
                                         "generation.text.enabled=true",
                                         "generation.text.mode=weak",
                                         "generation.controlnet.enabled=false"])
    oracle = load_config(cfg_path, base + ["generation.conditioning_architecture="
                                           "text_adapter_concat",
                                           "generation.text.enabled=true",
                                           "generation.text.mode=oracle",
                                           "generation.controlnet.enabled=false"])
    cn = load_config(cfg_path, base + ["generation.conditioning_architecture="
                                       "text_adapter_concat_controlnet",
                                       "generation.text.enabled=true",
                                       "generation.text.mode=weak",
                                       "generation.controlnet.enabled=true",
                                       "generation.controlnet.model="
                                       "lllyasviel/sd-controlnet-canny"])
    legacy = load_config(cfg_path, base + ["generation.conditioning_architecture="
                                           "legacy_adapter",
                                           "generation.text.enabled=false",
                                           "generation.text.mode=none",
                                           "generation.controlnet.enabled=false"])

    weak_ckpt = {"conditioning": conditioning_metadata(weak)}

    def rejects(state, target_cfg):
        try:
            assert_adapter_compatible(state, target_cfg, allow=False)
            return False
        except ValueError:
            return True

    same_ok = True
    try:
        assert_adapter_compatible(weak_ckpt, weak, allow=False)
    except ValueError as exc:
        same_ok, _ = False, exc

    # A pre-multimodal checkpoint (no 'conditioning' key) is a legacy adapter.
    old = {"normalize_input": False, "num_tokens": num_neural_tokens(legacy)}
    legacy_ok = True
    try:
        assert_adapter_compatible(old, legacy, allow=False)
    except ValueError:
        legacy_ok = False

    ok = (same_ok and legacy_ok
          and (resolve_caption_field(weak) == resolve_caption_field(oracle)
               or rejects(weak_ckpt, oracle))
          and rejects(weak_ckpt, cn) and rejects(weak_ckpt, legacy))
    note = "" if resolve_caption_field(weak) != resolve_caption_field(oracle) \
        else " [EEG: weak==oracle by design, so they are interchangeable]"
    check("8. incompatible adapter checkpoints are rejected", ok, note.strip())

    # Escape hatch of §31 must still exist (smoke test only).
    allowed = True
    try:
        assert_adapter_compatible(weak_ckpt, cn, allow=True)
    except ValueError:
        allowed = False
    check("8b. allow_incompatible_adapter downgrades to a warning", allowed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None)
    ap.add_argument("--skip-data", action="store_true",
                    help="skip the tests that need the dataset on disk")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    log.info("Config: %s | architecture=%s | text=%s | controlnet=%s",
             args.config, resolve_architecture(cfg), text_mode(cfg),
             uses_controlnet(cfg))

    test_retrocompatibility(args.config, args.set)
    test_shapes_and_cfg(args.config, args.set)
    test_controlnet_disabled(cfg)
    test_brain_permutation_shared(cfg)
    test_checkpoint_compatibility(args.config, args.set)

    if not args.skip_data:
        from src.data import build_datamodule
        dm = build_datamodule(cfg).prepare()
        test_caption_alignment(cfg, dm)
        test_caption_permutation(cfg, dm)
    else:
        log.info("Skipping the dataset-dependent tests (--skip-data)")

    failed = [n for n, ok, _ in RESULTS if not ok]
    log.info("%d/%d checks passed", len(RESULTS) - len(failed), len(RESULTS))
    if failed:
        raise SystemExit("FAILED: " + ", ".join(failed))


if __name__ == "__main__":
    main()
