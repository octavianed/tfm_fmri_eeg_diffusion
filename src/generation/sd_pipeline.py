"""Frozen Stable Diffusion generation from fMRI-predicted representations.

Mechanism (spec §10.3): Stable Diffusion 1.5 is used **frozen**. The predicted
CLIP image embedding is turned into pseudo prompt-token embeddings by a small
trainable :class:`~src.models.adapters.TokenAdapter` and passed through the
public ``prompt_embeds`` API of ``StableDiffusionPipeline`` (Option B). The
UNet, VAE and text encoder are never trained. Optionally, the predicted
low-level VAE-PCA vector is inverted to a latent and used as an img2img
initialization (Option C), injecting fMRI-derived structure.

The token adapter is the only trainable module here; it is trained with a
frozen-UNet diffusion (epsilon) loss over precomputed VAE latents and CLIP
embeddings, so training needs neither the VAE nor CLIP in memory and fits a
16 GB GPU (spec §10.2, §4).
"""
from __future__ import annotations

import datetime as _dt
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from ..models.adapters import TokenAdapter
from ..utils import (CheckpointManager, CSVLogger, JsonlLogger, autocast,
                     get_device, get_logger, load_checkpoint, make_grad_scaler,
                     save_checkpoint, save_config)
from ..utils.paths import vae_latent_path
from ..features.load_features import inverse_pca_to_latent, load_pca_bundle
from ..features.load_features import load_split_features

logger = get_logger("sd")


# --- pipeline loading ------------------------------------------------------
def load_sd_pipeline(cfg, device, img2img: bool = False, unet=None):
    from diffusers import (StableDiffusionImg2ImgPipeline,
                           StableDiffusionPipeline)
    model = str(cfg.get("generation.sd_model",
                        "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    cls = StableDiffusionImg2ImgPipeline if img2img else StableDiffusionPipeline
    # Generation is conditioned via `prompt_embeds` (from the token adapter), so
    # the SD text encoder is only needed for the empty *negative* prompt. By
    # default we skip loading it: this sidesteps transformers/diffusers version
    # incompatibilities (a strict text-encoder loader can reject the SD-1.5
    # checkpoint) and matches this project's empty-prompt design — the
    # unconditional branch then falls back to a zero embedding. Set
    # generation.load_text_encoder=true to use the real empty-string embedding
    # when your transformers/diffusers versions are known-compatible.
    load_kwargs = dict(torch_dtype=dtype, safety_checker=None,
                       requires_safety_checker=False)
    if not bool(cfg.get("generation.load_text_encoder", False)):
        load_kwargs.update(text_encoder=None, tokenizer=None)
    # Reuse an already-loaded UNet (e.g. the one being used to train the adapter)
    # instead of loading a second ~1.7 GB copy — diffusers skips loading any
    # component passed explicitly. Used by the in-loop generation eval.
    if unet is not None:
        load_kwargs["unet"] = unet
    pipe = cls.from_pretrained(model, **load_kwargs)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    for module in (pipe.unet, pipe.vae, getattr(pipe, "text_encoder", None)):
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad_(False)
    if bool(cfg.get("generation.enable_attention_slicing", True)):
        pipe.enable_attention_slicing()
    if bool(cfg.get("generation.enable_vae_slicing", True)):
        pipe.enable_vae_slicing()
    if bool(cfg.get("generation.cpu_offload", False)):
        pipe.enable_model_cpu_offload()
    return pipe


class FrozenSDGenerator:
    """Wraps a frozen SD pipeline + a (optional) trained token adapter."""

    def __init__(self, cfg, device=None, mode: Optional[str] = None, unet=None):
        self.cfg = cfg
        self.device = device or get_device(cfg.get("runtime.device", "auto"))
        self.mode = mode or str(cfg.get("generation.mode", "adapter"))
        self.use_img2img = self.mode in ("lowlevel_img2img", "adapter_lowlevel")
        self.pipe = load_sd_pipeline(cfg, self.device, img2img=self.use_img2img,
                                     unet=unet)
        self.cross_dim = int(self.pipe.unet.config.cross_attention_dim)
        self.num_tokens = int(cfg.get("generation.num_tokens", 77))
        te = getattr(self.pipe, "text_encoder", None)
        self.embeds_dtype = te.dtype if te is not None else self.pipe.unet.dtype
        self.adapter = None
        self._uncond_cache = {}
        # Optional calibration of the *predicted* CLIP embedding norm before the
        # adapter (see `rescale_to_norm` in _prompt_embeds). Set by the inference
        # entry points when generation.rescale_clip_pred is enabled; left None
        # for the in-loop adapter eval, which already feeds real embeddings.
        self.rescale_to_norm: Optional[float] = None

    # -- adapter -----------------------------------------------------------
    def load_adapter(self, clip_dim: int, checkpoint_path=None):
        """Build the adapter and load its weights.

        ``normalize_input`` (Option B) is taken from the **checkpoint** when it
        records it, so an adapter trained on normalized embeddings can never be
        run un-normalized (or vice versa) by accident — that mismatch would
        silently produce garbage conditioning.
        """
        cfg_norm = bool(self.cfg.get("generation.adapter_normalize_input", False))
        normalize_input = cfg_norm
        state = None
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            state = load_checkpoint(checkpoint_path, map_location=self.device)
            if "normalize_input" in state:
                normalize_input = bool(state["normalize_input"])
                if normalize_input != cfg_norm:
                    logger.warning(
                        "generation.adapter_normalize_input=%s but the checkpoint "
                        "was trained with normalize_input=%s; honouring the "
                        "checkpoint (they must match).", cfg_norm, normalize_input)
            else:
                logger.info("Adapter checkpoint predates Option B (no "
                            "'normalize_input' key); assuming %s.", cfg_norm)

        self.adapter = TokenAdapter(
            clip_dim, cross_dim=self.cross_dim, num_tokens=self.num_tokens,
            hidden_dim=int(self.cfg.get("generation.adapter_hidden", 1024)),
            normalize_input=normalize_input,
            input_scale=float(self.cfg.get("generation.adapter_input_scale", 1.0)),
        ).to(self.device)
        if state is not None:
            self.adapter.load_state_dict(state["adapter_state_dict"])
            logger.info("Loaded token adapter from %s", checkpoint_path)
        if normalize_input:
            logger.info("Adapter is scale-invariant (normalize_input=True, "
                        "input_scale=%.3f)", self.adapter.input_scale)
            if self.rescale_to_norm is not None:
                logger.warning(
                    "generation.rescale_clip_pred is set but the adapter "
                    "normalizes its input, so the rescaling has NO effect. Use "
                    "generation.adapter_input_scale to tune conditioning strength.")
        self.adapter.eval()
        return self.adapter

    def _empty_text_embeds(self, num: int):
        if num in self._uncond_cache:
            return self._uncond_cache[num]
        tokenizer = getattr(self.pipe, "tokenizer", None)
        text_encoder = getattr(self.pipe, "text_encoder", None)
        if tokenizer is None or text_encoder is None:
            # No text encoder loaded -> zero unconditional embedding (empty-prompt
            # design; classifier-free guidance still steers toward the adapter tokens).
            emb = torch.zeros(num, self.num_tokens, self.cross_dim,
                              device=self.device, dtype=self.embeds_dtype)
        else:
            ids = tokenizer(
                [""] * num, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                emb = text_encoder(ids)[0]
        self._uncond_cache[num] = emb
        return emb

    def _prompt_embeds(self, clip_embeds):
        """Build [B, num_tokens, cross_dim] conditioning from CLIP embeddings."""
        clip_embeds = torch.as_tensor(clip_embeds, dtype=torch.float32,
                                      device=self.device)
        if self.rescale_to_norm is not None:
            # Neither CLIP loss term constrains the norm, so a decoder's
            # predictions can sit on a shell of a different radius than the one
            # the adapter was trained on (real CLIP embeddings have an almost
            # constant norm). Project onto that radius: keeps the direction —
            # which carries all the semantics — and removes a scale mismatch
            # that otherwise acts like an uncontrolled guidance strength and
            # differs from run to run.
            norm = clip_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            clip_embeds = clip_embeds / norm * float(self.rescale_to_norm)
        if self.adapter is None:
            # no adapter -> unconditional (Option C relies on the img2img init)
            return self._empty_text_embeds(clip_embeds.shape[0]).to(self.embeds_dtype)
        with torch.no_grad():
            tokens = self.adapter(clip_embeds)
        return tokens.to(self.embeds_dtype)

    # -- latent utils (Option C) ------------------------------------------
    def decode_latents_to_pil(self, latents_scaled):
        from ..data.image_transforms import tensor_to_pil
        vae = self.pipe.vae
        z = torch.as_tensor(latents_scaled, dtype=vae.dtype, device=self.device)
        with torch.no_grad():
            imgs = vae.decode(z / vae.config.scaling_factor).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1).float().cpu()
        return [tensor_to_pil(img) for img in imgs]

    # -- generation --------------------------------------------------------
    def generate(self, clip_embeds, seed: int = 123, guidance_scale: float = 3.0,
                 num_inference_steps: int = 50, init_images=None,
                 strength: float = 0.8) -> List:
        num = int(np.asarray(clip_embeds).shape[0])
        prompt_embeds = self._prompt_embeds(clip_embeds)
        negative = self._empty_text_embeds(num).to(prompt_embeds.dtype)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        common = dict(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative,
                      num_inference_steps=int(num_inference_steps),
                      guidance_scale=float(guidance_scale), generator=generator)
        with autocast(self.device, enabled=self.device.type == "cuda"):
            if self.use_img2img and init_images is not None:
                images = self.pipe(image=init_images, strength=float(strength),
                                   **common).images
            else:
                images = self.pipe(**common).images
        return images


# --- token-adapter training (frozen-UNet diffusion loss) -------------------
def _load_adapter_training_data(cfg, datamodule, split="train"):
    clip_parts, latent_parts = [], []
    for subj in datamodule.subjects:
        clip = load_split_features(cfg, subj, split, "clip")
        lat_path = vae_latent_path(cfg, subj, split)
        if clip is None or not lat_path.exists():
            raise FileNotFoundError(
                f"Adapter training needs CLIP + VAE latents for {subj}/{split}. "
                f"Run precompute_clip and precompute_vae_latents.")
        latents = np.load(lat_path)
        n = min(len(clip), len(latents))
        clip_parts.append(clip[:n])
        latent_parts.append(latents[:n])
    return (np.concatenate(clip_parts, 0).astype(np.float32),
            np.concatenate(latent_parts, 0).astype(np.float32))


def _build_adapter_scheduler(optimizer, cfg, steps_per_epoch: int, epochs: int):
    """Cosine LR schedule with warmup for the token adapter.

    A flat LR held constant for tens of thousands of steps was found to leave
    the adapter's diffusion loss essentially flat (it never learns to use the
    conditioning at all — the frozen UNet can already reach a similar loss
    unconditionally). Warming up then decaying, like the main decoder trainer
    already does (see trainer_utils.build_scheduler), gives optimization a much
    better chance of finding a useful conditioning direction.
    """
    kind = str(cfg.get("generation.adapter_scheduler", "cosine")).lower()
    if kind in ("none", "null", ""):
        return None
    total = max(1, steps_per_epoch * epochs)
    warmup = int(float(cfg.get("generation.adapter_warmup_ratio", 0.05)) * total)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_token_adapter(cfg, datamodule, resume=None) -> dict:
    """Train the TokenAdapter with a frozen SD-1.5 UNet epsilon-prediction loss."""
    from diffusers import DDPMScheduler, UNet2DConditionModel

    device = get_device(cfg.get("runtime.device", "auto"))
    from ..utils import get_experiment_paths
    paths = get_experiment_paths(cfg, ensure=True)
    save_config(cfg, paths.root / "config.yaml")
    ckpt_dir = paths.checkpoints
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = ckpt_dir / "adapter_last.pt", ckpt_dir / "adapter_best.pt"

    model_name = str(cfg.get("generation.sd_model",
                             "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    unet = UNet2DConditionModel.from_pretrained(
        model_name, subfolder="unet", torch_dtype=dtype).to(device)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)
    if bool(cfg.get("generation.grad_checkpointing", True)):
        unet.enable_gradient_checkpointing()
    noise_sched = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    pred_type = noise_sched.config.prediction_type

    clip_np, lat_np = _load_adapter_training_data(cfg, datamodule, "train")
    clip_dim = clip_np.shape[1]
    n = clip_np.shape[0]
    latent_shape = [4, int(cfg.get("features.vae_image_size", 512)) // 8,
                    int(cfg.get("features.vae_image_size", 512)) // 8]
    cross_dim = int(unet.config.cross_attention_dim)
    num_tokens = int(cfg.get("generation.num_tokens", 77))

    # Option B: train on L2-normalized CLIP embeddings so the adapter is
    # scale-invariant by construction. The flag is stored in every checkpoint so
    # inference can never mismatch it. Training always runs at input_scale=1.0.
    normalize_input = bool(cfg.get("generation.adapter_normalize_input", False))
    adapter = TokenAdapter(clip_dim, cross_dim=cross_dim, num_tokens=num_tokens,
                           hidden_dim=int(cfg.get("generation.adapter_hidden",
                                                  1024)),
                           normalize_input=normalize_input).to(device)
    if normalize_input:
        logger.info("Training a SCALE-INVARIANT adapter (normalize_input=True): "
                    "the predicted-norm drift of the decoder can no longer act as "
                    "an uncontrolled conditioning strength.")
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(cfg.get("generation.adapter_lr", 1e-4)),
        weight_decay=float(cfg.get("generation.adapter_weight_decay", 0.0)))
    amp = device.type == "cuda"
    scaler = make_grad_scaler(enabled=amp)
    epochs = int(cfg.get("generation.adapter_epochs", 20))
    batch_size = int(cfg.get("generation.adapter_batch_size", 4))
    steps_per_epoch = math.ceil(n / batch_size)
    scheduler = _build_adapter_scheduler(optimizer, cfg, steps_per_epoch, epochs)
    # Feature 1 — average the loss over several random timesteps per sample
    # (1 = off / original behaviour).
    n_timesteps = max(1, int(cfg.get("generation.adapter_timesteps_per_sample", 1)))
    # Periodic epoch_XXXX.pt snapshots (last/best alone hide the trajectory in
    # between — training loss on a single random timestep per sample is a noisy
    # proxy for generation quality, so "best" and "last" are not enough to know
    # which epoch actually generates well; keep several to compare visually).
    periodic_ckpt = CheckpointManager(
        ckpt_dir, save_last=False, save_best=False,
        save_every_n_epochs=int(cfg.get("generation.adapter_save_every_n_epochs", 5)),
        keep_last_n=int(cfg.get("generation.adapter_keep_last_n", 5)),
        keep_first_n=int(cfg.get("generation.adapter_keep_first_n", 3)))

    start_epoch, global_step, best_loss = 0, 0, float("inf")
    best_val_sim = float("-inf")
    if resume is None:
        resume = cfg.get("generation.adapter_resume", None)
    resume_path = None
    if resume in ("auto", True) and last_path.exists():
        resume_path = last_path
    elif isinstance(resume, str) and resume not in ("auto", "none", "null") \
            and Path(resume).exists():
        resume_path = Path(resume)
    if resume_path is not None:
        state = load_checkpoint(resume_path, map_location=device)
        prev_norm = state.get("normalize_input")
        if prev_norm is not None and bool(prev_norm) != normalize_input:
            raise ValueError(
                f"Refusing to resume: the checkpoint was trained with "
                f"normalize_input={bool(prev_norm)} but the config says "
                f"{normalize_input}. Mixing both regimes would corrupt training — "
                f"use a different experiment.name for the other setting.")
        adapter.load_state_dict(state["adapter_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        if state.get("scaler_state_dict"):
            scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch = int(state.get("epoch", -1)) + 1
        global_step = int(state.get("global_step", 0))
        best_loss = float(state.get("best_loss", float("inf")))
        best_val_sim = float(state.get("best_val_sim", float("-inf")))
        logger.info("Resumed adapter training from %s (epoch %d)",
                    resume_path, start_epoch)
        JsonlLogger(paths.resume_history).append({
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "phase": "token_adapter", "checkpoint_path": str(resume_path),
            "resumed_epoch": start_epoch, "resumed_global_step": global_step,
            "best_loss": best_loss})

    csv_logger = CSVLogger(paths.logs / "adapter_train_log.csv",
                           ["epoch", "global_step", "loss", "best_loss", "lr",
                            "val_clip_sim", "best_val_sim", "is_best", "seconds"])

    # Feature 2 — periodically generate a few held-out images with the current
    # adapter and score them by CLIP similarity to the real image (a direct
    # proxy of generation quality, unlike the training loss). Reuses the training
    # UNet (no second copy loaded). None if disabled.
    eval_ctx = _build_adapter_eval_context(cfg, datamodule, unet, device)
    eval_every = max(1, int(cfg.get("generation.adapter_eval_every_n_epochs", 5)))
    select_by = str(cfg.get("generation.adapter_select_by", "auto")).lower()
    if select_by == "auto":
        select_by = "clip_sim" if eval_ctx is not None else "loss"
    if select_by == "clip_sim" and eval_ctx is None:
        logger.warning("adapter_select_by=clip_sim but eval is disabled; "
                       "selecting best.pt by training loss instead.")
        select_by = "loss"
    logger.info("Adapter training: %d timestep(s)/sample | select best by '%s'"
                " | eval %s", n_timesteps, select_by,
                "on" if eval_ctx is not None else "off")

    clip_t = torch.from_numpy(clip_np)
    lat_t = torch.from_numpy(lat_np).view(-1, *latent_shape)
    rng = np.random.default_rng(int(cfg.get("project.seed", 42)))

    adapter.train()
    for epoch in range(start_epoch, epochs):
        order = rng.permutation(n)
        running, count = 0.0, 0
        t0 = time.time()
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            cond_in = clip_t[idx].to(device)
            z0 = lat_t[idx].to(device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, enabled=amp):
                # Adapter forward computed ONCE per batch; the (noise, timestep)
                # draw is what varies. Averaging the loss over several random
                # timesteps per sample (n_timesteps > 1) reduces the variance of
                # the per-step loss (each image is otherwise scored at a single
                # random t, whose difficulty swings a lot). Costs n_timesteps
                # extra frozen-UNet forwards per batch (more VRAM/time).
                cond = adapter(cond_in).to(dtype)
                step_losses = []
                for _ in range(n_timesteps):
                    noise = torch.randn_like(z0)
                    t = torch.randint(0, noise_sched.config.num_train_timesteps,
                                      (z0.shape[0],), device=device).long()
                    zt = noise_sched.add_noise(z0, noise, t)
                    target = noise if pred_type == "epsilon" else \
                        noise_sched.get_velocity(z0, noise, t)
                    pred = unet(zt, t, encoder_hidden_states=cond).sample
                    step_losses.append(
                        torch.nn.functional.mse_loss(pred.float(), target.float()))
                loss = torch.stack(step_losses).mean() if n_timesteps > 1 \
                    else step_losses[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            running += float(loss.item()) * z0.shape[0]
            count += z0.shape[0]
        epoch_loss = running / max(1, count)
        loss_improved = epoch_loss < best_loss
        if loss_improved:
            best_loss = epoch_loss  # always tracked (for logging), even when
            #                         selecting by clip_sim

        # Feature 2 — generation-quality eval (every N epochs and on the last one)
        val_sim = None
        run_eval = eval_ctx is not None and (
            epoch % eval_every == 0 or epoch == epochs - 1)
        if run_eval:
            val_sim = _eval_adapter_quality(eval_ctx, adapter, device)
        sim_improved = val_sim is not None and val_sim > best_val_sim
        if sim_improved:
            best_val_sim = val_sim

        is_best = sim_improved if select_by == "clip_sim" else loss_improved
        elapsed = time.time() - t0
        logger.info("[adapter] epoch %d/%d loss %.5f%s (%.0fs)%s", epoch, epochs,
                    epoch_loss,
                    f" clip_sim {val_sim:.4f}" if val_sim is not None else "",
                    elapsed, " *" if is_best else "")
        state = {"adapter_state_dict": adapter.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                 "scaler_state_dict": scaler.state_dict() if amp else None,
                 "epoch": epoch, "global_step": global_step,
                 "best_loss": best_loss, "best_val_sim": best_val_sim,
                 "select_by": select_by, "clip_dim": clip_dim,
                 # Option B flag: inference MUST use the same setting, so it
                 # travels with the weights (see FrozenSDGenerator.load_adapter).
                 "normalize_input": normalize_input,
                 "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)}
        save_checkpoint(state, last_path)
        if is_best:
            save_checkpoint(state, best_path)
        periodic_ckpt.save(state, epoch)
        csv_logger.log({"epoch": epoch, "global_step": global_step,
                        "loss": epoch_loss, "best_loss": best_loss,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "val_clip_sim": "" if val_sim is None else round(val_sim, 5),
                        "best_val_sim": "" if best_val_sim == float("-inf")
                        else round(best_val_sim, 5),
                        "is_best": int(is_best), "seconds": round(elapsed, 1)})
    _save_adapter_loss_figure(paths.logs / "adapter_train_log.csv", paths.figures)
    return {"adapter_checkpoint": str(best_path if best_path.exists() else last_path),
            "best_loss": best_loss,
            "best_val_sim": None if best_val_sim == float("-inf") else best_val_sim,
            "select_by": select_by, "clip_dim": clip_dim,
            "normalize_input": normalize_input}


def _build_adapter_eval_context(cfg, datamodule, unet, device):
    """Prepare in-loop generation eval, or return None if disabled.

    Uses a small held-out set of images and their *precomputed* real CLIP
    embeddings (the same kind of input the adapter is trained on) — so it
    measures the adapter's core job (CLIP-embedding -> image) directly, with no
    dependence on the fMRI decoder. Reuses the training UNet for the pipeline
    (no second UNet copy) and forces mode='adapter' (pure semantic path).
    """
    if not bool(cfg.get("generation.adapter_eval_enabled", False)):
        return None
    split = str(cfg.get("generation.adapter_eval_split", "val"))
    k = int(cfg.get("generation.adapter_eval_num_samples", 6))
    seed = int(cfg.get("generation.sample_seed", cfg.get("project.seed", 42)))
    clip_embs, reals = _load_adapter_eval_data(cfg, datamodule, split, k, seed)
    if clip_embs is None:
        logger.warning("adapter_eval_enabled but no CLIP features/images for "
                       "split '%s'; disabling in-loop eval.", split)
        return None
    from ..features.clip_model import load_clip
    gen = FrozenSDGenerator(cfg, device=device, mode="adapter", unet=unet)
    gs = cfg.get("generation.adapter_eval_guidance_scale")
    return {"gen": gen, "clip_bundle": load_clip(cfg, device),
            "clip_embs": clip_embs, "reals": reals, "seed": seed,
            "steps": int(cfg.get("generation.adapter_eval_steps", 25)),
            "gs": float(gs if gs is not None
                        else cfg.get("generation.guidance_scale", 3.0))}


def _load_adapter_eval_data(cfg, datamodule, split, k, seed):
    from ..data.image_transforms import load_image
    from ..features.load_features import load_split_features
    frame = datamodule.get_frame(split)
    if len(frame) == 0:
        return None, None
    rng = np.random.default_rng(seed + 777)
    idx = np.sort(rng.choice(len(frame), size=min(k, len(frame)), replace=False))
    rows = frame.iloc[idx]
    size = int(cfg.get("features.vae_image_size", 512))
    clip_cache, embs, reals = {}, [], []
    for r in rows.itertuples():
        subj = r.subject_id
        if subj not in clip_cache:
            clip_cache[subj] = load_split_features(cfg, subj, split, "clip")
        arr = clip_cache[subj]
        if arr is None:
            return None, None
        embs.append(arr[int(r.feat_idx)])
        reals.append(load_image(str(r.image_path)).resize((size, size)))
    return np.stack(embs).astype(np.float32), reals


def _eval_adapter_quality(ctx, adapter, device) -> float:
    """Generate from the held-out CLIP embeddings with the current adapter and
    return the mean CLIP similarity between generated and real images."""
    from ..evaluation.generation_metrics import compute_generation_metrics
    ctx["gen"].adapter = adapter
    was_training = adapter.training
    adapter.eval()
    try:
        images = ctx["gen"].generate(
            ctx["clip_embs"], seed=ctx["seed"], guidance_scale=ctx["gs"],
            num_inference_steps=ctx["steps"])
        res = compute_generation_metrics(ctx["reals"], images, ctx["clip_bundle"],
                                         device, ks=(1,))
        return float(res["metrics"]["mean_clip_similarity"])
    finally:
        if was_training:
            adapter.train()


def _save_adapter_loss_figure(csv_path, figures_dir):
    """Guarda outputs/<exp>/figures/adapter_loss_curve.png desde el CSV del adapter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:  # pragma: no cover
        return None
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if len(df) == 0 or "epoch" not in df.columns or "loss" not in df.columns:
        return None
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["loss"], marker="o", ms=3, label="loss (por época)")
    if "best_loss" in df.columns:
        ax.plot(df["epoch"], df["best_loss"], "--", label="best_loss")
    ax.set_xlabel("época")
    ax.set_ylabel("MSE de difusión (predicción de ε)")
    ax.set_title("TokenAdapter — curva de pérdida")
    ax.legend()
    fig.tight_layout()
    out = figures_dir / "adapter_loss_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved adapter loss curve: %s", out)
    return str(out)
