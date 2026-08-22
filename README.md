# Brain → Image Visual Decoding & Reconstruction (TFM) — fMRI + EEG

Decode and reconstruct the images a subject saw from their brain response. Two
**modalities** share one framework (same CLIP/VAE-PCA targets, heads, losses,
ablations and experimental flow — only the dataset and encoder differ):

- **fMRI** — **NSD Algonauts 2023** (spatial response per image → residual-MLP encoder).
- **EEG** — **THINGS-EEG2** (temporal multichannel signal → temporal-conv encoder). See
  [§10](#10-eeg-line-things-eeg2).

The project is deliberately split into a *decoding* stage (can the brain signal
predict a visual representation of the seen image?) and a *generation* stage (use
those predictions to guide a **frozen** image generator). CLIP, the VAE and Stable
Diffusion are never trained.

The guiding principle is falsifiable (spec §2):

```
fMRI correct  →  predicted embedding better than chance
fMRI permuted →  ~ chance          (negative control)
fMRI zero     →  ~ chance          (negative control)

claim of "using brain signal" is valid ONLY IF:  correct  >>  permuted ≈ zero
```

If the correct condition does **not** clearly beat the controls, the code and
report say so explicitly and do **not** attribute the generated images to real
brain information.

---

## 1. Hardware & install

Targets **32 GB RAM + 16 GB VRAM**. CLIP/VAE/SD stay frozen and features are
precomputed to disk, so training the decoder never loads Stable Diffusion.

```bash
python -m venv .venv && source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -e .            # installs deps from pyproject; makes `import src...` work
# Install the CUDA torch build matching your driver, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2. Dataset

Algonauts 2023 data (based on NSD) — <https://algonautsproject.com/2023/braindata.html>
(devkit: <https://github.com/gifale95/algonauts_2023>). Two on-disk layouts are
**auto-detected**:

**A) Official layout with released test fMRI** (recommended — this is the full
post-challenge release):

```
<root>/train_data/subjNN/training_split/training_fmri/{lh,rh}_training_fmri.npy
<root>/train_data/subjNN/training_split/training_images/train-XXXX_nsd-YYYYY.png
<root>/train_data/subjNN/roi_masks/...
<root>/test_data/subjNN/test_split/test_fmri/{lh,rh}_test_fmri.npy
<root>/test_data/subjNN/test_split/test_images/test-XXXX_nsd-YYYYY.png
<root>/test_data/subjNN/test_split/noise_ceiling/...
```

**B) Flat layout** (`<root>/subjNN/training_split/...`). During the 2023
challenge the test fMRI was withheld (only test *images* were public), so a real
held-out test needs the released test fMRI. If it is absent, the code carves an
internal test set from the labeled training data instead.

```yaml
dataset:
  root_dir: C:/Users/xxdia/Documents/Datasets/NSD_Algonauts_2023
  subject_selection: subj01          # or [subj01, subj02] or all
  test_split: official               # official (uses test_data) | internal (carve from train)
```

- `test_split: official` → **train/val** come from `train_data` (val carved with
  `val_ratio`); **test** is the official `test_data`. This is the standard
  Algonauts held-out test.
- `test_split: internal` → train/val/test are all carved from the training data
  (automatic fallback when no `test_data` is present).

Normalization is fitted **only on train** and reused for val/test (saved to
disk, no leakage). Recommended subject progression:
`subj01` → `subj01+subj02` → selected list → `all`.

## 3. Run the experiments (mandatory order — spec §6)

Every command takes `--set key.path=value` overrides, e.g.
`--set dataset.subject_selection=all --set training.batch_size=32`.

```bash
# 0) Prepare: resolve subjects, build splits, fit train-only normalization
python scripts/00_prepare_dataset.py    --config configs/fMRI/exp01_fmri_to_clip.yaml

# 1) Precompute frozen-CLIP image embeddings (the targets)
python scripts/01_precompute_clip.py    --config configs/fMRI/exp01_fmri_to_clip.yaml

# 2) EXPERIMENT 1 — train fMRI → CLIP (no Stable Diffusion loaded)
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml

# 3) EXPERIMENT 2 — retrieval ablation correct / permuted / zero (+ mean & ridge baselines)
python scripts/03_eval_retrieval_ablation.py --config configs/fMRI/exp02_retrieval_ablation.yaml

# 4) Precompute VAE latents + fit PCA (train only)
python scripts/04_precompute_vae_pca.py --config configs/fMRI/exp03_lowlevel_multitask.yaml

# 5) EXPERIMENT 3 — multitask decoder (CLIP + low-level PCA head)
python scripts/05_train_multitask.py    --config configs/fMRI/exp03_lowlevel_multitask.yaml

# 6) EXPERIMENT 4 — generate with frozen Stable Diffusion (trains only the token adapter)
python scripts/06_generate_images.py    --config configs/fMRI/exp04_generation.yaml --train-adapter

# 7) EXPERIMENT 5 — final generative comparison correct / permuted / zero
python scripts/07_eval_generation_ablation.py --config configs/fMRI/exp05_generation_ablation.yaml
```

Interactive, visual versions of each step live in `notebooks/` and import the
**same** `src/` functions (no duplicated logic).

## 4. Stop & resume training (spec §17)

Trainings checkpoint the *full* state (model, optimizer, scheduler, GradScaler,
epoch, `global_step`, best metric, early-stopping counter, RNG state, config,
library versions). Resume with either:

```bash
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml --resume
#                                                                             ^ = last.pt
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml --resume path/to/last.pt
```

or in the config:

```yaml
checkpointing:
  resume: auto      # looks for checkpoints/last.pt in the experiment output dir
```

`last.pt` and `best.pt` are never pruned; only periodic `epoch_XXXX.pt` beyond
`keep_last_n` are removed. Each resume appends to `logs/resume_history.jsonl`
and continues `logs/train_log.csv` without losing prior rows.

## 5. How generation is conditioned (spec §10.3)

Stable Diffusion 1.5 is **frozen**. The default mechanism is **Option B**: the
predicted CLIP image embedding is mapped to pseudo prompt-token embeddings by a
small **TokenAdapter** and fed through the public `prompt_embeds` API; the UNet
never trains. The adapter (the only trainable module) is trained with a
frozen-UNet epsilon-prediction loss over *precomputed* latents + CLIP
embeddings, so it needs neither CLIP nor the VAE in memory.

`generation.mode` also supports **Option C** (`lowlevel_img2img`): invert the
predicted low-level PCA vector to a VAE latent and use it as an img2img
initialization; and `adapter_lowlevel` (B semantics + C structure). Weak/empty
prompts are used so the fMRI-derived signal is not masked by text.

Since the multimodal extension (§12) the condition can additionally include text
tokens and ControlNet residuals; `generation.conditioning_architecture` defaults
to `legacy_adapter`, i.e. exactly the behaviour described above.

## 6. Outputs

```
outputs/<experiment>/
  config.yaml            checkpoints/{last,best,epoch_XXXX}.pt
  logs/{train_log.csv, resume_history.jsonl}
  metrics/*.{json,csv}   figures/*.png
  embeddings/*.npy       lowlevel/*.npy
  generated/{real,correct,permuted,zero}/*.png   grids/*.png
  metadata/{generation_params.json, generation_samples.json}   report/*.md
```

With the multimodal extension, `generated/` also holds the extra conditions
(`permuted_text`, `semantic_*`, `lowlevel_*`), `metrics/generation_deltas.csv`
holds the paired deltas, and `metadata/generation_samples.json` holds one record
per (sample, condition) with everything needed to rebuild that exact image
(resolved prompt, brain/text condition, seeds, ControlNet scale, ...).

Aggregated metrics carry `metric_name, condition, subject_id, split, value,
seed, checkpoint` (spec §16.3). `metrics/conclusion.json` (Exp 2) and
`report/exp05_summary.md` (Exp 5) state the correct-vs-controls verdict.

## 7. Interpreting the controls

- **fMRI correct** — the real response for each image.
- **fMRI permuted** — each sample gets *another* sample's response (a Sattolo
  derangement, so never its own). Expected ≈ chance.
- **fMRI zero** — a null vector. Expected ≈ chance.

Success = **correct clearly above permuted and zero** on retrieval (Top-k, mean
rank, cosine) and on generation CLIP similarity. Good-looking images alone are
**not** evidence of brain decoding (spec §20).

## 8. Project structure

```
configs/fMRI  base.yaml + one YAML per experiment (via `_base_`)   configs/EEG  same, EEG line
src/data    fMRI (datamodule) + EEG (eeg_datamodule/eeg_things_dataset/eeg_normalization) + factory
src/features precompute CLIP embeddings, VAE latents, fit PCA (train only) — shared by both modalities
src/models  fMRIEncoder, EEGEncoderTemporalConv, CLIP/LowLevel heads, adapters, multitask decoder
src/losses  cosine, InfoNCE contrastive, multitask combination
src/training train/val loops + full checkpoint & resume
src/evaluation retrieval, embedding, ablation (correct/permuted/zero), baselines, generation metrics
src/generation frozen-SD pipeline, conditioning architectures, ControlNet conditions, adapter training, generate-from-brain, grids
src/utils   config, seeding/RNG, device/AMP, logging, paths, checkpointing
scripts/    00–16 CLI entry points (modality via --config)   notebooks/{fMRI,EEG}/ 00–06 + 30_multimodal
```

## 9. Reproducibility

Seeds are fixed for Python/NumPy/PyTorch; the resolved config, library versions
and (optionally) RNG state are saved in every checkpoint. Splits and
normalization are cached and rebuilt only if their parameters change.

## 10. EEG line (THINGS-EEG2)

The EEG line reuses the whole framework through a modality switch — set
`dataset.modality: eeg` (already set in `configs/EEG/base.yaml`) and
`build_datamodule`/`build_model` pick the EEG datamodule and the temporal-conv
encoder. Everything downstream (CLIP/VAE-PCA targets, heads, losses, retrieval,
correct/permuted/zero ablation, frozen-SD generation, checkpointing) is shared.

**Dataset.** `THINGS-EEG2` under
`C:/Users/xxdia/Documents/Datasets/THINGS-EEG2`: `image_set/` (stimulus images +
`image_metadata.npy`) and `preprocessed_data/<sub>/preprocessed_eeg_{training,test}.npy`
(17-channel montage) or `<sub>__63_channels/` (63 channels; the trailing `stim`
channel is dropped). Each file is a dict with `preprocessed_eeg_data` of shape
`[images, repetitions, channels, times]` — train `(16540, 4, C, 100)`, test
`(200, 80, C, 100)` — sampled at 100 Hz over −200→790 ms. Read with NumPy (no MNE).

**EEG-specific config.** `dataset.channels` (17|63), `dataset.time_window_ms`
(`null` or `[a, b]`), `dataset.subject_selection` (`sub-01` | list | `all`), and
`dataset.trial_aggregation` (`{train: none, val: mean, test: mean}`). Training uses
**per-trial** samples (augmentation across repetitions); **evaluation/ablation run
at the image level** (mean over repetitions → unique candidates, high SNR — the
standard THINGS-EEG2 protocol). `training.num_workers: 0` (EEG arrays live in RAM).

**Run it (same scripts, EEG configs).**

```bash
python scripts/00_prepare_dataset.py         --config configs/EEG/exp01_eeg_to_clip.yaml
python scripts/01_precompute_clip.py          --config configs/EEG/exp01_eeg_to_clip.yaml
python scripts/02_train_fmri_to_clip.py       --config configs/EEG/exp01_eeg_to_clip.yaml   # trains EEG (config-driven)
python scripts/03_eval_retrieval_ablation.py  --config configs/EEG/exp02_retrieval_ablation.yaml
python scripts/04_precompute_vae_pca.py       --config configs/EEG/exp03_lowlevel_multitask.yaml
python scripts/05_train_multitask.py          --config configs/EEG/exp03_lowlevel_multitask.yaml
python scripts/06_generate_images.py          --config configs/EEG/exp04_generation.yaml --train-adapter
python scripts/07_eval_generation_ablation.py --config configs/EEG/exp05_generation_ablation.yaml
```

Visual companions live in `notebooks/EEG/00..06`; `notebooks/30_multimodal_comparison.ipynb`
compares the fMRI and EEG lines against each modality's own chance level.

**Chance (image-level).** test (200 candidates): Top-1 ≈ 0.5%, Top-5 ≈ 2.5%,
Top-10 ≈ 5%. As with fMRI, EEG is only credited with visual decoding if
`correct ≫ permuted ≈ zero`; and decoding (Exp 2) is reported separately from
generation (Exp 5), since EEG may decode in retrieval yet not clearly steer
generation.

## 11. Own EEG preprocessing from raw (+ ablations)

Besides the official derivatives, the EEG line can start from the **raw 63-channel
recordings** (`<root>/raw-eeg/`) and apply a parameterized pipeline of our own,
plus a battery of preprocessing ablations. Full details in
[docs/05](docs/05_preprocesamiento_eeg_raw_y_ablaciones.md).

Reference configuration (`configs/EEG/preproc/baseline.yaml`): 63 channels,
0.1–100 Hz on the continuous signal, epoch −200…1000 ms, baseline −200…0 ms,
250 Hz, half-open crop `[0, 1000)` ms (**exactly 250 samples**), no
ICA/ASR/CAR/notch, **MVNN fitted on training only and applied to each repetition
before averaging**, average-4 in training, all 80 test repetitions kept →
`63 × 250` per image.

Step 09 needs **the project venv's interpreter** (MNE lives there, not in the
system Python). The baseline pins `preprocessing.filter.backend: mne`, so a wrong
interpreter fails in ~2 s with an actionable message instead of silently
degrading to a scipy IIR filter — which would make that variant incomparable
with the others. The variant cache is also guarded by a config hash: editing a
preprocessing parameter and re-running without `--force` is refused.

```bash
.tfm_fmri_diffusion_3_11/Scripts/python.exe scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/baseline.yaml
python scripts/10_validate_eeg_preproc.py  --config configs/EEG/preproc/baseline.yaml
python scripts/02_train_fmri_to_clip.py    --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
python scripts/11_eval_test_repetitions.py --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
```

A variant writes `data/processed/eeg_preproc/<variant>/<subject>/` using the
**same file contract as the official derivatives**, so the datamodule, encoder
and Exp1–Exp5 consume it unchanged. Ten one-factor ablations ship as minimal
overrides (`ablate_mvnn`, `channels_17`, `temporal_100_600`, `temporal_200_400`,
`sampling_100hz`, `frequency_0_5_40`, `train_independent_trials`,
`reference_car`, `baseline_minus100`, `baseline_none`), and the test-repetition
curve (`R = 1…80`) is an evaluation protocol that needs **no retraining**.

**What to re-run when the preprocessing changes:** rebuild the variant (`09`) and
the metadata (`00`), then retrain Exp1/Exp2/Exp3 and regenerate images + Exp5.
**CLIP and VAE/PCA features are NOT recomputed** — they depend only on the
stimulus images, and with an unchanged split they stay aligned by `feat_idx`, so
they are shared across every variant. The **token adapter is not retrained**
either: it maps CLIP embeddings to VAE latents and never sees the EEG.

## 12. Multimodal extension: text prompts + ControlNet

The generation stage can optionally be conditioned on a caption of the seen
image and on a pretrained ControlNet. Full details in
[docs/08](docs/08_ampliacion_multimodal_texto_y_controlnet.md).

Three architectures, chosen with `generation.conditioning_architecture`:

| value | condition fed to the frozen UNet |
|---|---|
| `legacy_adapter` *(default)* | `[K neural tokens]` — the previous behaviour, no caption at all |
| `text_adapter_concat` | `[77 text tokens ; K neural tokens]` via cross-attention |
| `text_adapter_concat_controlnet` | the above **plus** frozen ControlNet residuals from `low_pred` |

ControlNet does **not** replace the concatenation: text and neural tokens keep
arriving through cross-attention while ControlNet adds spatial residuals built
from `low_pred → inverse PCA → VAE decode → Canny`. The **TokenAdapter is still
the only trainable module**; the ControlNet is pretrained and frozen.

```bash
# Architecture 1 — weak text (the textual baseline)
python scripts/13_precompute_text_embeddings.py --config configs/fMRI/exp04_generation_text_weak.yaml
python scripts/06_generate_images.py            --config configs/fMRI/exp04_generation_text_weak.yaml --train-adapter
python scripts/07_eval_generation_ablation.py   --config configs/fMRI/exp05_generation_text_weak_ablation.yaml

# Architecture 2 — + ControlNet
python scripts/14_precompute_controlnet_conditions.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml
python scripts/06_generate_images.py                  --config configs/fMRI/exp04_generation_controlnet_weak.yaml --train-adapter
python scripts/16_sweep_controlnet_scale.py           --config configs/fMRI/exp04_generation_controlnet_weak.yaml

# Minimum tests (seconds, CPU, no Stable Diffusion)
python scripts/15_validate_multimodal.py --config configs/fMRI/exp04_generation_text_weak.yaml
```

**Text modes.** `generation.text.mode: none | weak | oracle` resolves to
`prompt_categories` / `primary_caption` for fMRI and to `primary_caption` for
EEG — so in EEG **oracle == weak** (THINGS-EEG2 has no more detailed caption);
do not report it as a more informative condition. `weak` is the textual
*baseline*; `oracle` is a performance ceiling.

**What the extension measures.** Beyond `correct ≫ permuted ≈ zero`, Experiment 5
now writes `metrics/generation_deltas.csv` with `delta_brain` (text fixed, brain
permuted), `delta_text` (brain fixed, caption permuted) and — for the ControlNet
architecture — `delta_semantic` and `delta_lowlevel`, which separate the
contribution of the CLIP branch from that of the VAE-PCA branch. A detailed
caption is *expected* to shrink `delta_brain`: that is informative about the
text, not a failure of the decoder.

**Permuted captions** are a Sattolo derangement with seed 42 drawn **within each
`(subject, split)` and within the same caption family**, using the very same
helper as the brain permutation. The `permuted_caption_seed42` column shipped in
the CSVs is *not* used by default: it was shuffled over the whole dataset split
(mixing this project's train and val) and has no counterpart for the weak fMRI
family.

**Checkpoints are not interchangeable.** Every adapter checkpoint stores its
conditioning identity (architecture, text mode, caption field, template, `K`,
ControlNet model/type); loading it under a different one fails with an explicit
error unless `generation.allow_incompatible_adapter: true` (smoke tests only).

**EEG preprocessing ablations stay compatible.** Captions, text embeddings,
ControlNet conditions and the adapter itself depend only on the images and the
split — never on the EEG — so a single cache and a single adapter serve every
variant of §11; only Exp4/Exp5 are re-run. Ready-made configs:
`configs/EEG/exp04_raw_<variant>_generation_text_weak.yaml`.
