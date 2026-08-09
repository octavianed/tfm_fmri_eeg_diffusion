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

## 6. Outputs

```
outputs/<experiment>/
  config.yaml            checkpoints/{last,best,epoch_XXXX}.pt
  logs/{train_log.csv, resume_history.jsonl}
  metrics/*.{json,csv}   figures/*.png
  embeddings/*.npy       lowlevel/*.npy
  generated/{real,correct,permuted,zero}/*.png   grids/*.png
  metadata/generation_params.json   report/*.md
```

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
src/generation frozen-SD pipeline, adapter training, generate-from-brain, grids
src/utils   config, seeding/RNG, device/AMP, logging, paths, checkpointing
scripts/    00–08 CLI entry points (modality via --config)   notebooks/{fMRI,EEG}/ 00–06 + 30_multimodal
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
