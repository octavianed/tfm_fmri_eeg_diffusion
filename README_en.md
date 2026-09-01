# Brain → Image: Visual Decoding and Reconstruction from fMRI and EEG

Master's thesis (TFM) code. The system decodes the image a subject was looking at
from their brain response, and uses that decoded representation to condition a
**frozen** latent diffusion model in order to reconstruct an approximation of the
stimulus.

Two neuroimaging **modalities** share one single framework — the same visual
targets, prediction heads, losses, negative controls, metrics and experimental
flow. Only the dataset and the brain encoder change:

| Modality | Dataset | Signal | Encoder |
|---|---|---|---|
| **fMRI** | NSD / Algonauts 2023 | Spatial response per image (39 548 vertices) | Residual MLP |
| **EEG** | THINGS-EEG2 | Multichannel time series (`channels × time`) | Temporal CNN with attention pooling |

The project deliberately separates two questions that are often merged:

1. **Decoding** — does the brain signal predict a visual representation of the
   seen image? Measurable *without generating anything*.
2. **Generation** — can those predictions steer a frozen image generator?

CLIP, the VAE, Stable Diffusion, the text encoder and ControlNet are **never
trained**. The only trainable modules are the brain encoder with its prediction
heads, and a small token adapter.

### Guiding principle

Everything is evaluated against a falsifiable criterion:

```
correct signal   →  clearly better than chance
permuted signal  →  ≈ chance          (negative control)
null signal      →  ≈ chance          (negative control)
noise            →  ≈ chance          (negative control)

"the brain signal is being used" is claimed ONLY IF:  correct ≫ permuted ≈ null
```

If the correct condition does not clearly beat the controls, the code says so
explicitly (`metrics/conclusion.json`, `report/*.md`) and the result is **not**
attributed to real brain information — visually convincing images are not, on
their own, evidence of decoding.

---

## Table of contents

1. [Hardware and installation](#1-hardware-and-installation)
2. [Datasets](#2-datasets)
3. [Full pipeline from scratch](#3-full-pipeline-from-scratch)
4. [Experiments and final configurations](#4-experiments-and-final-configurations)
5. [Worked example: reproducing configuration Cf](#5-worked-example-reproducing-configuration-cf)
6. [EEG: own preprocessing from raw + ablations](#6-eeg-own-preprocessing-from-raw--ablations)
7. [Conditioning architectures: text and spatial control](#7-conditioning-architectures-text-and-spatial-control)
8. [Stop and resume training](#8-stop-and-resume-training)
9. [Outputs](#9-outputs)
10. [Interpreting the controls](#10-interpreting-the-controls)
11. [Published results and checkpoints](#11-published-results-and-checkpoints)
12. [Project structure](#12-project-structure)
13. [Reproducibility](#13-reproducibility)
14. [Documentation](#14-documentation)

> Spanish version of this same document: [`README.md`](README.md).

---

## 1. Hardware and installation

Targets **32 GB RAM + 16 GB VRAM**. All frozen models are precomputed to disk, so
training the decoder never loads Stable Diffusion.

```bash
python -m venv .venv && .venv\Scripts\activate       # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux / macOS
pip install -e .
```

Then install the CUDA build of PyTorch matching your driver. The environment
used for every result in the thesis was **Python 3.11.9** with
**torch 2.12.1+cu130 / torchvision 0.27.1+cu130**:

```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

**Exact reproduction.** `requirements.txt` pins every dependency to the exact
version of that environment (transformers 5.6.2, diffusers 0.37.1,
open_clip_torch 3.3.0, numpy 2.4.4, mne 1.12.1, ...):

```bash
pip install -r requirements.txt
```

Notes:

- `pip install -e .` installs the dependencies declared in `pyproject.toml`
  (unpinned ranges) and makes `import src...` work from the repository root; the
  scripts also work without installing, since they add the repo root to
  `sys.path` themselves.
- **MNE** is only required for the EEG raw-preprocessing pipeline (§6). It is
  included in `requirements.txt` and available as the `eeg` extra
  (`pip install -e .[eeg]`). If you only use the official derivatives you can
  skip it.
- **Always run from the repository root.** All project paths (`data/`,
  `outputs/`, `configs/`) are relative. Notebooks include a first cell that
  changes the working directory automatically; run it.
- Behind a corporate proxy, `pip` may need
  `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

Every CLI entry point accepts `--set key.path=value` overrides, for example
`--set dataset.subject_selection=all --set training.batch_size=32`.

---

## 2. Datasets

### 2.1 fMRI — NSD / Algonauts 2023

Data: <https://algonautsproject.com/2023/braindata.html> (devkit:
<https://github.com/gifale95/algonauts_2023>). Two on-disk layouts are
auto-detected.

**A) Official layout with released test fMRI** (recommended — full
post-challenge release):

```
<root>/train_data/subjNN/training_split/training_fmri/{lh,rh}_training_fmri.npy
<root>/train_data/subjNN/training_split/training_images/train-XXXX_nsd-YYYYY.png
<root>/train_data/subjNN/roi_masks/...
<root>/test_data/subjNN/test_split/test_fmri/{lh,rh}_test_fmri.npy
<root>/test_data/subjNN/test_split/test_images/test-XXXX_nsd-YYYYY.png
```

**B) Flat layout** (`<root>/subjNN/training_split/...`), used when the released
test fMRI is not available; the code then carves an internal test split from the
labelled training data.

```yaml
# configs/fMRI/base.yaml
dataset:
  root_dir: C:/Users/xxdia/Documents/Datasets/NSD_Algonauts_2023
  subject_selection: subj01          # or [subj01, subj02] or all
  test_split: official               # official (uses test_data) | internal
```

For `subj01`: 39 548 vertices (both hemispheres concatenated), 8 857 training
images, 984 validation and **159 official test images**.

Captions used by the multimodal architectures live in
`<root>/auxiliar/generated_captions/` and are produced by
[`notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb`](notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb),
which derives them from the COCO annotations associated with each NSD stimulus.

### 2.2 EEG — THINGS-EEG2

```
<root>/image_set/                                        stimulus images + image_metadata.npy
<root>/image_set/generated_captions/                     captions
<root>/preprocessed_data/<sub>/preprocessed_eeg_{training,test}.npy      17-channel montage
<root>/preprocessed_data/<sub>__63_channels/...                          63-channel montage
<root>/raw-eeg/<sub>/ses-0{1..4}/raw_eeg_{training,test}.npy             raw, 1 kHz (see §6)
```

Each preprocessed file is a dict whose `preprocessed_eeg_data` has shape
`[images, repetitions, channels, times]`: train `(16540, 4, C, 100)` and test
`(200, 80, C, 100)`, sampled at 100 Hz over −200→790 ms. Test concepts are
**disjoint** from training concepts, so evaluation also measures generalisation
to unseen categories.

```yaml
# configs/EEG/base_63.yaml
dataset:
  modality: eeg
  root_dir: C:/Users/xxdia/Documents/Datasets/THINGS-EEG2
  subject_selection: sub-01
  channels: 63                       # 17 | 63
  trial_aggregation: {train: none, val: mean, test: mean}
```

Training uses **per-trial** samples (data augmentation across repetitions), while
**evaluation and ablations always run at image level** (mean over repetitions),
so retrieval candidates are unique images and the SNR is high.

Captions for the multimodal architectures live in
`<root>/image_set/generated_captions/` and are produced by
[`notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb`](notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb),
which derives them from the THINGS concept name of each stimulus folder.

---

## 3. Full pipeline from scratch

The order is mandatory: each step produces artefacts the next one consumes. The
same scripts serve both modalities — the modality is selected by `--config`.

### 3.1 fMRI line

```bash
# 0) Resolve subjects, build reproducible splits, fit train-only normalization
python scripts/00_prepare_dataset.py          --config configs/fMRI/exp01_fmri_to_clip.yaml

# 1) Precompute frozen-CLIP image embeddings (the semantic targets)
python scripts/01_precompute_clip.py          --config configs/fMRI/exp01_fmri_to_clip.yaml

# 2) EXPERIMENT 1 — train brain → CLIP (Stable Diffusion is never loaded)
python scripts/02_train_fmri_to_clip.py       --config configs/fMRI/exp01_fmri_to_clip.yaml

# 3) EXPERIMENT 2 — retrieval ablation + mean/ridge baselines
python scripts/03_eval_retrieval_ablation.py  --config configs/fMRI/exp02_retrieval_ablation.yaml

# 4) Precompute VAE latents and fit PCA (train only) — the structural targets
python scripts/04_precompute_vae_pca.py       --config configs/fMRI/exp03_lowlevel_multitask.yaml

# 5) EXPERIMENT 3 — multitask decoder (semantic + structural heads)
python scripts/05_train_multitask.py          --config configs/fMRI/exp03_lowlevel_multitask.yaml

# 6) EXPERIMENT 4 — train the token adapter and generate with frozen SD
python scripts/06_generate_images.py          --config configs/fMRI/exp04_generation_legacy.yaml --train-adapter

# 7) EXPERIMENT 5 — generative comparison against the negative controls
python scripts/07_eval_generation_ablation.py --config configs/fMRI/exp05_generation_legacy_ablation.yaml
```

### 3.2 EEG line

Identical sequence with the EEG configs:

```bash
python scripts/00_prepare_dataset.py          --config configs/EEG/exp01_63_eeg_to_clip.yaml
python scripts/01_precompute_clip.py          --config configs/EEG/exp01_63_eeg_to_clip.yaml
python scripts/02_train_fmri_to_clip.py       --config configs/EEG/exp01_63_eeg_to_clip.yaml
python scripts/03_eval_retrieval_ablation.py  --config configs/EEG/exp02_63_retrieval_ablation.yaml
python scripts/04_precompute_vae_pca.py       --config configs/EEG/exp03_63_lowlevel_multitask.yaml
python scripts/05_train_multitask.py          --config configs/EEG/exp03_63_lowlevel_multitask.yaml
python scripts/06_generate_images.py          --config configs/EEG/exp04_63_generation_legacy.yaml --train-adapter
python scripts/07_eval_generation_ablation.py --config configs/EEG/exp05_63_generation_legacy_ablation.yaml
```

*(The name of script `02` is historical: it trains whichever modality the config
declares.)*

### 3.3 Extra precomputation for the multimodal architectures

Needed only for the architectures that use text and/or spatial control (§7).
Both caches depend **only on the stimulus images and the split**, never on the
brain signal, so they are computed once and shared by every configuration and
every EEG preprocessing variant.

**Step 0 — build the captions.** The caption CSVs are not shipped with the
datasets; they are generated once per modality with these notebooks, which write
into the `generated_captions/` folder of each dataset root:

| Modality | Notebook | Output |
|---|---|---|
| fMRI | [`notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb`](notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb) | `<root>/auxiliar/generated_captions/<subj>_{train,test}_captions.csv` |
| EEG | [`notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb`](notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb) | `<root>/image_set/generated_captions/thingseeg2_{training,test}_image_captions.csv` |

In fMRI the weak caption is a comma-separated list of the main nouns extracted
from the COCO description of each stimulus; in EEG it is the THINGS concept name
recovered from the image folder. Run the notebook of the modality you need before
`scripts/13`.

**Steps 1–2 — the caches themselves.**

```bash
# Text embeddings (frozen SD text encoder, deduplicated and hashed)
python scripts/13_precompute_text_embeddings.py       --config configs/fMRI/exp04_generation_text_weak.yaml

# ControlNet spatial conditions (GT image → VAE → PCA → inverse → decode → Canny)
python scripts/14_precompute_controlnet_conditions.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml
```

### 3.4 Auxiliary tools

```bash
python scripts/08_sweep_adapter_checkpoints.py  --config <exp04 config>   # pick adapter by generation quality
python scripts/12_sweep_adapter_input_scale.py  --config <exp04 config>   # conditioning strength sweep
python scripts/16_sweep_controlnet_scale.py     --config <exp04 config>   # spatial control scale sweep
python scripts/15_validate_multimodal.py        --config <exp04 config>   # CPU checks, seconds, no SD
python scripts/plot_memoria_figures.py                                    # thesis tables and figures
```

---

## 4. Experiments and final configurations

The pipeline defines five experiments:

| Experiment | Script | Question it answers |
|---|---|---|
| **Exp 1** | `02_train_fmri_to_clip.py` | Does the brain signal predict the semantic representation? |
| **Exp 2** | `03_eval_retrieval_ablation.py` | Does that depend on the *correct* signal? (controls + baselines) |
| **Exp 3** | `05_train_multitask.py` | Can it also predict low-level structure, without losing semantics? |
| **Exp 4** | `06_generate_images.py` | Can the predictions steer a frozen generator? |
| **Exp 5** | `07_eval_generation_ablation.py` | Does the generated image depend on the correct signal? |

The thesis reports **seven generative configurations**, all of them with
`generation.mode: adapter`. The codes below are the ones used in the report.

| Code | Modality / line | Architecture | Exp 4 config | Exp 5 config |
|---|---|---|---|---|
| **Af** | fMRI | A · brain only | `configs/fMRI/exp04_generation_legacy.yaml` | `configs/fMRI/exp05_generation_legacy_ablation.yaml` |
| **Bf** | fMRI | B · weak text + brain | `configs/fMRI/exp04_generation_text_weak.yaml` | `configs/fMRI/exp05_generation_text_weak_ablation.yaml` |
| **Cf** | fMRI | C · text + brain + spatial control | `configs/fMRI/exp04_generation_controlnet_weak.yaml` | `configs/fMRI/exp05_generation_controlnet_weak_ablation.yaml` |
| **Ae** | EEG, official 63 ch | A · brain only | `configs/EEG/exp04_63_generation_legacy.yaml` | `configs/EEG/exp05_63_generation_legacy_ablation.yaml` |
| **Be** | EEG, official 63 ch | B · weak text + brain | `configs/EEG/exp04_63_generation_text_weak.yaml` | `configs/EEG/exp05_63_generation_text_weak_ablation.yaml` |
| **Ce** | EEG, official 63 ch | C · text + brain + spatial control | `configs/EEG/exp04_63_generation_controlnet_weak.yaml` | `configs/EEG/exp05_63_generation_controlnet_weak_ablation.yaml` |
| **Ct** | EEG, own preprocessing, 100–600 ms | C · text + brain + spatial control | `configs/EEG/exp04_raw_temporal_100_600_generation_controlnet_weak.yaml` | `configs/EEG/exp05_raw_temporal_100_600_generation_controlnet_weak_ablation.yaml` |

Their decoding counterparts (Exp 1–3) are shared per line: `exp01_fmri_to_clip`,
`exp02_retrieval_ablation` and `exp03_lowlevel_multitask` for fMRI;
`exp01_63_eeg_to_clip`, `exp02_63_retrieval_ablation` and
`exp03_63_lowlevel_multitask` for the official EEG line; and the
`exp0X_raw_temporal_100_600_*` family for the Ct line.

**One adapter per (modality, architecture, text mode).** Adapters are *not*
shared across modalities: fMRI and EEG use different stimulus sets, so each
modality trains its own. That gives **three adapters per modality — six in total
— covering the seven configurations**:

| Adapter | Trained in | Also used by |
|---|---|---|
| fMRI · architecture A | Af | — |
| fMRI · architecture B | Bf | — |
| fMRI · architecture C | Cf | — |
| EEG · architecture A | Ae | — |
| EEG · architecture B | Be | — |
| EEG · architecture C | Ce | **Ct** |

The only reuse is **Ct taking Ce's adapter**: both are EEG, architecture C and
weak text, and they differ *only* in the EEG preprocessing variant — which the
adapter never sees, since it is trained on CLIP embeddings and VAE latents of the
stimulus images. Reusing it is not just a shortcut: it keeps the generation stage
identical between the two lines, so any difference in Experiment 5 is
attributable to the preprocessing and the decoder rather than to two separately
trained adapters.

---

## 5. Worked example: reproducing configuration Cf

Full path from an empty `data/` and `outputs/` to
`exp05_generation_controlnet_weak_ablation`, the fMRI configuration with text and
spatial control.

```bash
# --- Stage 0: data preparation and frozen targets -------------------------
python scripts/00_prepare_dataset.py     --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/01_precompute_clip.py     --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/04_precompute_vae_pca.py  --config configs/fMRI/exp03_lowlevel_multitask.yaml

# --- Stage 1: brain decoder ------------------------------------------------
python scripts/02_train_fmri_to_clip.py  --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/05_train_multitask.py     --config configs/fMRI/exp03_lowlevel_multitask.yaml
```

Architecture C needs the **multitask** decoder (Exp 3), because its spatial
condition is derived from the structural prediction.

```bash
# --- Stage 2: negative-control ablation of the decoder ---------------------
# (a) on the Exp 1 decoder — the semantic-only model
python scripts/03_eval_retrieval_ablation.py --config configs/fMRI/exp02_retrieval_ablation.yaml

# (b) on the Exp 3 decoder — the one that actually feeds generation.
#     Same config, redirected source experiment and a separate output folder.
python scripts/03_eval_retrieval_ablation.py \
  --config configs/fMRI/exp02_retrieval_ablation.yaml \
  --set evaluation.source_experiment=exp03_fmri_lowlevel_multitask \
        experiment.name=exp02_fmri_retrieval_ablation_exp3
```

Variant **(b)** is the one reported in the thesis, so that the whole reported
chain uses a single decoder. Variant (a) is kept as a consistency check; both
reach the same qualitative verdict.

```bash
# --- Stage 3: multimodal caches (shared, computed once) --------------------
# Prerequisite: run notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb
# once, so the caption CSVs exist under <root>/auxiliar/generated_captions/.
python scripts/13_precompute_text_embeddings.py       --config configs/fMRI/exp04_generation_controlnet_weak.yaml
python scripts/14_precompute_controlnet_conditions.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml

# --- Stage 4: train the token adapter and generate -------------------------
python scripts/06_generate_images.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml --train-adapter

# --- Stage 5: paired evaluation against every control ----------------------
python scripts/07_eval_generation_ablation.py --config configs/fMRI/exp05_generation_controlnet_weak_ablation.yaml
```

This produces the eight conditions of architecture C and the paired deltas
(`delta_brain`, `delta_text`, `delta_semantic`, `delta_lowlevel` and their null
variants) with t-test and Wilcoxon p-values.

**Reusing an already-trained adapter** (for another preprocessing variant, or to
avoid retraining):

```bash
python scripts/06_generate_images.py \
  --config configs/EEG/exp04_raw_temporal_100_600_generation_controlnet_weak.yaml \
  --set generation.train_adapter=false \
  --adapter-checkpoint outputs/exp04_63_eeg_generation_controlnet_weak/checkpoints/adapter_best.pt
```

**Fast smoke test** (minutes instead of hours, not a result):

```bash
python scripts/06_generate_images.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml --train-adapter \
  --set generation.adapter_epochs=1 generation.adapter_max_train_samples=64 \
        generation.num_samples=4 generation.num_inference_steps=6
```

---

## 6. EEG: own preprocessing from raw + ablations

Besides the official derivatives, the EEG line can start from the **raw
63-channel recordings** and apply a parameterised pipeline of our own, which also
enables a battery of preprocessing ablations. Details in §10 of
[`docs/00_documentacion_general.md`](docs/00_documentacion_general.md).

Reference configuration (`configs/EEG/preproc/baseline.yaml`): 63 channels,
0.1–100 Hz band-pass on the continuous signal, epochs −200…1000 ms, baseline over
the 200 ms before onset, resampling to 250 Hz, half-open crop `[0, 1000)` ms
(**exactly 250 samples**), multivariate noise normalisation (MVNN) fitted on
training images only and applied **to every repetition before averaging**,
4-repetition average in training and all 80 test repetitions kept →
`63 × 250` per image.

```bash
# 1) Build a variant (once per variant and subject) — needs the venv interpreter
.tfm_fmri_diffusion_3_11/Scripts/python.exe scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/baseline.yaml

# 2) Validate it (exact shapes, no MVNN leakage, disjoint splits, channel list)
python scripts/10_validate_eeg_preproc.py  --config configs/EEG/preproc/baseline.yaml

# 3) Use it in the experiments (ready-made configs per variant)
python scripts/00_prepare_dataset.py       --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
python scripts/02_train_fmri_to_clip.py    --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml

# 4) Test-repetition curve (R = 1…80) — an evaluation protocol, no retraining
python scripts/11_eval_test_repetitions.py --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
```

> Step `09` **requires the project venv interpreter**: MNE is installed there and
> not in the system Python. The baseline pins `preprocessing.filter.backend: mne`
> so a wrong interpreter fails in about two seconds with an actionable message,
> instead of silently degrading to a scipy IIR filter that would make the variant
> incomparable with the rest. The variant cache is also guarded by a config hash:
> editing a preprocessing parameter and re-running without `--force` is refused.

Ten one-factor ablations ship as minimal overrides in `configs/EEG/preproc/`:
`ablate_mvnn`, `channels_17`, `temporal_100_600`, `temporal_200_400`,
`sampling_100hz`, `frequency_0_5_40`, `train_independent_trials`,
`reference_car`, `baseline_minus100` and `baseline_none`. Each one branches from
the raw signal — never from another variant — so the manipulated factor is truly
the only thing that changes.

A variant writes `data/processed/eeg_preproc/<variant>/<subject>/` using the
**same file contract as the official derivatives**, so the datamodule, the
encoder and Exp 1–5 consume it unchanged.

### What to re-run when the preprocessing changes

| Step | Re-run? | Why |
|---|---|---|
| `09` (build variant) and `00` (metadata) | **Yes** | The tensor and the normalisation are variant-specific |
| `01` (CLIP) and `04` (VAE + PCA) | **No** | They depend only on the images and the split, and stay aligned by index |
| Exp 1, Exp 2, Exp 3 | **Yes** | The brain input changed |
| Adapter training | **No** | The adapter maps CLIP to VAE latents and never sees the EEG |
| Generation (Exp 4) and Exp 5 | **Yes** | The decoder changed |

This holds as long as `dataset.val_ratio` and `dataset.split_seed` are unchanged.

---

## 7. Conditioning architectures: text and spatial control

Stable Diffusion 1.5 stays frozen. The brain signal can reach it through three
doors: the **conditioning sequence** (cross-attention), the **initial latent**,
and **spatial residuals** injected by a control network. On top of that,
`generation.conditioning_architecture` selects which information sources
accompany the neural condition:

| Value | Condition fed to the frozen UNet |
|---|---|
| `legacy_adapter` *(default)* | `[K neural tokens]` — brain only, no caption |
| `text_adapter_concat` | `[77 text tokens ; K neural tokens]` via cross-attention |
| `text_adapter_concat_controlnet` | the above **plus** frozen ControlNet residuals derived from the structural prediction |

The control network does **not** replace the concatenation: text and neural
tokens keep arriving through cross-attention while it adds spatial residuals
built from `low_pred → inverse PCA → VAE decode → Canny`. The **token adapter
remains the only trainable module**.

`generation.mode` is an **orthogonal axis** (`adapter`, `lowlevel_img2img`,
`adapter_lowlevel`) that decides where the structural prediction enters. All
reported experiments use `adapter`, which is the mode whose contrasts are
attributable to a single source. Since the mode is not part of the adapter's
identity, switching it requires no retraining.

**Text modes.** `generation.text.mode: none | weak | oracle` resolves to
`prompt_categories` / `primary_caption` in fMRI and to `primary_caption` in EEG,
so in EEG **oracle == weak** (THINGS-EEG2 has no more detailed caption) and it
must not be reported as a more informative condition. `weak` is the textual
*baseline*; `oracle` is a declared performance ceiling.

**Permuted captions** are a Sattolo derangement with seed 42 drawn **within each
`(subject, split)` and within the same caption family**, using the very same
helper as the brain permutation.

**Adapter checkpoints are not interchangeable.** Each one stores its conditioning
identity (architecture, text mode, caption field, template, `K`, control network
model and condition type). Loading it under a different one fails with an
explicit error unless `generation.allow_incompatible_adapter: true`, which is
meant for smoke tests only.

---

## 8. Stop and resume training

Trainings checkpoint the *full* state: model, optimizer, scheduler, gradient
scaler, epoch, `global_step`, best metric, early-stopping counter, RNG state,
resolved config and library versions.

```bash
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml --resume
#                                                                              ^ resumes last.pt
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml --resume path/to/last.pt
```

or, in the config:

```yaml
checkpointing:
  resume: auto      # looks for checkpoints/last.pt in the experiment output dir
```

`last.pt` and `best.pt` are never pruned; only periodic `epoch_XXXX.pt` beyond
`keep_last_n` are removed. Each resume appends to `logs/resume_history.jsonl` and
continues `logs/train_log.csv` without losing previous rows.

> Each fMRI checkpoint is about 2.7 GB (weights plus optimizer state). Use
> `checkpointing.save_every_n_epochs: 5` and `keep_last_n: 1` to save disk.

---

## 9. Outputs

```
outputs/<experiment>/
  config.yaml                                   fully resolved configuration
  checkpoints/{last,best,epoch_XXXX}.pt         decoder, or adapter_{best,last}.pt in Exp 4
  logs/{train_log.csv, resume_history.jsonl}
  metrics/*.{json,csv}                          per-experiment metrics
  figures/*.png                                 curves, per-condition bars, PCA correlation
  embeddings/*.npy   lowlevel/*.npy             predictions and targets
  generated/{real,correct,permuted,zero,...}/*.png
  grids/*.png                                   comparison and best/median/worst grids
  metadata/{generation_params.json, generation_samples.json}
  report/*.md                                   human-readable summary and verdict
```

With the multimodal architectures, `generated/` also holds the extra conditions
(`permuted_text`, `semantic_*`, `lowlevel_*`), `metrics/generation_deltas.csv`
holds the paired deltas with their p-values, and `metadata/generation_samples.json`
holds one record per (sample, condition) with everything needed to rebuild that
exact image: resolved prompt, brain and text condition, seeds and control scale.

Aggregated metrics use a tidy format: `metric_name, condition, subject_id, split,
value, seed, checkpoint`. The verdict lives in `metrics/conclusion.json` (Exp 2)
and `report/exp05_summary.md` (Exp 5).

---

## 10. Interpreting the controls

| Condition | What it is | Expected |
|---|---|---|
| **Correct** | The real brain response for each image | Clearly above chance |
| **Permuted** | Each sample receives *another* sample's response (Sattolo derangement, never its own) | ≈ chance |
| **Null** | A zero brain vector, propagated through the decoder | ≈ chance |
| **Noise** | Gaussian noise with comparable statistics | ≈ chance |

Chance levels depend on the number of candidates: with `N` candidates, Top-k is
about `k/N`. For fMRI test (159 candidates) Top-5 ≈ 3.14 %; for EEG test (200
candidates) Top-5 ≈ 2.50 %. **Never compare splits against each other** — compare
each one against its own chance level.

Two warnings that the code documents explicitly:

- **Mean cosine similarity is misleading.** CLIP embeddings are anisotropic, so
  the mean-predictor baseline reaches a cosine around 0.75 while performing
  exactly at chance in retrieval. Retrieval is the honest metric.
- **The R² of the semantic head is negative by design** (it is optimised for
  direction, not magnitude). R² is informative only for the structural branch.

Decoding (Exp 2) and generation (Exp 5) verdicts are reported **separately**: a
signal may decode above chance and still fail to steer generation.

---

## 11. Published results and checkpoints

What is published is not just the checkpoints: it is the **complete output
folders** of each experiment, exactly as the system writes them, for the seven
reported configurations. They are hosted on the UOC Drive folder:

> ⬜ **Link:** `https://drive.google.com/drive/folders/1br4gnjzXwemb_9REs0okzS4cmvpgywlx?usp=sharing`

Structure:

```
Arquitectura Ae (EEG, solo señal)                                 Exp 4 + Exp 5
Arquitectura Af (fMRI, solo señal)                                Exp 4 + Exp 5
Arquitectura Be (EEG, texto débil + señal)                        Exp 4 + Exp 5
Arquitectura Bf (fMRI, texto débil + señal)                       Exp 4 + Exp 5
Arquitectura Ce (EEG, texto + señal + control espacial)           Exp 4 + Exp 5
Arquitectura Cf (fMRI, texto + señal + control espacial)          Exp 4 + Exp 5
Arquitectura Ct (EEG 100-600ms, texto + señal + control espacial) Exp 4 + Exp 5
Ejecuciones comunes - Arquitecturas EEG                           Exp 1 + Exp 2 + Exp 3
Ejecuciones comunes - Arquitecturas fMRI                          Exp 1 + Exp 2 + Exp 3
Ejecuciones comunes - Arquitectura Ct (EEG 100-600ms)             Exp 1 + Exp 2 + Exp 3
```

- Each **architecture folder** contains its Experiment 4 and Experiment 5
  outputs. The **adapter checkpoint (`adapter_best.pt`)** lives in the
  Experiment 4 folder.
- The **shared-runs folders** contain Experiments 1, 2 and 3 for that modality —
  the decoders that feed every architecture of that line.

### What each folder contains

Each subfolder is a complete `outputs/<experiment>/` directory with the structure
described in §9. Specifically:

- **`config.yaml`**, the fully resolved configuration that experiment ran with. It
  is the authoritative record of the parameters used: it lets you check exactly
  what was done, without relying on the files under `configs/` having stayed
  unchanged afterwards.
- **`metrics/`**, every metric in JSON and CSV: retrieval per condition, reference
  baselines, structural-branch metrics, generation metrics per condition, paired
  contrasts with their p-values, and the automatic verdict.
- **`figures/`** and **`grids/`**, the generated figures and grids, including the
  per-condition comparisons and the best, median and worst cases.
- **`generated/`**, the generated images for **all** conditions of that
  experiment, alongside the real reference image.
- **`embeddings/`** and **`lowlevel/`**, predictions and targets as `.npy`, which
  allow any metric to be recomputed without running the model again.
- **`metadata/`**, the generation parameters plus one record per (sample,
  condition) with everything needed to rebuild that specific image.
- **`logs/`** and **`report/`**, the per-epoch training history and the
  human-readable summary with the conclusion.

### About the checkpoints

To keep the total size reasonable, **only the `best` checkpoints were kept**:
`best.pt` for the decoding experiments and `adapter_best.pt` for the generation
ones. The `last.pt` files and the periodic `epoch_XXXX.pt` snapshots were removed —
in the fMRI line each one is about 2.7 GB.

The practical consequence is that this material is enough to **evaluate, generate
and reproduce every table and figure without retraining anything**, since the
evaluation and generation scripts use precisely the `best` checkpoint. What is not
possible is to *resume* a training exactly where it stopped, as that requires the
corresponding `last.pt`.

### How to reuse them

Copy each subfolder into `outputs/` keeping its experiment name — the one recorded
in `experiment.name` in its configuration, which can also be read from the included
`config.yaml` — and the evaluation and generation scripts will find them
automatically. To generate without retraining the adapter, pass
`--set generation.train_adapter=false` together with `--adapter-checkpoint`.

### Also in this repository

These same output folders are **versioned in this repository**, under `outputs/`,
so the metrics, figures and reports of every experiment can be browsed directly on
GitHub without downloading anything. What `.gitignore` excludes is only the heavy
or regenerable part:

| In the repository | Drive only |
|---|---|
| `config.yaml` (resolved configuration) | `checkpoints/` (model weights) |
| `metrics/` (JSON and CSV) | `generated/` (individual images) |
| `figures/` and `grids/` (PNG) | `embeddings/` and `lowlevel/` (`.npy`) |
| `logs/` (per-epoch history) | Any `.npy`, `.npz`, `.pkl`, `.pt` or `.pth` |
| `metadata/` (JSON) and `report/` (Markdown) | |

Note that **the grids under `grids/` are versioned**: the individual generated
images are not pushed, but the comparison grids — real image against each
condition, plus the best, median and worst cases — are, so the qualitative
comparison can also be reviewed from the repository.

In short: the repository is enough to **read** the results; downloading the Drive
folders is needed to **re-run** generation or evaluation, or to reuse a decoder or
an adapter.

The datasets themselves are **not** redistributed in either place; they must be
downloaded from their original sources (§2).

---

## 12. Project structure

```
configs/fMRI        base.yaml + one YAML per experiment (composed via `_base_`)
configs/EEG         same for the EEG line, plus preproc/ with the raw pipeline variants
scripts/            00–16 CLI entry points (modality selected by --config) + figure generation
notebooks/          visual companions: {fMRI,EEG}/00..06 and 30_multimodal_comparison
docs/               full technical documentation (see §14)

src/data            fMRI datamodule, EEG datamodule and dataset, normalizers, split, captions, factory
src/preprocessing   own EEG pipeline from raw: loader, filters, epoching, MVNN, QC, variant builder
src/features        CLIP embeddings, VAE latents, train-only PCA, text embeddings, feature loading
src/models          brain encoders (fMRI, EEG), prediction heads, adapters, multitask decoder
src/losses          cosine, InfoNCE contrastive, multitask combination
src/training        train/validation loops, full checkpointing and resume
src/evaluation      retrieval, embedding and generation metrics, baselines, ablations, deltas
src/generation      frozen-SD pipeline, conditioning architectures, control conditions, sweeps, grids
src/utils           config, seeding, device/AMP, logging, paths, checkpointing, permutations
```

**Golden rule:** all logic lives in `src/`. Scripts and notebooks only
orchestrate and import from it, so there is no duplicated implementation between
the interactive and the batch paths.

---

## 13. Reproducibility

- Seeds are fixed for Python, NumPy and PyTorch. Negative-control permutations
  use a stable, process-independent seed, so two runs of the same experiment draw
  the same permutation.
- Splits are computed by a **shared function** used by both the preprocessing
  pipeline and the datamodule, so statistics fitted "on training" are fitted on
  exactly the images the training loop calls training.
- Precomputed artefacts are guarded by configuration hashes: changing a parameter
  without recomputing fails with an actionable message instead of silently mixing
  incompatible caches.
- Every checkpoint stores the resolved configuration and library versions; every
  generation stores enough metadata to rebuild each image.
- Automatic checks: `scripts/10_validate_eeg_preproc.py` (exact shapes, MVNN
  leakage, disjoint splits) and `scripts/15_validate_multimodal.py` (shapes,
  caption alignment and permutation, disabled control equivalence, adapter
  compatibility) run on CPU in seconds.

---

## 14. Documentation

The full technical documentation lives in a single document:

**[`docs/00_documentacion_general.md`](docs/00_documentacion_general.md)** (in
Spanish). It covers the system design and the five functional blocks, the code
organisation, the **configuration system and how to write your own configs**, the
data flow and on-disk artefacts, the five experiments, both processing blocks in
detail, the evaluation protocol with its controls and metrics, the EEG line,
reproducibility, computational costs and troubleshooting.

---

## Citation

If you use this code, please cite the associated Master's thesis.

> ⬜ **PENDIENTE:** add the final citation (author, title, university, year) once
> the thesis is deposited.
