# Documento 4 — Ampliación multimodal: línea EEG (THINGS-EEG2)

> Amplía los Documentos 1–3 (fMRI) incorporando **EEG** como segunda modalidad dentro del
> **mismo marco**. Explica *qué* cambia y *qué se reutiliza*, la estructura del dataset
> THINGS-EEG2, el encoder EEG, las decisiones de agregación de repeticiones y cómo se ejecuta.
> El principio rector no cambia: `correcto ≫ permutado ≈ cero` decide si hay decodificación real.
>
> Para el **pipeline de preprocesamiento propio desde el EEG raw** (63 canales, MVNN, 250 Hz) y
> sus **ablaciones de preprocesamiento**, ver `docs/05_preprocesamiento_eeg_raw_y_ablaciones.md`.

---

## 1. Idea y motivación

La línea fMRI (NSD Algonauts) demostró decodificación visual verificable (ver Doc 2 §8). La
ampliación añade **EEG** (dataset THINGS-EEG2) **sin duplicar el proyecto**: los *targets*
visuales (embedding CLIP, latente VAE, PCA de bajo nivel), las cabezas, las pérdidas, las
métricas, las ablaciones (correcto/permutado/cero/ruido), la generación con Stable Diffusion
congelado y el checkpointing son **los mismos**. Solo cambian **el dataset y el encoder**,
porque la naturaleza de la señal es distinta:

- **fMRI**: respuesta **espacial** por imagen, un vector `[V]` → MLP residual (`FMRIEncoder`).
- **EEG**: señal **temporal multicanal** `[C, T]` (canales × tiempo) → red convolucional
  temporal (`EEGEncoderTemporalConv`).

El diseño es un **switch de modalidad aditivo**: `dataset.modality: fmri|eeg` elige, vía
`src/data/factory.py::build_datamodule` y `src/models/multitask_decoder.py::build_model`, el
datamodule y el encoder. La línea fMRI no cambia de lógica.

---

## 2. Dataset THINGS-EEG2

Estructura en `C:/Users/xxdia/Documents/Datasets/THINGS-EEG2`:

```
image_set/image_metadata.npy                 # dict con listas de ficheros train/test
image_set/training_images/<concepto>/<fichero>.jpg   # 16 540 imágenes únicas (1 654 conceptos)
image_set/test_images/<concepto>/<fichero>.jpg       # 200 imágenes únicas (conceptos disjuntos)
preprocessed_data/<sub>/preprocessed_eeg_{training,test}.npy         # 17 canales (por defecto)
preprocessed_data/<sub>__63_channels/preprocessed_eeg_*.npy          # 63 canales (+ 'stim')
```

Cada `.npy` es un **dict** con `preprocessed_eeg_data` de forma
`[imágenes, repeticiones, canales, tiempos]`:

| split | forma (17 canales) | repeticiones/imagen |
|---|---|---|
| training | `(16540, 4, 17, 100)` | 4 |
| test | `(200, 80, 17, 100)` | 80 |

`ch_names` (17 occipito-parietales: Pz, P3, O1, Oz…) y `times` (−0.2 → 0.79 s, 100 Hz → 100
muestras). La variante de 63 canales trae 64 columnas siendo la última `stim`, que **se
descarta** (canal de disparo, no EEG). Se lee con **numpy** (sin dependencia de MNE). Sujetos
descomprimidos actualmente: `sub-01`, `sub-08` (hay 10 en total).

**Alineación imagen↔EEG↔target**: la fila `i` del array EEG corresponde a `train_img_files[i]`
(o `test_img_files[i]`), cuya ruta es `training_images/<concepto_i>/<fichero_i>`. De ahí salen
los targets CLIP/VAE, calculados **una vez por imagen única**.

---

## 3. Splits, `feat_idx` y agregación de repeticiones (la parte sutil)

Los splits se hacen **por imagen** (todas las repeticiones de una imagen caen en el mismo
split, sin fuga): train/val se reparten de las 16 540 imágenes de entrenamiento (`val_ratio`),
y test son las 200 imágenes oficiales.

`EegDataModule` usa **dos vistas** del mismo metadato para que el código compartido no cambie:

- `subject_split_frame(subject, split)` → **una fila por imagen única** (orden `feat_idx`).
  La usan el precómputo de features y `load_subject_matrices`; los arrays CLIP/VAE/PCA son
  por-imagen y se alinean por identidad.
- `get_frame(split)` → filas **conscientes de la agregación** (`dataset.trial_aggregation`):
  - `none` → una fila por **(imagen, repetición)** = trial (aumento de datos en entrenamiento);
  - `mean` → una fila por imagen (media sobre repeticiones).

`feat_idx` = rango de la imagen dentro de `(sujeto, split)`; el dataset lee `arr[feat_idx]`, así
que cada trial recibe el target CLIP/PCA de **su** imagen.

**Decisión (importante para la memoria):** por defecto `{train: none, val: mean, test: mean}`.
El **entrenamiento es por-trial**; la **evaluación y las ablaciones son a nivel de imagen**
(media sobre las repeticiones). Esto es lo correcto y estándar en THINGS-EEG2: el conjunto de
candidatos del retrieval son **imágenes únicas** (no trials duplicados) y el promedio sobre 80
repeticiones del test eleva mucho la SNR. Es una mejora metodológica deliberada sobre "eval por
trial".

**Normalización por canal** (`EegNormalizer`), ajustada **solo con train**, media/desv por
canal (agrupando sobre trials y tiempo), aplicada a train/val/test; se guarda a disco
namespaced por nº de canales (`<sub>_eeg{C}ch_norm.npz`).

---

## 4. Encoder EEG

`src/models/eeg_encoder.py::EEGEncoderTemporalConv` (`[B, C, T] → [B, output_dim]`):

```
EEG [B, C, T]
  -> BatchNorm por canal + ChannelDropout
  -> Conv1D temporal (stem, mezcla canales)  -> GroupNorm -> GELU
  -> Conv separable/depthwise temporal
  -> N bloques temporales residuales (pre-norm)
  -> pooling temporal (atención | media)     # robusto a T
  -> LayerNorm + Linear(-> output_dim=2048)
  -> h
```

Mantiene el **contrato `output_dim`** (2048, igual que el encoder fMRI), de modo que las cabezas
`CLIPHead`/`LowLevelHead` y el `TokenAdapter` de generación se reutilizan **sin cambios**. Es
modesto en parámetros (~4,6 M con 17 canales) porque la señal es pequeña y conviene no
sobreajustar. El pooling temporal lo hace **robusto al número de muestras `T`** (distinta
ventana o frecuencia).

**Multi-sujeto**: se soporta a nivel de datos (normalización por sujeto). Se recomienda empezar
single-subject. Los adaptadores por sujeto de la línea fMRI (Linear plano sobre `[V]`) **no** se
usan en EEG (todos los sujetos comparten `(C, T)`); un adaptador EEG específico es extensión
futura.

---

## 5. Qué se reutiliza y qué es nuevo

| Componente | fMRI | EEG |
|---|---|---|
| Dataset/reader | `SubjectData`, `AlgonautsDataset`, `FmriDataModule` | `EegSubjectData`, `EegDataset`, `EegDataModule` (nuevos) |
| Normalización | por vóxel (`FmriNormalizer`) | por canal (`EegNormalizer`, nuevo) |
| Encoder | `FMRIEncoder` (MLP) | `EEGEncoderTemporalConv` (nuevo) |
| Targets CLIP/VAE/PCA | precómputo compartido (`src/features/*`) | **el mismo** (por imagen THINGS) |
| Cabezas / pérdidas | `CLIPHead`/`LowLevelHead`, `MultitaskLoss` | **las mismas** |
| Retrieval / ablación / baselines | `src/evaluation/*` | **los mismos** |
| Generación (adapter → SD-1.5) | `src/generation/*` | **la misma** |
| Checkpointing / resume | `src/utils/checkpointing` | **el mismo** |

Cambios en código compartido: `build_datamodule` (factory), rama de modalidad en
`build_model`/`MultitaskDecoder`, y sustituir `FmriDataModule(cfg)` por `build_datamodule(cfg)`
en los puntos de entrada. Nada más aguas abajo.

---

## 6. Configuración y ejecución

Configs en `configs/EEG/` (espejo de `configs/fMRI/`): `base.yaml` + `exp01_eeg_to_clip`,
`exp02_retrieval_ablation`, `exp03_lowlevel_multitask`, `exp04_generation`,
`exp05_generation_ablation`. Los mismos scripts 00–08 sirven; la modalidad la fija el `--config`:

```bash
python scripts/00_prepare_dataset.py         --config configs/EEG/exp01_eeg_to_clip.yaml
python scripts/01_precompute_clip.py          --config configs/EEG/exp01_eeg_to_clip.yaml
python scripts/02_train_fmri_to_clip.py       --config configs/EEG/exp01_eeg_to_clip.yaml
python scripts/03_eval_retrieval_ablation.py  --config configs/EEG/exp02_retrieval_ablation.yaml
python scripts/04_precompute_vae_pca.py       --config configs/EEG/exp03_lowlevel_multitask.yaml
python scripts/05_train_multitask.py          --config configs/EEG/exp03_lowlevel_multitask.yaml
python scripts/06_generate_images.py          --config configs/EEG/exp04_generation.yaml --train-adapter
python scripts/07_eval_generation_ablation.py --config configs/EEG/exp05_generation_ablation.yaml
```

Notebooks visuales en `notebooks/EEG/00..06`; `notebooks/30_multimodal_comparison.ipynb` compara
las dos líneas. Palancas EEG útiles por CLI: `--set dataset.channels=63`,
`--set dataset.subject_selection=sub-08`, `--set dataset.time_window_ms='[0, 800]'`.

---

## 7. Interpretación (EEG)

- **Retrieval a nivel de imagen** es la métrica honesta (igual que en fMRI). Azar test (200
  candidatos): Top-1 ≈ 0,5%, Top-5 ≈ 2,5%, Top-10 ≈ 5%; val (~1 654) Top-1 ≈ 0,06%. Compara cada
  split con **su** azar.
- El EEG suele decodificar por encima del azar en retrieval, pero la reconstrucción visual es más
  difícil que en fMRI y puede **no** guiar la generación con claridad. **Separa siempre** la
  conclusión de decodificación (Exp2) de la de generación (Exp5) y **reporta resultados negativos**
  si las ablaciones no separan `correcto` de los controles.
- Como en fMRI: no te fíes del coseno absoluto; el R² del CLIPHead es negativo por diseño y solo
  es informativo para la rama low-level (PCA).

---

## 8. Validación sin GPU

Patrón de smoke-test (torch CPU, python base) sobre `sub-01` real + features CLIP/PCA
**fabricadas** (aleatorias) alineadas por `feat_idx`: `build_datamodule().prepare()`,
`build_model` (encoder EEG), forward `[B, 17, 100] → clip/low`, un par de pasos de entrenamiento
y `evaluate_ablation`. Con targets aleatorios el veredicto debe salir `fmri_not_clearly_used`
(no hay señal que decodificar) — confirma que la maquinaria de ablación **reporta con honestidad**.
Los entrenamientos reales (con CLIP/VAE/SD) se lanzan en el venv GPU del proyecto.
