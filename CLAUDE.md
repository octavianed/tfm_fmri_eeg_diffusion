# CLAUDE.md

Guía operativa para trabajar en este repositorio. Para explicación conceptual y de
resultados en profundidad, ver `docs/01_objetivos_y_experimentos.md`,
`docs/02_guia_tecnica_y_resultados.md`, `docs/03_lowlevel_multitarea_y_generacion.md`
(VAE+PCA, multitarea, métricas y modos de generación),
`docs/04_ampliacion_eeg_multimodal.md` (línea EEG / THINGS-EEG2) y
`docs/05_preprocesamiento_eeg_raw_y_ablaciones.md` (pipeline propio desde el EEG raw,
ablaciones de preprocesamiento y qué reejecutar) y
`docs/06_tokenadapter_y_generacion.md` (qué espacios conecta el TokenAdapter, con qué se entrena
y el proceso de generación paso a paso) y
`docs/07_decoder_cerebro_a_clip.md` (el decoder cerebro→CLIP: Exp1 vs Exp3, encoders por
modalidad, pérdidas y bucle de entrenamiento) — orientados al autor del TFM.

## Qué es este proyecto

Decodificación y reconstrucción visual a partir de señal cerebral, como línea del TFM. Dos
**modalidades** dentro del **mismo marco** (mismos targets CLIP/VAE-PCA, mismas cabezas,
pérdidas, ablaciones y flujo experimental; solo cambian dataset y encoder):
- **fMRI** — dataset **NSD Algonauts 2023** (línea original).
- **EEG** — dataset **THINGS-EEG2** (ampliación; ver `## Modalidad EEG`).

Diseño en dos etapas, deliberadamente separadas:
1. **Decodificación**: ¿la señal cerebral predice una representación visual de la imagen vista
   (embedding CLIP semántico y/o PCA de latentes VAE de bajo nivel)? — medible sin generar.
2. **Generación**: usar esas predicciones para guiar Stable Diffusion **congelado**.

**Principio rector (falsable, es el corazón del proyecto):**
`correcto ≫ permutado ≈ cero`. Si no se cumple, NO se afirma que el modelo
use señal cerebral real (aunque las imágenes “se vean bien”). Está codificado en
`src/evaluation/ablation_eval.py::conclusion_from_summary`.

## Regla de oro (arquitectura)

- **La lógica vive en `src/`**. `scripts/` y `notebooks/` solo orquestan e importan de `src/`.
  No dupliques lógica de entrenamiento/evaluación en notebooks: llama a las mismas funciones.
- `torch`/`diffusers`/`open_clip`/`transformers` se importan **de forma perezosa** dentro de
  funciones, para que importar un módulo no arrastre dependencias pesadas.

## Convenciones críticas (léelas antes de tocar nada)

- **Ejecutar SIEMPRE desde la raíz del repo.** Las rutas del proyecto son relativas
  (`data/`, `outputs/`, `configs/`). Los scripts se lanzan como `python scripts/0X_*.py`.
  Los **notebooks** llevan una primera celda que hace `os.chdir()` a la raíz (busca la carpeta
  con `src/` y `configs/`): hay que ejecutarla. Si el cwd queda en `notebooks/`, se crean
  `notebooks/data` y `notebooks/outputs` fantasma y falla (“No precomputed CLIP features”).
- **Config**: por modalidad → `configs/fMRI/base.yaml` y `configs/EEG/base.yaml`, cada uno con
  un `expNN.yaml` por experimento con `_base_: [base.yaml]` (solo sobrescribe lo que cambia).
  `dataset.modality` (`fmri`|`eeg`) selecciona datamodule + encoder vía
  `src/data/factory.py::build_datamodule` y `src/models/multitask_decoder.py::build_model`.
  Overrides por CLI: `--set clave.subclave=valor`. El loader es `src/utils/config.py`
  (acceso por punto: `cfg.training.lr`, `cfg.get("a.b", default)`).
- **Dataset**: layout auto-detectado. El actual es el “oficial” con test fMRI liberado:
  `<root>/train_data/subjNN/...` + `<root>/test_data/subjNN/test_split/test_fmri/...`.
  `dataset.test_split: official` usa `test_data` como test real; `internal` lo recorta del
  train (fallback si no hay `test_data`). subj01: V=39 548 vértices (19 004 lh + 20 544 rh),
  9 841 imágenes train, 159 test.
- **Features precomputadas y alineadas por `feat_idx`**: CLIP/VAE/PCA se guardan a `.npy` en
  `data/features/...`, con la fila `k` correspondiente al `feat_idx` `k` dentro de
  `(sujeto, split)`. El `metadata_*.csv` lleva columna **`source`** ('train'/'test') que indica
  de qué array de fMRI leer. No rompas esta alineación.
- **Modelos congelados**: CLIP, VAE y Stable Diffusion NUNCA se entrenan. Lo único entrenable
  es el `fMRIEncoder`+cabezas (Exp1/3) y el pequeño `TokenAdapter` (Exp4).
- **Normalización solo en train**, guardada a disco. Nada de leakage (PCA, scaler, baselines
  se ajustan solo con train).
- **Multi-sujeto**: si hay >1 sujeto, el modelo activa adaptadores por sujeto (distinto nº de
  vértices) y un batch sampler homogéneo por sujeto.

## Pipeline (orden obligatorio)

Los mismos scripts sirven para ambas modalidades; la modalidad la fija el `--config`
(`configs/fMRI/...` o `configs/EEG/...`). Ejemplo fMRI:

```
python scripts/00_prepare_dataset.py        --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/01_precompute_clip.py         --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/02_train_fmri_to_clip.py      --config configs/fMRI/exp01_fmri_to_clip.yaml   # --resume para reanudar
python scripts/03_eval_retrieval_ablation.py --config configs/fMRI/exp02_retrieval_ablation.yaml
python scripts/04_precompute_vae_pca.py      --config configs/fMRI/exp03_lowlevel_multitask.yaml
python scripts/05_train_multitask.py         --config configs/fMRI/exp03_lowlevel_multitask.yaml
python scripts/06_generate_images.py         --config configs/fMRI/exp04_generation.yaml --train-adapter
python scripts/07_eval_generation_ablation.py --config configs/fMRI/exp05_generation_ablation.yaml
python scripts/08_sweep_adapter_checkpoints.py --config configs/fMRI/exp04_generation.yaml  # opcional: elegir checkpoint del adapter por calidad
```
Para **EEG**, el mismo orden con `configs/EEG/exp0X_*.yaml` (p. ej.
`scripts/02_train_fmri_to_clip.py --config configs/EEG/exp01_eeg_to_clip.yaml`; el nombre del
script es histórico, entrena la modalidad del config). Notebooks en `notebooks/fMRI/` y
`notebooks/EEG/` (`00..06`) = versiones visuales; `notebooks/30_multimodal_comparison.ipynb`
compara ambas líneas (importan de `src/`).

## Modalidad EEG (THINGS-EEG2)

Ampliación aditiva: la línea fMRI **no cambia de lógica**; EEG entra por un switch de modalidad.

- **Dataset**: `C:/Users/xxdia/Documents/Datasets/THINGS-EEG2` con `image_set/` (imágenes +
  `image_metadata.npy`) y `preprocessed_data/<sub>/preprocessed_eeg_{training,test}.npy` (17
  canales) o `<sub>__63_channels/` (63; se descarta el canal `stim`). Cada `.npy` es un dict con
  `preprocessed_eeg_data` de forma `[imágenes, repeticiones, canales, tiempos]`: train
  `(16540, 4, C, 100)`, test `(200, 80, C, 100)`; ventana −200→790 ms a 100 Hz. Se lee con numpy
  (sin MNE). Sujetos extraídos: `sub-01`, `sub-08`.
- **Config**: `dataset.modality: eeg` en `configs/EEG/base.yaml`. Claves propias:
  `dataset.channels` (17|63), `dataset.time_window_ms` (null|[a,b]), `dataset.trial_aggregation`
  (`{train: none, val: mean, test: mean}`), `dataset.subject_selection` (`sub-01`|lista|`all`).
  `model.encoder_type: eeg_temporalconv` + bloque `model.eeg_encoder` (in_channels/in_times se
  toman del datamodule). `training.num_workers: 0` (arrays en RAM; workers los duplicarían).
- **Encoder**: `src/models/eeg_encoder.py::EEGEncoderTemporalConv` (`[B,C,T]→[B,2048]`, conv
  temporal + separable + bloques residuales + pooling temporal; robusto a T). Mantiene el
  contrato `output_dim` → las cabezas CLIP/LowLevel y el `TokenAdapter` se reutilizan igual.
- **Datos**: `EegSubjectData`/`EegDataset`/`EegDataModule` (`src/data/eeg_*`), normalización
  **por canal** (`EegNormalizer`), y `build_datamodule(cfg)` (`src/data/factory.py`) que elige
  fMRI/EEG. `subject_split_frame` devuelve **una fila por imagen única** (precompute +
  `load_subject_matrices` alinean por identidad); `get_frame` expande a trials según
  `trial_aggregation`. `feat_idx` = rango de la imagen en `(sujeto, split)`.
- **Agregación de repeticiones**: entrenamiento **por-trial** (train=none, aumento de datos);
  **eval/ablación siempre a nivel de imagen** (media sobre repeticiones → candidatos únicos y
  alto SNR, estándar en THINGS-EEG2). Esto es una mejora metodológica deliberada.
- **Multi-sujeto EEG**: soportado a nivel de datos (normalización por sujeto). Se recomienda
  empezar single-subject; los adaptadores por sujeto (flat Linear) **no** se usan en EEG.
- **Reutilización**: CLIP/VAE/PCA, pérdidas, retrieval, ablación, generación (adapter → SD-1.5),
  checkpointing y scripts 00–08 son los mismos. Los targets CLIP/VAE se calculan por imagen
  única de THINGS. Rutas de features no colisionan con fMRI (sujetos `sub-01` vs `subj01`).

## Preprocesamiento propio del EEG desde raw (`dataset.source: raw`)

Además del preprocesado **oficial** (`source: preprocessed`, 17|63 canales — sigue siendo el
**default**, nada cambia), se puede partir del **raw de 63 canales** y aplicar un pipeline propio
parametrizable + ablaciones. Ver `docs/05_preprocesamiento_eeg_raw_y_ablaciones.md`.

- **Raw**: `raw-eeg/<sub>/ses-0{1..4}/raw_eeg_{training,test}.npy` = dict con `raw_eeg_data`
  `(64, N)` (**63 EEG + canal `stim`**), `sfreq=1000`, hardware ya filtrado 0.01–100 Hz. ~15 GB
  por sujeto. Hoy solo está descargado `sub-08`.
- **Eventos**: canal `stim`, muestras sueltas cuyo valor es el **índice de imagen (1-based)**;
  `99999` = target/catch (se descarta). ⚠️ **Cada imagen de training aparece en exactamente 2 de
  las 4 sesiones (2 reps c/u → 4 reps)** con reparto entrelazado; el test son 20 reps/sesión → 80.
  ⇒ **las repeticiones se agrupan por código de imagen**, NUNCA concatenando sesiones por el eje
  de repeticiones. Las reps extra se recortan con selección con semilla (como el código oficial).
- **Baseline** (`configs/EEG/preproc/baseline.yaml`): 63 ch · 0.1–100 Hz · epoch −200…1000 ms ·
  baseline −200…0 · 250 Hz · crop **half-open** `[0,1000)` (= **250 muestras exactas**, no 251) ·
  sin ICA/ASR/CAR/notch · **MVNN fit solo con train y aplicado a cada repetición ANTES de
  promediar** · avg-4 en train · 80 reps guardadas en test → **63×250**.
- **10 ablaciones** en `configs/EEG/preproc/*.yaml` (overrides mínimos, un factor cada una):
  `ablate_mvnn`, `channels_17` (17 posteriores **desde el raw**, MVNN 17×17), `temporal_100_600`,
  `temporal_200_400`, `sampling_100hz`, `frequency_0_5_40` (desde raw, sin encadenar filtros),
  `train_independent_trials`, `reference_car` (CAR **antes** de MVNN), `baseline_minus100`,
  `baseline_none`.
- **Salida = mismo contrato que los ficheros oficiales** (`preprocessed_eeg_data`
  `[n_img, n_reps, C, T]`, `ch_names`, `times`) en
  `data/processed/eeg_preproc/<variante>/<sub>/` ⇒ el datamodule/encoder/experimentos **no
  cambian**. Además `metadata.json` (hash de config, filtro, MVNN, QC, versiones) y `qc/`.
- **Anti-leakage**: split **por imagen** con función compartida `src/data/eeg_split.py`, usada por
  el pipeline y por el datamodule ⇒ MVNN se ajusta justo sobre las imágenes que el train llama
  train. `dataset.normalize: false` en variantes raw (el doc prohíbe z-score sobre MVNN).
- **Coste**: ~2,1 GB por sujeto-variante; decenas de minutos por sujeto.

⚠️ **El preprocesado raw necesita el python del venv** (MNE solo está ahí; `python` a secas es el
python base del sistema y NO lo tiene). `preprocessing.filter.backend: mne` está **fijado a
propósito** en `configs/EEG/preproc/baseline.yaml`: si MNE no está, **falla en 2 s con un mensaje
accionable** en vez de degradar en silencio a un IIR de scipy (que produciría una variante NO
comparable con las demás, violando el §3.3). Para desviarte a propósito:
`--set preprocessing.filter.backend=scipy`.

```
.tfm_fmri_diffusion_3_11/Scripts/python.exe scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/baseline.yaml
python scripts/10_validate_eeg_preproc.py  --config configs/EEG/preproc/baseline.yaml   # tests §14
python scripts/02_train_fmri_to_clip.py    --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
python scripts/11_eval_test_repetitions.py --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml  # curva R (no reentrena)
```

Además, el cache está protegido por **hash de config** (§13): si editas un config de
preprocesado y reejecutas sin `--force`, el script **se niega** a reutilizar el cache viejo en
vez de mezclar parámetros en silencio.

### ⚠️ Qué reejecutar al cambiar de preprocesamiento

| Paso | ¿Reejecutar? |
|---|---|
| `09` (variante) y `00` (metadata) | **Sí** |
| `01_precompute_clip` y `04_precompute_vae_pca` | **NO** — dependen solo de las imágenes; mismo set y mismo split ⇒ mismo `feat_idx`. **Se comparten entre todas las variantes** y con la línea oficial |
| Exp1 (`02`), Exp2 (`03`), Exp3 (`05`) | **Sí** (cambia la entrada cerebral) |
| **Entrenar el TokenAdapter** (`06`) | **NO** — se entrena con `CLIP → latentes VAE`, no ve el EEG |
| **Generar imágenes** (`06`) y Exp5 (`07`) | **Sí** |

Condición: no cambiar `dataset.val_ratio`/`split_seed` ni descartar imágenes
(`repetitions.*.on_missing: fail`). Si cambia (C,T) es otro `experiment.name`, pero **no**
invalida las features.

## Mapa de `src/`

- `utils/` — infraestructura: `config` (YAML+`_base_`+overrides), `seed` (RNG), `device`
  (AMP, `make_grad_scaler`), `logging` (CSV/JSONL), `paths` (rutas centralizadas),
  `checkpointing` (estado completo, retención, resume).
- `data/` — fMRI: `subject_selection`, `algonauts_dataset` (`SubjectData` lee fMRI por `source`
  con mmap; `AlgonautsDataset`), `fmri_normalization`, `datamodule` (`FmriDataModule`: resuelve
  layout/sujetos/splits/normalización/DataLoaders + `SubjectHomogeneousBatchSampler`). EEG:
  `eeg_things_dataset` (`EegSubjectData`/`EegDataset`), `eeg_normalization` (`EegNormalizer`,
  por canal), `eeg_datamodule` (`EegDataModule`), `eeg_split` (split por imagen **compartido**
  con el preprocesado). `factory` (`build_datamodule` por modalidad).
- `preprocessing/` — pipeline propio EEG desde raw: `things_raw_loader` (layout, eventos del
  canal `stim`, QC), `filters` (backend MNE|scipy, FIR fase cero + resample antialias),
  `epoching` (epoch por bloques, baseline, crop half-open), `mvnn` (Ledoit–Wolf en NumPy,
  verificado contra sklearn), `build_variant` (orquesta y guarda la variante), `qc` (figuras).
- `features/` — `clip_model` (carga CLIP congelado), `precompute_clip_embeddings`,
  `precompute_vae_latents`, `fit_vae_pca` (PCA solo en train), `load_features`.
- `models/` — `fmri_encoder` (MLP residual), `eeg_encoder` (`EEGEncoderTemporalConv`, conv
  temporal `[B,C,T]`), `heads` (CLIP/LowLevel), `adapters` (subject + token), `multitask_decoder`
  (`build_model`/`build_model_from_checkpoint`, elige encoder por `dataset.modality`).
- `losses/` — `cosine`, `contrastive` (InfoNCE simétrico), `multitask_losses` (`build_loss`).
- `training/` — `trainer_utils` (loops, early stopping, checkpoint/resume completo),
  `train_multitask_decoder` (**núcleo** `run_training(use_lowlevel)`), `train_clip_decoder`,
  `train_lowlevel_decoder`.
- `evaluation/` — `retrieval_metrics`, `embedding_metrics`, `eval_data` (matrices por sujeto),
  `baselines` (media + ridge dual), `ablation_eval` (`evaluate_ablation`, `conclusion_...`),
  `generation_metrics`.
- `generation/` — `sd_pipeline` (`FrozenSDGenerator`, `load_sd_pipeline`, `train_token_adapter`),
  `generate_from_fmri` (`generate_images` + helpers públicos `predict_condition_embeddings`,
  `lowlevel_init_images`, `save_condition_images`), `make_grids`, `checkpoint_sweep`
  (`sweep_adapter_checkpoints`, `discover_adapter_checkpoints`).

## Salidas (`outputs/<exp>/`)

`config.yaml`, `checkpoints/{last,best,epoch_XXXX}.pt`, `logs/{train_log.csv,resume_history.jsonl}`,
`metrics/*.{json,csv}`, `figures/*.png`, `embeddings/*.npy`, `lowlevel/*.npy`,
`generated/{real,correct,permuted,zero}/*.png`, `grids/*.png`, `report/summary.md`.
Las métricas agregadas de la ablación usan formato tidy: `metric_name, condition, subject_id,
split, value, seed, checkpoint`.

## Checkpointing / resume

- Reanudar: `--resume` (=`last.pt`) o `--resume PATH`, o `checkpointing.resume: auto`. Restaura
  modelo, optimizador, scheduler, GradScaler, época, `global_step`, mejor métrica, early stopping
  y estados RNG; continúa el CSV sin perder historial y registra en `resume_history.jsonl`.
- ⚠️ **Cada checkpoint ≈ 2,7 GB** (pesos + estados AdamW). Para no llenar disco:
  `checkpointing.save_every_n_epochs: 5`, `keep_last_n: 1`. Los `epoch_XXXX.pt` se pueden borrar;
  `best.pt`/`last.pt` bastan.

## Generación (Exp4/5)

- Mecanismo por defecto (**Opción B**): embedding CLIP predicho → `TokenAdapter` → `prompt_embeds`
  de SD-1.5 congelado. **Opción C** (`generation.mode: lowlevel_img2img`): invertir el vector PCA
  → latente → img2img. `adapter_lowlevel` combina B+C.
- ⚠️ **`generation.load_text_encoder: false` por defecto**: SD carga sin el text encoder de CLIP
  (solo servía para el prompt negativo vacío) porque `transformers` 5.x lo rechaza al cargar el
  checkpoint de SD-1.5 (`RuntimeError ... ignore_mismatched_sizes`). El negativo pasa a ceros
  (coherente con prompt vacío). Verificado con el venv del usuario (genera 512×512). Si cambias
  código de generación y lo pruebas en notebook, **reinicia el kernel** (cachea el módulo viejo).
- El **entrenamiento del adapter** es lo más lento (pérdida de difusión con U-Net congelada):
  1–4 h según GPU/`adapter_epochs`. Palancas: bajar `adapter_epochs`/`num_samples` o usar Opción C.
- ⚠️ **La pérdida de entrenamiento del adapter NO predice la calidad de generación** (es una MSE
  de difusión de un solo timestep aleatorio; generar son 50 pasos + CFG). No elijas el checkpoint
  por `best_loss`; usa `scripts/08_sweep_adapter_checkpoints.py` / `notebooks/06_...` (genera con
  varios checkpoints y puntúa por similitud CLIP). El adapter guarda snapshots periódicos
  `epoch_XXXX.pt` (`generation.adapter_save_every_n_epochs`/`adapter_keep_last_n`).
- El adapter usa `generation.adapter_scheduler` (cosine + `adapter_warmup_ratio`) — igual que
  Exp1/Exp3, no LR plana. Dos palancas OPCIONALES (off por defecto) para mitigar lo del proxy de
  pérdida, combinables e independientes (ver `docs/03_...md` §6):
  - `generation.adapter_timesteps_per_sample` (>1): promedia la pérdida sobre N timesteps/muestra
    → menos ruidosa (cuesta N forwards extra de U-Net/batch).
  - `generation.adapter_eval_enabled: true`: cada `adapter_eval_every_n_epochs` genera unas
    imágenes held-out y elige `adapter_best.pt` por **similitud CLIP** (no por pérdida); reutiliza
    la U-Net de entrenamiento (sin segunda copia), fuerza `mode='adapter'`, registra
    `val_clip_sim`/`best_val_sim`. `adapter_select_by: auto|loss|clip_sim`.
- `train_token_adapter` y `generate_images` guardan `config.yaml` (config resuelta completa),
  además del `metadata/generation_params.json` (resumen acotado de esa generación).
- ⚠️ **Cambiar `generation.sd_model` NO basta**: SD-2.1 tiene `cross_attention_dim=1024` (768 en
  1.5) → reentrenar el adapter desde cero; y en modos low-level, `features.vae_model` debe apuntar
  al MISMO modelo y hay que rehacer `04`+`05` (el espacio latente del VAE cambia). El `.pkl` de la
  PCA está namespaced por `vae_model` (no se pisan entre modelos). El CLIP de Exp1 (`ViT-L-14`) es
  independiente del backbone de difusión: no se toca. Configs de ejemplo: `configs/exp0X_..._sd21.yaml`.

## Interpretación de métricas (importante, evita conclusiones falsas)

- **El retrieval es la métrica honesta** (Top-1/5/10, mean/median rank). Azar subj01: test
  (159 candidatos) Top-1≈0,63%, Top-5≈3,1%; val (984) Top-1≈0,10%. `val` es más difícil que
  `test` por tener más candidatos: no los compares entre sí, compara cada uno con su azar.
- **`mean_cosine` engaña**: los embeddings CLIP son anisótropos (el baseline de media da coseno
  ~0,75 y aun así retrieval a nivel de azar). No concluyas por coseno absoluto.
- **R² del CLIPHead es negativo por diseño** (se optimiza dirección/coseno, no magnitud L2). El
  R² **sí** es informativo para la rama **low-level** (targets PCA centrados con MSE): R²>0 = bueno.
- Compara siempre contra el baseline **ridge** (fuerte en fMRI) y **media** (≈ azar).
- **EEG (image-level)**: azar test (200 candidatos) Top-1≈0,5%, Top-5≈2,5%, Top-10≈5%; val
  (~1 654 candidatos) Top-1≈0,06%. El EEG suele decodificar por encima del azar en retrieval,
  pero puede **no** guiar la generación con claridad — separa siempre la conclusión de
  decodificación (Exp2) de la de generación (Exp5); reporta negativos si las ablaciones no separan.

## Entorno

- **venv del proyecto (stack completo GPU)**: `.tfm_fmri_diffusion_3_11\Scripts\python.exe`
  (torch cu130 + CUDA, diffusers 0.37.1, transformers 5.6.2 — versiones muy nuevas/bleeding-edge;
  **`mne` 1.12.1** para el preprocesado EEG desde raw). Úsalo para cualquier prueba real que
  necesite diffusers/open_clip/SD/MNE.
- ⚠️ **`sklearn` puede fallar de forma intermitente** con
  `ImportError: DLL load failed while importing arrayfuncs: Una directiva de Control de
  aplicaciones bloqueó este archivo` (política WDAC del equipo; se observó en
  `sklearn.covariance`/`linear_model` y luego funcionó). Por eso el MVNN implementa
  Ledoit–Wolf **en NumPy puro** (validado contra sklearn a ~1e-17) y no depende de él.
- El **python base** del sistema (3.11.9) tiene solo numpy/pandas/sklearn/pillow + torch CPU
  (instalados para validación ligera): sirve para lógica sin GPU, no para generación real.
- **pip falla por SSL corporativo**: usa `--trusted-host pypi.org --trusted-host
  files.pythonhosted.org --trusted-host pypi.python.org` (+ `--trusted-host download.pytorch.org`
  para torch). Ver memoria `pip-ssl-trusted-host`.
- Datasets: fMRI en `C:/Users/xxdia/Documents/Datasets/NSD_Algonauts_2023` (en `configs/fMRI/base.yaml`),
  EEG en `C:/Users/xxdia/Documents/Datasets/THINGS-EEG2` (en `configs/EEG/base.yaml`).

## Cómo validar cambios

- Sintaxis de todo: `python -m compileall -q src scripts`.
- Lógica sin GPU (fMRI): patrón de smoke test con dataset sintético (crear `train_data`/`test_data`
  falsos + features `.npy` alineadas y ejercitar `FmriDataModule`, retrieval, ridge, ablación,
  train→resume con torch CPU). No requiere el dataset real ni GPU.
- Lógica sin GPU (EEG): con `sub-01` real + features CLIP/PCA **fabricadas** (aleatorias)
  alineadas por `feat_idx`, ejercitar `build_datamodule`, `build_model` (encoder EEG), un par de
  pasos de entrenamiento, `evaluate_ablation` **y `evaluate_baselines`** en CPU (con targets
  aleatorios el veredicto debe salir `fmri_not_clearly_used`). Ojo: el brain-tensor EEG es 3-D
  `[N,C,T]`; `RidgeRegression`/`evaluate_baselines` lo aplanan a `[N,C·T]` (ridge primal si
  features≤muestras, como en EEG; dual si features≫muestras, como en fMRI).
- Generación / diffusers reales: ejecuta con el **python del venv** (tiene SD en caché).
- Estado actual de resultados (subj01) e interpretación: `docs/02_...md` §8 (Exp1/2/3 dieron
  **positivo claro**: correcto Top-5 87% ≫ controles ~3%; el MLP supera a ridge; low-level R²>0).

## Qué NO hacer (spec §20)

- No entrenar ni fine-tunear Stable Diffusion / CLIP / VAE (van congelados).
- No usar captions detallados como condición principal (taparían el aporte fMRI; prompt vacío).
- No mezclar train/val/test al ajustar PCA/normalizador/baselines.
- No portar la arquitectura EEG (convoluciones temporales) como modelo principal de fMRI.
- No atribuir imágenes generadas a la señal cerebral si la ablación (Exp2) no da
  `correcto ≫ permutado ≈ cero`.
- No cargar el NSD raw completo (se usa Algonauts 2023 preprocesado).
