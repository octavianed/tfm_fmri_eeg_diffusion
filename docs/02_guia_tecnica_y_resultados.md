# Documento 2 — Guía técnica: código, ejecución, salidas, métricas, tiempos y resultados

> Referencia práctica para entender el código, saber **qué ejecutar**, **qué sale**, **dónde**,
> **qué formato/significado** tiene, **qué valores son buenos** y **cuánto tarda**. Incluye la
> **interpretación de tus resultados reales** (subj01) en la sección 8.

Índice:
1. Mapa del repositorio
2. Cómo se relacionan los módulos
3. Configuración (`base.yaml`, `_base_`, overrides)
4. Convención clave: ejecutar desde la raíz
5. Flujo de datos y formatos (dónde vive cada cosa)
6. Paso a paso por experimento (entradas, salidas, tiempos, interpretación)
7. Métricas explicadas y valores de referencia (azar para subj01)
8. **Tus resultados actuales (subj01) e interpretación**
9. Checkpointing / reanudación / tamaño en disco
10. Tabla de tiempos aproximados
11. Errores comunes y consejos
12. Cómo escalar a más sujetos

---

## 1. Mapa del repositorio

```
tfm_fmri_diffusion/
  configs/            base.yaml + exp01..05.yaml         (configuración; ver §3)
  scripts/            00..07_*.py                          (puntos de entrada CLI)
  notebooks/          00..05_*.ipynb                       (versiones visuales; llaman a src/)
  src/
    utils/    config, seed, device, logging, paths, checkpointing   (infraestructura)
    data/     subject_selection, algonauts_dataset, fmri_normalization, datamodule
    features/ clip_model, precompute_clip_embeddings, precompute_vae_latents, fit_vae_pca, load_features
    models/   fmri_encoder, heads, adapters, multitask_decoder
    losses/   cosine, contrastive, multitask_losses
    training/ trainer_utils, train_multitask_decoder (núcleo), train_clip_decoder, train_lowlevel_decoder
    evaluation/ retrieval_metrics, embedding_metrics, eval_data, baselines, ablation_eval, generation_metrics
    generation/ sd_pipeline, generate_from_fmri, make_grids
  data/               raw/ processed/ features/ splits/    (datos y features precomputadas)
  outputs/            exp01../ exp05../ final_report/       (resultados por experimento)
  docs/               estos documentos
```

Regla de oro: **la lógica vive en `src/`**; scripts y notebooks solo la orquestan. Los
notebooks importan exactamente las mismas funciones que los scripts (no duplican lógica).

---

## 2. Cómo se relacionan los módulos

```
config (YAML) ──► src.utils.load_config ──► objeto cfg (acceso por punto: cfg.training.lr)
                                              │
      ┌───────────────────────────────────────┼───────────────────────────────┐
      ▼                                        ▼                                ▼
 src.data.FmriDataModule            src.features.precompute_*         src.models.build_model
 (resuelve sujetos, splits,         (CLIP/VAE/PCA a .npy en disco)    (fMRIEncoder+heads)
  normalización, DataLoaders)                    │                            │
      │                                          ▼                            ▼
      └────────────► src.training.run_training ◄── src.losses.build_loss ── entrena
                              │                    (cosine+InfoNCE[+MSE])
                              ├─ val: src.evaluation.retrieval_metrics
                              └─ guarda checkpoints (src.utils.checkpointing) + métricas

 src.evaluation.evaluate_ablation ◄── src.models.build_model_from_checkpoint  (Exp2)
 src.generation.FrozenSDGenerator ◄── TokenAdapter + Stable Diffusion congelado (Exp4/5)
```

Dependencias: `utils` no depende de nadie; `data`/`features`/`models`/`losses` dependen de
`utils`; `training` y `evaluation` dependen de los anteriores; `generation` depende de todo.
Import ligero: `torch`/`diffusers`/`open_clip` se importan de forma perezosa donde hacen
falta, así que importar un módulo no arrastra dependencias pesadas si no se usan.

---

## 3. Configuración

- **`configs/base.yaml`** contiene todos los valores por defecto (dataset, features, modelo,
  entrenamiento, pérdidas, checkpointing, evaluación, generación).
- Cada **`exp0X.yaml`** hace `_base_: [base.yaml]` y **solo** sobrescribe lo que cambia.
- **Overrides por CLI**: `--set clave.subclave=valor`, p. ej.
  `--set dataset.subject_selection=all --set training.batch_size=32`.

Campos que más vas a tocar:
```yaml
dataset:
  root_dir: C:/Users/xxdia/Documents/Datasets/NSD_Algonauts_2023
  subject_selection: subj01        # subj01 | [subj01, subj02] | all
  test_split: official             # official (usa test_data) | internal
training:
  batch_size: 64
  epochs: 100                      # con early stopping suele parar antes
  lr: 1.0e-4
  early_stopping_patience: 10
checkpointing:
  monitor: val_top5                # métrica que decide el "mejor" modelo
  keep_last_n: 3                   # ↓ para ahorrar disco (ver §9)
  save_every_n_epochs: 1           # ↑ (p.ej. 5) para ahorrar disco
losses:
  lambda_lowlevel: 0.25            # peso de la rama low-level (Exp3)
```

El `cfg` resuelto se guarda en `outputs/<exp>/config.yaml` (reproducibilidad).

---

## 4. Convención clave: ejecutar desde la RAÍZ del proyecto

Las rutas del proyecto son **relativas** (`data/`, `outputs/`, `configs/`). Por eso:
- **Scripts**: ejecútalos desde la raíz: `cd tfm_fmri_diffusion; python scripts/0X_*.py ...`.
- **Notebooks**: la primera celda hace `os.chdir()` a la raíz automáticamente (busca la
  carpeta que contiene `src/` y `configs/`). **Ejecuta siempre esa primera celda** (o
  Kernel → Restart & Run All). Si solo re-ejecutas una celda intermedia, el directorio de
  trabajo podría no ser la raíz y no encontraría las features (`data/features/...`).

---

## 5. Flujo de datos y formatos (dónde vive cada cosa)

### Dataset de entrada (tu descarga, layout “oficial”)
```
NSD_Algonauts_2023/
  train_data/subjNN/training_split/training_fmri/{lh,rh}_training_fmri.npy   [N_train, V_hemi]
  train_data/subjNN/training_split/training_images/train-XXXX_nsd-YYYYY.png
  test_data/subjNN/test_split/test_fmri/{lh,rh}_test_fmri.npy                [N_test, V_hemi]
  test_data/subjNN/test_split/test_images/test-XXXX_nsd-YYYYY.png
```
El código detecta este layout automáticamente. `subj01`: V=39 548 (19 004+20 544),
N_train=9 841, N_test=159.

### Artefactos que se generan (subj01)
| Fichero | Qué es | Formato |
|---|---|---|
| `data/processed/metadata_subj01.csv` | tabla maestra: sujeto, source (train/test), local_index, image_id, image_path, split, feat_idx | CSV |
| `data/processed/metadata_subj01.split.json` | firma del split (para invalidar caché si cambian ratios/seed) | JSON |
| `data/processed/normalization/subj01_fmri_norm.npz` | media/desv. por vóxel (ajustadas en train) | NPZ |
| `data/features/clip/ViT-L-14/subj01_{train,val,test}.npy` | embeddings CLIP objetivo, alineados por `feat_idx` | float32 `[N, 768]` |
| `data/features/vae/.../subj01_{split}_latents.npy` | latentes VAE aplanados | float32 `[N, 16384]` |
| `data/features/vae/.../subj01_{split}_pca.npy` | vector PCA (objetivo low-level) | float32 `[N, 512]` |
| `data/processed/pca/subj01_vae_pca_model.pkl` | modelo PCA (train) + shape + scaling | pickle |
| `data/processed/pca/subj01_vae_pca_model.evr.json` | varianza explicada por componente | JSON |

### Salidas por experimento (`outputs/<exp>/`)
```
config.yaml                       # config exacta usada
checkpoints/{last,best,epoch_XXXX}.pt
logs/train_log.csv                # una fila por época (curvas)
logs/resume_history.jsonl         # registro de reanudaciones
metrics/*.json|*.csv              # métricas (ver cada experimento)
figures/*.png                     # curvas, barras por condición, correlación PCA
embeddings/*.npy                  # predicciones/targets CLIP
lowlevel/*.npy                    # predicciones/targets low-level (Exp3)
generated/{real,correct,permuted,zero}/*.png   # imágenes (Exp4)
grids/*.png                       # rejillas comparativas (Exp5)
report/summary.md                 # resumen legible con conclusión preliminar
```

---

## 6. Paso a paso por experimento

Cada bloque: **qué ejecutar**, **entrada → salida**, **tiempo aprox.**, **cómo interpretar**.

### Paso 0 — Preparar dataset
- **Ejecutar**: `python scripts/00_prepare_dataset.py --config configs/exp01_fmri_to_clip.yaml`
- **Hace**: resuelve sujetos, crea splits reproducibles, ajusta normalización (solo train).
- **Salida**: `metadata_subj01.csv`, `normalization/subj01_fmri_norm.npz`.
- **Tiempo**: ~10 s por sujeto.

### Paso 1 — Precomputar CLIP (`01_precompute_clip.py`)
- **Entrada**: imágenes. **Salida**: `data/features/clip/ViT-L-14/subj01_{split}.npy`.
- **Tiempo**: ~2–6 min en GPU (bastante más en CPU) para ~11 000 imágenes.

### Experimento 1 — fMRI→CLIP (`02_train_fmri_to_clip.py` / notebook 01)
- **Entrada**: fMRI (normalizada) + features CLIP. **No** carga Stable Diffusion.
- **Salida**:
  - `checkpoints/{best,last,epoch_*}.pt`
  - `logs/train_log.csv` (columnas: `epoch, global_step, lr, train_total,
    train_clip_cosine, train_clip_infonce, train_lowlevel_mse, val_loss, val_top1,
    val_top5, val_top10, val_mean_rank, val_mean_cosine, best_metric, is_best, seconds`)
  - `metrics/{val,test}_metrics.json` (bloques `retrieval` y `embedding`)
  - `embeddings/{val,test}_clip_{pred,target}.npy`
  - `figures/{loss_curve.png, retrieval_topk.png}`, `report/summary.md`
- **Tiempo**: en tu equipo ~**50 s/época**; con early stopping suele parar en 10–20 épocas
  (⇒ ~10–20 min). En GPU dedicada sería bastante más rápido.
- **Bueno si**: `test` Top-k muy por encima del azar (Top-1≈0,63%, Top-5≈3,1% con 159
  candidatos) y por encima del baseline de media. **Ojo**: `val` usa 984 candidatos (más
  difícil) que `test` (159), así que el Top-k de val se ve más bajo aunque el modelo sea el
  mismo — no te asustes por esa diferencia.

### Experimento 2 — Ablación de retrieval (`03_eval_retrieval_ablation.py` / notebook 02)
- **Entrada**: `best.pt` de Exp1 + features CLIP.
- **Salida**:
  - `metrics/summary_table.csv` (formato *tidy*: `metric_name, condition, subject_id, split,
    value, seed, checkpoint`)
  - `metrics/retrieval_{correct,permuted,zero,noise}.json`
  - `metrics/baselines.json` (media y ridge)
  - `metrics/conclusion.json` (`decision`, `correct`, `best_control`, `message`)
  - `figures/{topk_by_condition.png, cosine_by_condition.png}`
- **Tiempo**: ~30–60 s (incluye ajuste del ridge, que resuelve un sistema en el espacio de
  muestras).
- **Bueno si**: `correct` **≫** `permuted ≈ zero ≈ noise`. Es el veredicto del proyecto.

### Paso 4 — Precomputar VAE + PCA (`04_precompute_vae_pca.py`)
- **Salida**: `..._latents.npy` (grande), `..._pca.npy`, `subj01_vae_pca_model.pkl` + `.evr.json`.
- **Tiempo**: ~10–25 min en GPU (el VAE a 512 px es más pesado que CLIP) + ~1 min de PCA.
- **Disco**: los latentes de train ocupan ~**550 MB** por sujeto.

### Experimento 3 — Multitarea (`05_train_multitask.py` / notebook 03)
- **Entrada**: fMRI + features CLIP + features PCA. **Requiere** haber hecho el Paso 4.
- **Salida** (además de lo de Exp1):
  - `metrics/{val,test}_lowlevel_metrics.json`
  - `lowlevel/{val,test}_low_{pred,target}.npy`
  - `figures/pca_component_correlation_{val,test}.png`
- **Tiempo**: similar a Exp1 (~50 s/época en tu equipo).
- **Bueno si**: el retrieval CLIP **se mantiene o mejora** respecto a Exp1, y la rama
  low-level da **R² > 0** y **Pearson medio > 0** (predice algo real, no ruido).

### Experimento 4 — Generación (`06_generate_images.py` / notebook 04)
- **Entrada**: decoder (best.pt de Exp3 o Exp1) + (se entrena) TokenAdapter.
- **Salida**: `generated/{real,correct,permuted,zero}/*.png`,
  `metadata/generation_params.json`, `checkpoints/adapter_{best,last}.pt`.
- **Tiempo**: ⚠️ el **entrenamiento del adapter** es lo más lento (pérdida de difusión con la
  U-Net congelada): puede tardar **1–4 h** según GPU y `generation.adapter_epochs`. La
  generación en sí de ~16 muestras × 3 condiciones × 50 pasos son unos minutos. Palancas para
  acelerar: bajar `generation.adapter_epochs`, bajar `generation.num_samples`, o usar
  `generation.mode: lowlevel_img2img` (no requiere adapter, usa la rama low-level).
- **Bueno si**: al inspeccionar, la imagen con fMRI correcta se parece más a la real que las
  de los controles (se cuantifica en Exp5).

### Experimento 5 — Comparación generativa (`07_eval_generation_ablation.py` / notebook 05)
- **Entrada**: imágenes generadas de Exp4.
- **Salida**: `metrics/summary_generation_metrics.csv`, `metrics/statistical_tests.json`,
  `grids/{comparison_grid_ablation,best_cases,median_cases,worst_cases}.png`,
  `report/exp05_summary.md`.
- **Tiempo**: minutos (codifica con CLIP las generadas/reales y arma rejillas).
- **Bueno si**: `correct` tiene mayor **similitud CLIP** con la real que `permuted`/`zero`,
  idealmente con test estadístico significativo.

---

## 7. Métricas explicadas y valores de referencia (azar para subj01)

Con `subj01`: **test = 159 candidatos**, **val = 984 candidatos**.

| Métrica | Qué mide | Azar (test) | Azar (val) | Bueno |
|---|---|---|---|---|
| Top-1 | correcto en el 1.º | 0,63% | 0,10% | ≫ azar |
| Top-5 | correcto en top-5 | 3,14% | 0,51% | ≫ azar |
| Top-10 | correcto en top-10 | 6,29% | 1,02% | ≫ azar |
| mean_rank | puesto medio (1=perfecto) | ~80 | ~492 | lo más bajo posible |
| mean_cosine | parecido de dirección | — | — | ⚠️ engañoso solo (ver abajo) |
| R² (low-level) | ¿mejor que predecir la media? | 0 | 0 | > 0 |
| Pearson (low-level) | correlación por componente | 0 | 0 | > 0 |

**Avisos importantes de interpretación**
- **`mean_cosine` puede engañar**: los embeddings CLIP son anisótropos (todos “miran” a una
  región común del espacio), así que incluso predicciones malas pueden dar coseno ~0,2–0,7.
  Por eso **el retrieval es la métrica honesta**, no el coseno absoluto.
- **R² del CLIPHead es negativo y NO es un problema**: el CLIPHead se optimiza por dirección
  (coseno/InfoNCE), no por magnitud L2; el R² sobre embeddings CLIP sin centrar sale negativo
  por construcción. Para el CLIPHead mira **retrieval**. El R² **sí** es informativo para la
  rama **low-level** (targets PCA centrados y optimizados con MSE) → ahí, R²>0 = bueno.
- **val vs test**: val tiene 6× más candidatos que test, así que su Top-k se ve más bajo. No
  compares val con test directamente; compara cada uno con su azar.

---

## 8. Tus resultados actuales (subj01) e interpretación

> Ejecutado: features (CLIP+VAE+PCA), Exp1, Exp2 (ablación + baselines), Exp3. Aquí van tus
> números reales y qué significan.

### Exp1 — fMRI→CLIP (test, 159 candidatos)
| | Top-1 | Top-5 | Top-10 | mean_rank | median_rank | mean_cosine |
|---|---|---|---|---|---|---|
| **tu resultado** | **47,2%** | **86,8%** | **95,6%** | **3,46** | **2,0** | 0,536 |
| azar | 0,63% | 3,14% | 6,29% | ~80 | ~80 | — |

**Interpretación**: excelente. La imagen correcta queda de media en el puesto ~3 sobre 159,
y en el 87% de los casos está en el top-5. La fMRI de subj01 codifica el contenido semántico
de forma claramente recuperable.

- Curva de entrenamiento: mejor `val_top5` en la **época 3** (0,615), luego el modelo empieza
  a **sobreajustar** (la pérdida de train sigue bajando pero `val_loss` sube). El **early
  stopping** paró en la época 13 y guardó como `best.pt` el de la época 3 — funcionó como
  debe. El sobreajuste es esperable: el modelo tiene ~239 M de parámetros (la primera capa es
  39 548→4 096) frente a 8 857 muestras; por eso hay dropout de vóxeles, weight decay y early
  stopping.

### Exp2 — Ablación (test) — **el resultado decisivo**
| Condición | Top-1 | Top-5 | Top-10 | mean_rank | mean_cosine |
|---|---|---|---|---|---|
| **correcto** | **47,2%** | **86,8%** | **95,6%** | **3,46** | 0,536 |
| permutado | 1,3% | 1,9% | 4,4% | 82,6 | 0,270 |
| cero | 0,6% | 3,1% | 6,3% | 80,0 | 0,016 |
| ruido | 0,6% | 2,5% | 6,9% | 84,7 | 0,145 |
| baseline media | 0,6% | 3,1% | 6,3% | 80,0 | 0,747 |
| baseline ridge | 24,5% | 66,7% | 83,6% | 7,2 | 0,179 |

**Veredicto del código**: `fmri_used` (correcto Top-5 0,868 ≫ mejor control 0,031).

**Interpretación** (esto es lo que da valor al TFM):
- `correcto` **aplasta** a permutado/cero/ruido, que se quedan **en el azar**. ⇒ la mejora
  depende de usar la fMRI **correcta**: hay decodificación cerebral real, no un artefacto.
- Fíjate en el aviso del coseno: `permutado` tiene `mean_cosine`=0,27 y el baseline de
  **media** tiene 0,747 (¡altísimo!), pero **ambos rinden a nivel de azar** en retrieval.
  Es la demostración perfecta de por qué no hay que fiarse del coseno absoluto.
- **Ridge** es un baseline fuerte (Top-5 66,7%), pero tu **MLP lo supera con claridad**
  (86,8% vs 66,7%; Top-1 47% vs 24,5%). ⇒ el encoder no lineal **aporta** sobre el lineal.

### Exp3 — Multitarea (test)
| | Top-1 | Top-5 | Top-10 | mean_rank |
|---|---|---|---|---|
| Exp1 (solo CLIP) | 47,2% | 86,8% | 95,6% | 3,46 |
| **Exp3 (multitarea)** | **52,2%** | **89,3%** | **97,5%** | **3,15** |

- Retrieval CLIP: **mejora ligeramente** al añadir la rama low-level (no la perjudica) ✔.
- Rama low-level (PCA, 512 dims): **R²=0,225 (>0)**, Pearson medio=0,052, coseno=0,435.
  El R² positivo significa que predice el bajo nivel **mejor que la media**; la fMRI también
  contiene algo de estructura/color, aunque de forma **más débil** que la semántica (normal:
  el bajo nivel es más difícil y ruidoso). Pearson pequeño pero positivo es un resultado
  honesto y presentable.
- PCA: 512 componentes capturan **58,5%** de la varianza del latente VAE (10 comps→29%,
  100→47%). Es razonable; el latente VAE tiene mucho detalle de alta frecuencia que la PCA
  truncada descarta.

### Conclusión de tus resultados
Tienes un **caso positivo y limpio** para el TFM: decodificación CLIP muy por encima del
azar y del baseline lineal, con la **ablación** confirmando que depende de la señal correcta,
y una rama low-level que aporta sin romper la semántica. Es exactamente lo que el diseño
buscaba demostrar. El siguiente paso natural es Exp4/Exp5 (generación) para la parte
cualitativa, recordando que su valor **se apoya** en que Exp2 ya salió positivo.

---

## 9. Checkpointing, reanudación y tamaño en disco

- **Reanudar** un entrenamiento interrumpido:
  `python scripts/02_train_fmri_to_clip.py --config configs/exp01_fmri_to_clip.yaml --resume`
  (o `checkpointing.resume: auto`). Restaura modelo, optimizador, scheduler, época, mejor
  métrica, contador de early stopping y estados RNG, y continúa el `train_log.csv` sin perder
  historial (se registra en `logs/resume_history.jsonl`).
- ⚠️ **Tamaño**: cada checkpoint pesa ~**2,7 GB** (pesos + estados de AdamW). Con
  `save_every_n_epochs=1` y `keep_last_n=3` puedes acumular ~5 checkpoints (~14 GB) **por
  experimento**. Recomendación: pon `checkpointing.save_every_n_epochs: 5` y
  `keep_last_n: 1`. Puedes **borrar sin miedo los `epoch_XXXX.pt`**: `best.pt` (para evaluar)
  y `last.pt` (para reanudar) son suficientes.

---

## 10. Tabla de tiempos aproximados (subj01, orientativo)

| Paso | Script | Tiempo aprox. | Nota |
|---|---|---|---|
| Preparar | 00 | ~10 s | rápido |
| CLIP | 01 | 2–6 min (GPU) | ~11k imágenes |
| Entrenar Exp1 | 02 | ~50 s/época; 10–20 min total | early stopping suele acortar |
| Ablación Exp2 | 03 | 30–60 s | incluye ridge |
| VAE+PCA | 04 | 10–25 min (GPU) + ~1 min PCA | latentes ~550 MB |
| Entrenar Exp3 | 05 | ~50 s/época; 10–20 min total | como Exp1 |
| Generación Exp4 | 06 | **1–4 h** (adapter) + minutos | ver palancas en §6 |
| Comparación Exp5 | 07 | minutos | métricas + rejillas |

Multiplica por el nº de sujetos si usas una lista o `all` (y el disco de features/checkpoints
crece en proporción).

---

## 11. Errores comunes y consejos

- **“No precomputed CLIP features found”** en un notebook → estás ejecutando con el directorio
  de trabajo mal (p. ej. `notebooks/`). Ejecuta la **primera celda** (hace `chdir` a la raíz)
  o Restart & Run All. (Ya corregido en los notebooks actuales.)
- **VRAM 16 GB**: mantén `mixed_precision: true`; si generas y vas justo, activa
  `generation.cpu_offload: true` (más lento pero cabe).
- **No te fíes del `mean_cosine`** ni del **R² del CLIPHead** (ver §7). Usa **retrieval**.
- **Sobreajuste**: es normal que la pérdida de train baje mucho; confía en `val_top5` y el
  early stopping, y compara siempre con el baseline **ridge**.
- **El adapter (Exp4) es lo caro**: si solo quieres cerrar el pipeline rápido, usa
  `generation.mode: lowlevel_img2img` o baja `adapter_epochs`/`num_samples`.
- **`RuntimeError: You set ignore_mismatched_sizes to False` al cargar Stable Diffusion**:
  incompatibilidad entre el cargador estricto de `transformers` 5.x y el text encoder de
  SD-1.5. En este proyecto el text encoder **solo** sirve para el prompt negativo vacío, así
  que por defecto **no se carga** (`generation.load_text_encoder: false`) y el negativo pasa
  a ser un embedding de ceros (coherente con el diseño de prompt vacío). Si tras cambiar el
  código sigues viéndolo en un notebook, **reinicia el kernel** (Jupyter cachea el módulo
  antiguo). Si algún día tienes versiones compatibles y quieres el negativo exacto de "",
  pon `generation.load_text_encoder: true`.

---

## 12. Cómo escalar a más sujetos

- Un sujeto: `--set dataset.subject_selection=subj01`.
- Lista: `--set dataset.subject_selection='[subj01, subj02]'`.
- Todos: `--set dataset.subject_selection=all`.

Con varios sujetos, el modelo activa automáticamente **adaptadores por sujeto** (cada uno con
distinto nº de vértices) y un *sampler* que hace lotes homogéneos por sujeto. Recuerda repetir
los pasos 0/1/4 (preparación y features) para todos los sujetos, y que los tiempos y el disco
escalan con el número de sujetos. El orden recomendado sigue siendo:
`subj01 → subj01+subj02 → lista → all`.
