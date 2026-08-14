# Documento 7 — El decoder cerebro → CLIP: Experimento 1 (`02`) y Experimento 3 (`05`)

> Documento de referencia del **modelo que sí se entrena de verdad** en este proyecto: el que
> traduce la señal cerebral (EEG o fMRI) a una representación visual. Es el corazón del TFM,
> porque es donde vive la afirmación falsable `correcto ≫ permutado ≈ cero`.
>
> Complementa al Documento 6 (TokenAdapter y generación): **aquí se produce el embedding CLIP
> predicho; allí se usa** para generar imágenes.

**La idea clave, por delante:** `02_train_fmri_to_clip.py` y `05_train_multitask.py` **entrenan
el mismo modelo con el mismo bucle**. Literalmente la misma función:

```python
# src/training/train_clip_decoder.py      (Experimento 1, script 02)
def train_clip(cfg, resume=None):
    return run_training(cfg, use_lowlevel=False, resume=resume)

# src/training/train_multitask_decoder.py (Experimento 3, script 05)
def train_multitask(cfg, resume=None):
    return run_training(cfg, use_lowlevel=True, resume=resume)
```

Toda la diferencia entre Exp1 y Exp3 cabe en ese booleano: **Exp3 añade una segunda cabeza y un
tercer término a la pérdida.** El resto —encoder, optimizador, validación, checkpoints, criterio
de «mejor modelo»— es idéntico.

---

## 1. Qué se predice y por qué

El objetivo es un **embedding CLIP de imagen**: un vector de 768 dimensiones (modelo `ViT-L-14`)
que resume el **contenido semántico** de la imagen que el sujeto estaba viendo («qué hay en la
escena»).

```
imagen estímulo ──CLIP congelado──► clip_target [768]     ← lo que el modelo debe predecir
señal cerebral  ──── decoder ─────► clip_pred   [768]
```

Tres decisiones de diseño importantes:

1. **CLIP va congelado y se precomputa** (paso `01_precompute_clip.py`). El entrenamiento del
   decoder no carga CLIP ni Stable Diffusion: solo lee `.npy` de disco. Por eso cabe en memoria
   y es rápido.
2. **Alineación por `feat_idx`**: la fila `k` del array de features corresponde al `feat_idx`
   `k` dentro de `(sujeto, split)`. El dataset lee `arr[feat_idx]`. Romper esa correspondencia
   rompería silenciosamente todo el proyecto.
3. **Los embeddings se guardan sin normalizar**, y las cabezas también emiten sin normalizar. La
   L2-normalización se aplica **dentro de las pérdidas y del retrieval**, para tener una única
   convención en todo el código.

> ¿Por qué predecir CLIP y no la imagen directamente? Porque un vector de 768 dimensiones
> semánticamente estructurado es un objetivo **medible sin generar nada**: basta con comprobar si
> la predicción se parece más a *su* imagen que a las demás (retrieval). Esto separa la pregunta
> «¿hay información cerebral?» de la pregunta «¿sabemos dibujar?».

---

## 2. Arquitectura

El modelo es un **tronco compartido** (encoder, dependiente de la modalidad) más **una o dos
cabezas** (comunes a ambas modalidades):

```
señal cerebral ──► ENCODER ──► h [B, 2048] ──┬──► CLIPHead     ──► clip_pred [768]   (Exp1 y Exp3)
                (por modalidad)               └──► LowLevelHead ──► low_pred  [512]   (solo Exp3)
```

Todo el código está en [`src/models/multitask_decoder.py`](../src/models/multitask_decoder.py):

```python
def forward(self, fmri, subject=None) -> dict:
    h = self.encode(fmri, subject)
    out = {"h": h, "clip": self.clip_head(h)}
    if self.low_head is not None:
        out["low"] = self.low_head(h)
    return out
```

### 2.1. El tronco: un encoder por modalidad

Aquí es donde EEG y fMRI se separan, porque **la naturaleza de la señal es distinta**. El
contrato que ambos respetan es el mismo: exponen `output_dim` y devuelven `[B, output_dim]`, de
modo que las cabezas se enchufan igual.

#### fMRI — MLP residual ([`fmri_encoder.py`](../src/models/fmri_encoder.py))

La fMRI de Algonauts **no es una serie temporal**: es una **respuesta espacial por imagen**, un
vector plano de vértices (39 548 en `subj01`, LH+RH concatenados).

```
fMRI [B, V]
 → LayerNorm(V)
 → VoxelDropout(0.1)        # tira vóxeles enteros al azar
 → Linear(V, 4096) + GELU + Dropout
 → ResidualMLPBlock(4096) ×2
 → Linear(4096, 2048)  →  h
```

#### EEG — convolución temporal ([`eeg_encoder.py`](../src/models/eeg_encoder.py))

El EEG **sí** es una señal temporal multicanal `[C, T]`, así que se usa una red convolucional
temporal (spec §7.2):

```
EEG [B, C, T]
 → BatchNorm1d(C) + ChannelDropout   # tira canales enteros al azar
 → Conv1d temporal (kernel 7) + GroupNorm + GELU     ← "stem", mezcla canales
 → Conv separable/depthwise temporal
 → ResidualTemporalBlock ×2
 → pooling temporal por atención     ← robusto al valor de T
 → LayerNorm + Linear(hidden, 2048)  →  h
```

El *pooling* temporal es lo que hace al encoder **robusto a `T`**: la misma arquitectura acepta
63×250, 63×125, 63×50 o 63×100 sin cambios, que es justo lo que exigen las ablaciones de ventana
temporal y de frecuencia de muestreo (Documento 5).

> ⚠️ **Prohibido cruzarlos** (spec §20): no se debe usar la conv temporal del EEG como modelo
> principal de fMRI, ni la MLP plana para EEG (destruiría la estructura temporal).

#### La consecuencia práctica: dos regímenes de sobreajuste muy distintos

| Encoder | Parámetros | Muestras de train | Régimen |
|---|---:|---:|---|
| fMRI (`V=39 548`, hidden 4096) | **237,6 M** | ~8 900 (subj01) | sobreajusta **enseguida** (el mejor `val` suele caer en la época 3–5) |
| EEG (`63×250`, hidden 512) | **6,8 M** | ~14 900 imágenes (o 4× si trials sueltos) | mucho más holgado |

De ahí que la línea fMRI dependa tanto de la regularización (`voxel_dropout`, `weight_decay`,
early stopping) y que ese contraste sea, en sí mismo, un resultado que comentar en la memoria.

### 2.2. Adaptadores por sujeto (solo fMRI multi-sujeto)

Cada sujeto de fMRI tiene un número **distinto** de vértices, así que no se pueden apilar en un
mismo tensor. Cuando hay más de uno, `build_model` activa automáticamente `SubjectAdapters`: un
`Linear(V_sujeto → common_dim)` por sujeto que proyecta a una dimensión común antes del tronco,
y el `SubjectHomogeneousBatchSampler` garantiza lotes de un solo sujeto.

En **EEG no se usan**: todos los sujetos comparten `(C, T)`, y las diferencias entre sujetos se
tratan con la normalización por canal (o con MVNN en las variantes raw).

### 2.3. Las cabezas ([`heads.py`](../src/models/heads.py))

Ambas son la misma clase `ProjectionHead` con distinto nombre y dimensión de salida: por defecto
un único `Linear(2048 → out_dim)` (opcionalmente con capa oculta, dropout y LayerNorm final).

| Cabeza | Salida | Objetivo | Parámetros |
|---|---|---|---|
| `CLIPHead` | `[B, 768]` | embedding CLIP (semántico) | 1,57 M |
| `LowLevelHead` | `[B, 512]` | vector PCA del latente VAE (bajo nivel) | 1,05 M |

Son deliberadamente **pequeñas**: el trabajo pesado lo hace el tronco, y así ambas tareas
comparten representación en lugar de resolverse por separado.

---

## 3. La pérdida

### 3.1. Los dos términos de CLIP (Exp1 y Exp3)

```python
l_cos = cosine_similarity_loss(clip_pred, clip_tgt)          # 1 - cos
l_con = info_nce_loss(clip_pred, clip_tgt, temperature)      # InfoNCE simétrica
total = lambda_cosine * l_cos + lambda_contrastive * l_con
```

**Coseno (`1 − cos`)** empuja la predicción a apuntar en la **dirección** del target. Nótese que
*no* penaliza la magnitud: es una decisión consciente (lo que importa de un embedding CLIP es su
dirección), pero tiene una consecuencia que reaparece más adelante — **la norma del vector
predicho no queda calibrada** (medido: 0,54×–1,39× la norma real según el run). Es irrelevante
para el retrieval (invariante a escala) y **solo importa en generación**, donde ya está resuelto
en el Documento 6 §2.3 bis/ter.

**InfoNCE (contrastiva)** es la que realmente produce buen retrieval:

```python
logits = (pred_n @ tgt_n.t()) / temperature      # matriz [B, B] de similitudes
labels = arange(B)                                # el emparejamiento correcto es la diagonal
loss   = 0.5 * (cross_entropy(logits, labels) + cross_entropy(logits.t(), labels))
```

En palabras: dentro del lote se construye la matriz de todos-contra-todos y se pide que **cada
predicción se parezca a *su* imagen y se diferencie de las demás**. Es simétrica (cerebro→imagen
e imagen→cerebro), como en CLIP original. La `temperature` (0,07) controla cuán exigente es la
separación.

> **Por qué importa la distinción**: el coseno solo dice «acércate»; la InfoNCE dice «acércate a
> la tuya *y aléjate de las otras*». Sin el término contrastivo el modelo puede minimizar la
> pérdida prediciendo algo genérico y parecido para todas las imágenes — exactamente el fallo que
> el baseline de la media exhibe (coseno alto, retrieval a nivel de azar).

`lambda_nmse` (MSE entre embeddings normalizados) existe pero está a **0** por defecto.

### 3.2. El término de bajo nivel (**solo Exp3**)

```python
if self.use_lowlevel and outputs.get("low") is not None and targets.get("low") is not None:
    l_low = F.mse_loss(outputs["low"], targets["low"])
    total = total + self.lambda_lowlevel * l_low
```

Un MSE sencillo contra el vector PCA del latente VAE (por defecto `lambda_lowlevel = 0.25`).
Obsérvese la condición: **si no hay cabeza low-level o no hay targets, el término simplemente no
existe** — de ahí que la misma clase de pérdida sirva para Exp1 y Exp3.

### 3.3. La pérdida total

```
L = λ_cosine · (1 − cos)  +  λ_contrastive · InfoNCE  [+ λ_lowlevel · MSE_lowlevel]
                └──────────── Exp1 y Exp3 ───────────┘ └────── solo Exp3 ──────┘
```

---

## 4. El bucle de entrenamiento (idéntico en ambos experimentos)

`run_training` ([`train_multitask_decoder.py`](../src/training/train_multitask_decoder.py)):

1. **Semillas** (`project.seed`) y dispositivo; se guarda la config resuelta en
   `outputs/<exp>/config.yaml`.
2. **Datamodule por modalidad** vía `build_datamodule(cfg)` (fMRI o EEG, según `dataset.modality`).
3. **Dimensiones automáticas**: `clip_dim` (y `low_dim` si procede) se leen de las features
   precomputadas con `peek_feature_dim`. Si faltan, falla con un mensaje que dice qué script
   ejecutar. Los `kinds` que carga el DataLoader dependen del experimento:
   `("clip",)` en Exp1, `("clip", "low")` en Exp3.
4. **Optimización**: `AdamW` (`lr` 1e-4, `weight_decay` 0.01) + **scheduler coseno con warmup
   por paso** (`warmup_ratio` 0.02) + AMP en CUDA + recorte de gradiente (`grad_clip` 1.0).
5. **Un lote** consume del batch: `batch["fmri"]` (el tensor cerebral, sea `[B,V]` o `[B,C,T]`),
   `batch["clip_target"]` y, en Exp3, `batch["low_target"]`.
6. **Validación por época** (§4.1).
7. **Early stopping + checkpoints** según `checkpointing.monitor` (por defecto `val_top5`).
8. **Finalización**: se recarga `best.pt` y se evalúa en `val` y `test`, guardando métricas,
   embeddings y figuras.

### 4.1. La validación: retrieval contra el banco completo

Esto es lo que distingue a este proyecto de un simple regresor. En cada época,
[`validate`](../src/training/trainer_utils.py) recorre el split de validación, acumula **todas**
las predicciones y targets, y calcula retrieval **por sujeto** (macro-promediando después):

```python
for s in np.unique(subs):
    m, _ = compute_retrieval_metrics(preds[mask], targets[mask], ks=ks)
```

Es decir: el banco de candidatos son **todas las imágenes de validación de ese sujeto**, no solo
las del lote. Por eso `val` es *más difícil* que `test` (más candidatos) y sus Top-k se ven más
bajos aunque el modelo sea el mismo. **Compara cada split con su propio azar, nunca entre sí.**

Métricas registradas: `val_loss`, `val_top1/5/10`, `val_mean_rank`, `val_median_rank`,
`val_mean_cosine`.

### 4.2. Selección del «mejor» modelo

`checkpointing.monitor: val_top5` (modo `max`). Es una decisión metodológica: **el mejor modelo
se elige por retrieval, no por la pérdida**, porque la pérdida puede bajar mientras la capacidad
discriminativa se degrada. Y sigue siendo `val_top5` **también en Exp3**, para que la rama de
bajo nivel no pueda «secuestrar» la selección a costa de la semántica.

### 4.3. Reanudación

Cada checkpoint guarda el estado **completo** (modelo, optimizador, scheduler, GradScaler, época,
`global_step`, mejor métrica, contador de early stopping, estados RNG, config y versiones de
librerías). Se reanuda con `--resume` (= `last.pt`) o `--resume PATH`, continuando el
`train_log.csv` sin perder historial y anotando en `resume_history.jsonl`.

---

## 5. Experimento 1 — `02_train_fmri_to_clip.py`

**Pregunta:** ¿la señal cerebral contiene información suficiente para aproximar el embedding CLIP
de la imagen vista?

- **Modelo**: encoder + **solo** `CLIPHead`.
- **Pérdida**: coseno + InfoNCE.
- **Requisitos previos**: `00_prepare_dataset` y `01_precompute_clip`. **No carga Stable
  Diffusion ni el VAE.**
- **Config**: `model.use_lowlevel: false`.

```bash
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/02_train_fmri_to_clip.py --config configs/EEG/exp01_63_eeg_to_clip.yaml
```

El nombre del script es histórico: **entrena la modalidad que diga el `--config`**.

---

## 6. Experimento 3 — `05_train_multitask.py`

**Pregunta:** ¿puede la señal cerebral predecir **además** información visual de **bajo nivel**
(estructura, composición, color), y hacerlo **sin estropear** la semántica?

### 6.1. El objetivo de bajo nivel

Se construye en el paso `04_precompute_vae_pca.py`:

```
imagen → VAE congelado → latente [4,64,64] → aplanar (16 384) → PCA (ajustada SOLO en train) → [512]
```

La PCA se ajusta **solo con train** (anti-leakage) y captura ~58 % de la varianza del latente con
512 componentes.

### 6.2. Qué cambia respecto a Exp1

Solo tres cosas:

```yaml
model:
  use_lowlevel: true
  lowlevel_head: { output_dim: 512 }   # debe coincidir con features.pca_dim
losses:
  lambda_lowlevel: 0.25
```

Y en el código, `use_lowlevel=True` hace que: (a) se construya la `LowLevelHead`, (b) el
DataLoader cargue también `low_target`, (c) la pérdida sume el MSE de bajo nivel.

### 6.3. Un solo tronco, dos cabezas, **un solo `backward`**

No hay entrenamiento alterno ni dos optimizadores: se suma un único escalar y se retropropaga una
vez. Pero **cada parámetro recibe gradiente solo de los términos de los que depende**, por la
topología del grafo:

| Parámetros | Reciben gradiente de… | Por qué |
|---|---|---|
| `clip_head` | solo coseno + InfoNCE | `low_pred` no depende de `clip_head` |
| `low_head` | solo MSE de bajo nivel | los términos CLIP no dependen de `low_head` |
| **encoder (tronco)** | **de los tres términos** | `h` alimenta a **ambas** cabezas |

Las cabezas están desacopladas entre sí; **el tronco recibe una señal combinada**. Los `λ`
regulan esa mezcla. Ese acoplamiento *a través del tronco compartido* es el único mecanismo por
el que una tarea puede influir en la otra.

### 6.4. Por qué añadir la rama de bajo nivel puede **mejorar** también el CLIP

1. **Regularización por tarea auxiliar (la razón principal en fMRI).** Un tronco de 237 M de
   parámetros frente a ~8 900 muestras memoriza atajos que no generalizan. Exigirle **además**
   predecir el bajo nivel invalida esos atajos y empuja a `h` hacia información visual más
   general.
2. **Fuerza estadística compartida**: ambos objetivos vienen de la misma imagen.
3. **Gradiente denso y bien condicionado**: el MSE sobre vectores PCA centrados complementa a la
   señal más direccional de CLIP.

> **Matiz honesto para la memoria** (resultados reales de subj01): la mejora fue **pequeña**
> (test Top-5 86,8 % → 89,3 %; Top-1 47,2 % → 52,2 %). Con solo 159 muestras de test, cada imagen
> vale ~0,63 puntos, así que ese +5 en Top-1 son ~8 imágenes: es consistente con «no perjudica y
> probablemente ayuda un poco», pero con margen de ruido. Lo que **sí** se puede afirmar con
> solidez es lo esencial de Exp3: **la rama de bajo nivel no rompe la semántica** (R² low-level
> > 0 y retrieval CLIP igual o mejor).

### 6.5. Salidas adicionales de Exp3

Además de todo lo de Exp1:

- `metrics/{val,test}_lowlevel_metrics.json`
- `lowlevel/{val,test}_low_{pred,target}.npy`
- `figures/pca_component_correlation_{val,test}.png` (Pearson por componente PCA)

---

## 7. Exp1 vs Exp3, en una tabla

| | **Exp1** (`02`) | **Exp3** (`05`) |
|---|---|---|
| Función que se ejecuta | `run_training(use_lowlevel=False)` | `run_training(use_lowlevel=True)` |
| Encoder | idéntico | idéntico |
| Cabezas | `CLIPHead` | `CLIPHead` + `LowLevelHead` |
| Pérdida | coseno + InfoNCE | coseno + InfoNCE + λ·MSE |
| Features necesarias | CLIP | CLIP **y** VAE-PCA (paso `04`) |
| Criterio de `best.pt` | `val_top5` | `val_top5` (igual) |
| Pregunta que responde | ¿hay semántica decodificable? | ¿hay además bajo nivel, sin perder semántica? |
| Habilita en generación | modo `adapter` | también `lowlevel_img2img` y `adapter_lowlevel` |

En la práctica **Exp3 es el modelo que se usa para generar**, porque su checkpoint tiene las dos
cabezas y sirve para los tres modos.

---

## 8. Cómo se interpretan las métricas (evita conclusiones falsas)

- **El retrieval es la métrica honesta.** Azar con `N` candidatos: Top-k ≈ `k/N`.
  - fMRI subj01 test (159 candidatos): Top-1 ≈ 0,63 %, Top-5 ≈ 3,1 %.
  - EEG test (200 candidatos): Top-1 ≈ 0,5 %, Top-5 ≈ 2,5 %.
- ⚠️ **`mean_cosine` engaña.** Los embeddings CLIP son **anisótropos** (todos apuntan a una región
  común), así que el baseline de la **media** alcanza coseno ~0,75 **y aun así rinde a nivel de
  azar** en retrieval. Es la demostración perfecta de por qué no hay que fiarse del coseno.
- ⚠️ **El R² del `CLIPHead` es negativo por diseño**: se optimiza dirección, no magnitud L2. Para
  el CLIPHead mira **retrieval**. El R² **sí** es informativo para la rama **low-level** (targets
  PCA centrados, entrenados con MSE): ahí **R² > 0 = bueno**.
- **Baselines obligatorios**: la **media** (≈ azar) y **ridge** (fMRI → CLIP lineal). Ridge es
  fuerte en fMRI; si el MLP no lo supera, no aporta gran cosa. En subj01 el MLP lo supera con
  claridad (Top-5 86,8 % vs 66,7 %).

### 8.1. La prueba decisiva: el Experimento 2

El decoder entrenado aquí se somete en `03_eval_retrieval_ablation.py` a las condiciones
`correcto / permutado / cero / ruido`, construidas **sobre la señal cerebral** antes del encoder
(el permutado usa un *derangement* de Sattolo, así que ninguna muestra recibe su propia señal).
El veredicto lo escribe automáticamente `conclusion_from_summary`:

```
correcto ≫ permutado ≈ cero   →   fmri_used
en caso contrario             →   fmri_not_clearly_used
```

**Sin este resultado, nada de lo que venga después (generación) puede atribuirse al cerebro.**

---

## 9. Qué se guarda

```
outputs/<experimento>/
  config.yaml                       # config resuelta completa
  checkpoints/{best,last,epoch_XXXX}.pt
  logs/train_log.csv                # una fila por época (ver CSV_FIELDS)
  logs/resume_history.jsonl
  metrics/{val,test}_metrics.json   # bloques 'retrieval' y 'embedding'
  embeddings/{val,test}_clip_{pred,target}.npy
  figures/{loss_curve.png, retrieval_topk.png}
  report/summary.md
  # solo Exp3:
  metrics/{val,test}_lowlevel_metrics.json
  lowlevel/{val,test}_low_{pred,target}.npy
  figures/pca_component_correlation_{val,test}.png
```

Columnas del `train_log.csv`: `epoch, global_step, lr, train_total, train_clip_cosine,
train_clip_infonce, train_lowlevel_mse, val_loss, val_top1, val_top5, val_top10, val_mean_rank,
val_mean_cosine, best_metric, is_best, seconds`.

⚠️ Cada checkpoint pesa ~2,7 GB en fMRI (pesos + estados de AdamW). Usa
`checkpointing.save_every_n_epochs: 5` y `keep_last_n: 1`; los `epoch_XXXX.pt` se pueden borrar.

---

## 10. Cómo encaja con el resto del pipeline

```
01_precompute_clip ──► clip_target
                         │
04_precompute_vae_pca ──► low_target (solo Exp3)
                         │
señal cerebral ──► [ESTE DOCUMENTO: Exp1 / Exp3] ──► clip_pred (+ low_pred)
                         │
                         ├──► Exp2: ablación correcto/permutado/cero  ← el veredicto del TFM
                         │
                         └──► Exp4/Exp5: clip_pred ─TokenAdapter─► SD congelado ─► imagen
                                                    (Documento 6)
```

Un detalle que enlaza ambos documentos: el `CLIPHead` se optimiza por **dirección**, así que la
**norma** de `clip_pred` no queda calibrada respecto a la de los embeddings reales con los que se
entrenó el TokenAdapter (medido: 0,54×–1,39× según el run).

Qué implica y qué no:

- **No afecta a Exp1 ni a Exp2.** El retrieval usa coseno, que es invariante a escala: las
  conclusiones de decodificación son inmunes a esto.
- **Sí afecta a la generación**, donde actúa como una intensidad de condicionamiento
  descontrolada y distinta en cada run. **Ya no es una limitación abierta**: está caracterizada y
  con dos mitigaciones implementadas (Documento 6 §2.3 bis y §2.3 ter) —la opción B la elimina
  **por construcción**—, aunque ambas están **OFF por defecto**.
- **No hace falta tocar el entrenamiento del decoder para arreglarlo.** Se podría supervisar la
  norma aquí (`losses.lambda_nmse > 0`), pero eso tocaría la métrica primaria (retrieval) para
  resolver un problema que vive aguas abajo; se resuelve mejor en la etapa generativa.
