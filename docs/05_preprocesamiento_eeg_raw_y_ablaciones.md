# Documento 5 — Preprocesamiento propio del EEG (raw) y ablaciones de preprocesamiento

> Amplía el Documento 4 (línea EEG). Hasta ahora la línea EEG consumía los **derivados
> oficiales** de THINGS-EEG2 (17 canales @100 Hz, o 63 canales). Este documento describe el
> **pipeline propio desde el EEG raw de 63 canales**, su configuración de referencia (*baseline*),
> las **10 ablaciones de preprocesamiento**, y —muy importante— **qué hay que reejecutar** de los
> experimentos 1–5 al cambiar de preprocesamiento.

---

## 1. Qué se puede elegir ahora

Dos ejes independientes, ambos por configuración:

| Eje | Clave | Valores |
|---|---|---|
| Fuente de los datos | `dataset.source` | `preprocessed` (oficial, **por defecto**) · `raw` (pipeline propio) |
| Montaje oficial | `dataset.channels` | `17` · `63` (solo si `source: preprocessed`) |
| Variante propia | `dataset.preproc_variant` | `baseline` + 10 ablaciones (solo si `source: raw`) |

**Nada cambia por defecto**: sin tocar nada, la línea EEG sigue leyendo el preprocesado oficial
y la línea fMRI no se ve afectada en absoluto.

---

## 2. Los datos raw (estructura verificada)

```
THINGS-EEG2/raw-eeg/<sub>/ses-0{1..4}/raw_eeg_{training,test}.npy
```

Cada `.npy` es un dict con `raw_eeg_data` `(64, N)` — **63 canales EEG + un canal `stim`** —,
`ch_names`, `ch_types`, `sfreq=1000`, `highpass=0.01` y `lowpass=100` (filtro de adquisición ya
aplicado en hardware). **≈15 GB por sujeto.**

**Eventos**: viven en el canal `stim` como muestras sueltas no-cero cuyo valor es el **índice de
imagen (1-based)**; el código `99999` marca *target/catch* y se descarta.

**Estructura de repeticiones (importante):**

- **Test**: 200 códigos × 20 reps por sesión × 4 sesiones = **80 reps por imagen**.
- **Training**: cada sesión cubre **8270 imágenes × 2 reps**, y **cada imagen aparece en
  exactamente 2 de las 4 sesiones** con un reparto entrelazado (`ses1∩ses2=∅`, `ses3∩ses4=∅`,
  `ses1∩ses3=4110`, `ses1∩ses4=4160`…). La unión de las 4 sesiones es exactamente `1..16540`
  → **4 reps por imagen**.

⚠️ Consecuencia: las repeticiones **se agrupan por código de imagen**, nunca concatenando
sesiones por el eje de repeticiones (eso daría un resultado incorrecto). Algunas condiciones
traen una repetición extra; se recorta al tope por sesión con una selección con semilla, igual
que hace el código oficial.

---

## 3. La baseline (configuración de referencia)

`configs/EEG/preproc/baseline.yaml`:

```
63 canales · 0.1–100 Hz · epoch −200…1000 ms · baseline −200…0 ms · 250 Hz ·
crop [0,1000) ms · sin ICA/ASR/CAR/notch · MVNN (fit solo con training, aplicado
antes de promediar) · avg-4 en train · 80 reps guardadas en test  →  tensor 63 × 250
```

Orden de operaciones (ramificando desde el raw lo antes posible, nunca encadenando variantes):

```
raw → QC no destructivo → seleccionar canales → filtrar continuo → referencia →
epoch [−200,1000) → baseline → resample → crop → split por imagen →
MVNN fit (solo train) → MVNN aplicado a CADA repetición → agregación → guardar
```

Detalles que importan metodológicamente:

- **Contrato half-open** `[tmin, tmax)`: `[0,1000)` ms a 250 Hz son **exactamente 250 muestras**
  (nunca 251 por incluir la muestra de t=1000 ms).
- **Filtrado**: sobre la señal **continua**, antes del epoching. Backend **MNE** (FIR Hamming de
  fase cero, la referencia de THINGS-EEG2/NICE/ATM) con *fallback* automático a scipy si MNE no
  está; el backend y todos los parámetros quedan registrados en `metadata.json`.
- **Resample**: con protección antialias (`mne.filter.resample`, FFT). Nunca por *slicing*.
- **MVNN**: residuos por imagen → covarianza espacial con **shrinkage de Ledoit–Wolf** →
  `W = Σ^(-1/2)` con suelo de autovalores. Implementado en NumPy puro y **verificado contra
  `sklearn.covariance.LedoitWolf`** (coincide a ~1e-17); es además chunked, porque la matriz de
  residuos de una sesión de training ronda los 2 GB.
- **Sin z-score extra**: en las variantes raw, `dataset.normalize: false` (el doc prohíbe añadir
  z-score sobre MVNN). En el preprocesado oficial se mantiene `true` (comportamiento actual).
- **Anti-leakage**: el split train/val es **por imagen** (las 4 reps de una imagen nunca se
  separan) y lo calcula una función **compartida** (`src/data/eeg_split.py`) que usan tanto el
  pipeline como el datamodule, así que MVNN se ajusta exactamente sobre las imágenes que el
  entrenamiento llama *train*.

### Coste

~2,1 GB por sujeto y variante (train promediado `[16540, 1, 63, 250]` + test completo
`[200, 80, 63, 250]`), y del orden de decenas de minutos por sujeto.

---

## 4. Las 10 ablaciones

Cada una es un override mínimo sobre la baseline (`configs/EEG/preproc/*.yaml`) y **cambia un
solo factor**. Todas parten del raw y **recalculan MVNN** cuando cambia la representación.

| Variante | Cambio | Tensor | Pregunta |
|---|---|---|---|
| `ablate_mvnn` | sin MVNN | 63×250 | ¿cuánto aporta el whitening? |
| `channels_17` | 17 posteriores (desde raw) | 17×250 | ¿cuánto aporta lo no posterior? |
| `temporal_100_600` | crop [100,600) | 63×125 | ¿basta la ventana visual de NICE? |
| `temporal_200_400` | crop [200,400) | 63×50 | ventana estrecha tipo ATM |
| `sampling_100hz` | 100 Hz | 63×100 | ¿compensa la resolución temporal? |
| `frequency_0_5_40` | filtro 0.5–40 Hz | 63×250 | ¿aporta el broadband? |
| `train_independent_trials` | 4 trials sueltos | 63×250 | SNR vs nº de ejemplos |
| `reference_car` | CAR (antes de MVNN) | 63×250 | efecto de la referencia |
| `baseline_minus100` | baseline −100…0 | 63×250 | sensibilidad al RSVP |
| `baseline_none` | sin baseline | 63×250 | sensibilidad al RSVP |

Ojo con `channels_17`: los 17 canales se seleccionan **desde el raw** (no se usa el derivado
oficial 17×100 Hz, que mezclaría canales + frecuencia de muestreo) y se ajusta una MVNN 17×17.

Además, la **curva de repeticiones de test** (`R = 1,2,4,8,20,40,80`) **no requiere reentrenar**:
es un protocolo de evaluación (`scripts/11_eval_test_repetitions.py`) con `n_draws` subconjuntos
reproducibles por semilla para `R < 80`.

---

## 5. Cómo se ejecuta

> ⚠️ **Usa el python del venv para el paso 09**: MNE está instalado en
> `.tfm_fmri_diffusion_3_11`, no en el python base del sistema. La baseline fija
> `preprocessing.filter.backend: mne`, así que con el intérprete equivocado **falla en 2 s con un
> mensaje claro** en lugar de degradar en silencio a un IIR de scipy (que daría una variante no
> comparable con el resto, §3.3). Desviación deliberada:
> `--set preprocessing.filter.backend=scipy`.
>
> El cache está protegido por **hash de configuración** (§13): si cambias un parámetro de
> preprocesado y reejecutas sin `--force`, el script se niega a reutilizar el cache antiguo.

```bash
# 1) construir una variante (una vez por variante y sujeto) — con el python del venv
.tfm_fmri_diffusion_3_11/Scripts/python.exe scripts/09_preprocess_eeg_raw.py \
    --config configs/EEG/preproc/baseline.yaml
python scripts/10_validate_eeg_preproc.py --config configs/EEG/preproc/baseline.yaml

# 2) usarla en los experimentos (por CLI sobre los configs EEG de siempre)
python scripts/00_prepare_dataset.py --config configs/EEG/exp01_63_eeg_to_clip.yaml \
  --set dataset.source=raw --set dataset.preproc_variant=baseline \
  --set dataset.normalize=false --set experiment.name=exp01_raw_baseline_eeg_to_clip
python scripts/02_train_fmri_to_clip.py --config configs/EEG/exp01_63_eeg_to_clip.yaml \
  --set dataset.source=raw --set dataset.preproc_variant=baseline \
  --set dataset.normalize=false --set experiment.name=exp01_raw_baseline_eeg_to_clip

# o con la plantilla ya montada
python scripts/02_train_fmri_to_clip.py --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml

# 3) curva de repeticiones de test (no reentrena)
python scripts/11_eval_test_repetitions.py --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
```

Salida de una variante: `data/processed/eeg_preproc/<variante>/<sub>/` con
`preprocessed_eeg_{training,test}.npy` (**mismo contrato que los ficheros oficiales**),
`metadata.json` (hash de config, canales, sfreq, ventanas, filtro, stats de MVNN, QC, versiones)
y figuras de QC en `qc/`.

---

## 6. ⚠️ Qué hay que reejecutar al cambiar de preprocesamiento

Esta es la tabla clave para no repetir trabajo caro:

| Paso | ¿Reejecutar? | Por qué |
|---|---|---|
| `09_preprocess_eeg_raw` | **Sí**, 1× por variante | genera el tensor de la variante |
| `00_prepare_dataset` | **Sí** (segundos) | metadata/normalización van *namespaced* por variante |
| `01_precompute_clip` | **NO** | depende **solo de las imágenes**. Mismo set (16 540/200) y mismo split ⇒ mismo `feat_idx`. **Se comparte entre TODAS las variantes** e incluso con la línea del preprocesado oficial |
| `04_precompute_vae_pca` | **NO** | igual (imágenes + PCA ajustada en train con el mismo split) |
| `02_train_fmri_to_clip` (Exp1) | **Sí** | cambia la entrada cerebral |
| `03_eval_retrieval_ablation` (Exp2) | **Sí** | usa el checkpoint de Exp1 |
| `05_train_multitask` (Exp3) | **Sí** | cambia la entrada cerebral |
| `06` — **entrenar el TokenAdapter** | **NO** | el adapter se entrena con `CLIP → latentes VAE`: **no ve el EEG**. Se reutiliza entre variantes |
| `06` — **generar imágenes** | **Sí** | el decoder cambió |
| `07_eval_generation_ablation` (Exp5) | **Sí** | mide sobre las imágenes nuevas |
| `11_eval_test_repetitions` | **No reentrena** | solo evalúa el checkpoint ya entrenado |

**Condición para que esto se cumpla**: mantener `dataset.val_ratio` y `dataset.split_seed`
constantes y no descartar imágenes (política `repetitions.*.on_missing: fail`, que además avisa
si algo se sale de lo esperado). Si una variante cambia canales o muestras (17×250, 63×125…),
el encoder cambia de forma de entrada: es **otro experimento** (otro `experiment.name`), pero
**no** invalida las features CLIP/VAE.

---

## 7. Comparabilidad entre variantes (protocolo)

Para poder atribuir las diferencias al preprocesamiento, cada variante que se reentrene debe
usar **el mismo split, arquitectura, hiperparámetros, criterio de checkpoint, seeds, épocas y
métricas**. Se recomienda repetir con **3 seeds** (5 si el tiempo lo permite) y tratar el
**sujeto como unidad de análisis**, reportando por sujeto y agregado, con la diferencia absoluta
y relativa frente a la baseline.

Recuerda además la interpretación honesta de la línea EEG (Documento 4 §7): el retrieval a nivel
de imagen es la métrica fiable, y la conclusión de **decodificación** (Exp2) se reporta **por
separado** de la de **generación** (Exp5).

---

## 8. Validación implementada

`scripts/10_validate_eeg_preproc.py` ejecuta los tests obligatorios del documento de requisitos:
formas exactas por variante (63×250, 17×250, 63×125, 63×50, 63×100, incluido el *half-open* sin
off-by-one), **no-leakage de MVNN** (alterar los trials held-out no cambia `W`), splits disjuntos
por imagen, conteo de repeticiones (y que el promedio no dependa del orden), lista exacta de los
17 canales posteriores, y que tras CAR la media entre canales es ≈ 0.
