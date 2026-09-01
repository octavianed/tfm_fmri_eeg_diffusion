# Cerebro → Imagen: decodificación y reconstrucción visual a partir de fMRI y EEG

Código del Trabajo de Fin de Máster (TFM). El sistema decodifica, a partir de la
respuesta cerebral, la imagen que una persona estaba observando, y utiliza esa
representación decodificada para condicionar un modelo de difusión latente
**congelado** con el que reconstruir una aproximación del estímulo.

Dos **modalidades** de neuroimagen comparten un único marco experimental: los
mismos objetivos visuales, cabezas de predicción, funciones de pérdida, controles
negativos, métricas y flujo de trabajo. Solo cambian el conjunto de datos y el
codificador cerebral:

| Modalidad | Conjunto de datos | Señal | Codificador |
|---|---|---|---|
| **fMRI** | NSD / Algonauts 2023 | Respuesta espacial por imagen (39 548 vértices) | Perceptrón multicapa residual |
| **EEG** | THINGS-EEG2 | Serie temporal multicanal (`canales × tiempo`) | Red convolucional temporal con agregación por atención |

El proyecto separa deliberadamente dos preguntas que suelen presentarse mezcladas:

1. **Decodificación** — ¿predice la señal cerebral una representación visual de la
   imagen observada? Es medible *sin generar nada*.
2. **Generación** — ¿pueden esas predicciones guiar un generador de imágenes
   congelado?

CLIP, el VAE, Stable Diffusion, el codificador de texto y ControlNet **nunca se
entrenan**. Los únicos módulos entrenables son el codificador cerebral con sus
cabezas de predicción y un pequeño adaptador de tokens.

### Principio rector

Todo se evalúa contra un criterio falsable:

```
señal correcta   →  claramente por encima del azar
señal permutada  →  ≈ azar          (control negativo)
señal nula       →  ≈ azar          (control negativo)
ruido            →  ≈ azar          (control negativo)

se afirma "se está usando la señal cerebral" SOLO SI:  correcta ≫ permutada ≈ nula
```

Si la condición correcta no supera claramente a los controles, el código lo
declara de forma explícita (`metrics/conclusion.json`, `report/*.md`) y el
resultado **no** se atribuye a información cerebral real: unas imágenes visualmente
convincentes no son, por sí solas, evidencia de decodificación.

---

## Índice

1. [Hardware e instalación](#1-hardware-e-instalación)
2. [Conjuntos de datos](#2-conjuntos-de-datos)
3. [Pipeline completo desde cero](#3-pipeline-completo-desde-cero)
4. [Experimentos y configuraciones finales](#4-experimentos-y-configuraciones-finales)
5. [Ejemplo completo: reproducir la configuración Cf](#5-ejemplo-completo-reproducir-la-configuración-cf)
6. [EEG: preprocesamiento propio desde la señal cruda y ablaciones](#6-eeg-preprocesamiento-propio-desde-la-señal-cruda-y-ablaciones)
7. [Arquitecturas de condicionamiento: texto y control espacial](#7-arquitecturas-de-condicionamiento-texto-y-control-espacial)
8. [Detener y reanudar entrenamientos](#8-detener-y-reanudar-entrenamientos)
9. [Salidas](#9-salidas)
10. [Interpretación de los controles](#10-interpretación-de-los-controles)
11. [Resultados y puntos de control publicados](#11-resultados-y-puntos-de-control-publicados)
12. [Estructura del proyecto](#12-estructura-del-proyecto)
13. [Reproducibilidad](#13-reproducibilidad)
14. [Documentación](#14-documentación)

> Versión en inglés de este mismo documento: [`README_en.md`](README_en.md).

---

## 1. Hardware e instalación

Dimensionado para **32 GB de RAM y 16 GB de memoria de vídeo**. Todas las salidas
de los modelos congelados se precalculan en disco, de modo que el entrenamiento
del decodificador nunca carga Stable Diffusion.

```bash
python -m venv .venv && .venv\Scripts\activate       # Windows
# python -m venv .venv && source .venv/bin/activate  # Linux / macOS
pip install -e .
```

A continuación hay que instalar la versión de PyTorch con CUDA correspondiente al
driver. El entorno utilizado para todos los resultados de la memoria fue
**Python 3.11.9** con **torch 2.12.1+cu130 / torchvision 0.27.1+cu130**:

```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

**Reproducción exacta.** El fichero `requirements.txt` fija todas las dependencias
a la versión exacta de ese entorno (transformers 5.6.2, diffusers 0.37.1,
open_clip_torch 3.3.0, numpy 2.4.4, mne 1.12.1, …):

```bash
pip install -r requirements.txt
```

Notas:

- `pip install -e .` instala las dependencias declaradas en `pyproject.toml` (con
  rangos de versión, sin fijar) y hace que `import src...` funcione desde la raíz
  del repositorio; los guiones también funcionan sin instalar nada, porque añaden
  ellos mismos la raíz del repositorio a `sys.path`.
- **MNE** solo es necesario para el pipeline de preprocesamiento del EEG desde la
  señal cruda (§6). Está incluido en `requirements.txt` y disponible como extra
  `eeg` (`pip install -e .[eeg]`). Si solo se usan los derivados oficiales, puede
  omitirse.
- **Ejecutar siempre desde la raíz del repositorio.** Todas las rutas del proyecto
  (`data/`, `outputs/`, `configs/`) son relativas. Los cuadernos incluyen una
  primera celda que cambia el directorio de trabajo automáticamente: hay que
  ejecutarla.
- Detrás de un proxy corporativo, `pip` puede necesitar
  `--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

Todos los puntos de entrada aceptan sobrescrituras con `--set clave.ruta=valor`,
por ejemplo `--set dataset.subject_selection=all --set training.batch_size=32`.

---

## 2. Conjuntos de datos

### 2.1 fMRI — NSD / Algonauts 2023

Datos: <https://algonautsproject.com/2023/braindata.html> (kit de desarrollo:
<https://github.com/gifale95/algonauts_2023>). Se detectan automáticamente dos
disposiciones en disco.

**A) Disposición oficial con la fMRI de test liberada** (recomendada: es la
publicación completa posterior al reto):

```
<root>/train_data/subjNN/training_split/training_fmri/{lh,rh}_training_fmri.npy
<root>/train_data/subjNN/training_split/training_images/train-XXXX_nsd-YYYYY.png
<root>/train_data/subjNN/roi_masks/...
<root>/test_data/subjNN/test_split/test_fmri/{lh,rh}_test_fmri.npy
<root>/test_data/subjNN/test_split/test_images/test-XXXX_nsd-YYYYY.png
```

**B) Disposición plana** (`<root>/subjNN/training_split/...`), utilizada cuando la
fMRI de test no está disponible; en ese caso el código extrae un conjunto de
prueba interno de los datos de entrenamiento etiquetados.

```yaml
# configs/fMRI/base.yaml
dataset:
  root_dir: C:/Users/xxdia/Documents/Datasets/NSD_Algonauts_2023
  subject_selection: subj01          # o [subj01, subj02] o all
  test_split: official               # official (usa test_data) | internal
```

Para `subj01`: 39 548 vértices (ambos hemisferios concatenados), 8 857 imágenes de
entrenamiento, 984 de validación y **159 imágenes de prueba oficiales**.

Los *captions* que emplean las arquitecturas multimodales viven en
`<root>/auxiliar/generated_captions/` y los genera
[`notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb`](notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb),
que los deriva de las anotaciones de COCO asociadas a cada estímulo del NSD.

### 2.2 EEG — THINGS-EEG2

```
<root>/image_set/                                        imágenes estímulo + image_metadata.npy
<root>/image_set/generated_captions/                     captions
<root>/preprocessed_data/<sub>/preprocessed_eeg_{training,test}.npy      montaje de 17 canales
<root>/preprocessed_data/<sub>__63_channels/...                          montaje de 63 canales
<root>/raw-eeg/<sub>/ses-0{1..4}/raw_eeg_{training,test}.npy             señal cruda, 1 kHz (§6)
```

Cada fichero preprocesado es un diccionario cuyo `preprocessed_eeg_data` tiene
forma `[imágenes, repeticiones, canales, tiempos]`: entrenamiento
`(16540, 4, C, 100)` y prueba `(200, 80, C, 100)`, muestreados a 100 Hz en la
ventana −200→790 ms. Los conceptos de prueba son **disjuntos** de los de
entrenamiento, por lo que la evaluación mide además la generalización a categorías
no vistas.

```yaml
# configs/EEG/base_63.yaml
dataset:
  modality: eeg
  root_dir: C:/Users/xxdia/Documents/Datasets/THINGS-EEG2
  subject_selection: sub-01
  channels: 63                       # 17 | 63
  trial_aggregation: {train: none, val: mean, test: mean}
```

El entrenamiento emplea muestras **por ensayo** (aumento de datos entre
repeticiones), mientras que **la evaluación y las ablaciones se ejecutan siempre a
nivel de imagen** (media sobre repeticiones), de modo que los candidatos de la
recuperación son imágenes únicas y la relación señal-ruido es alta.

Los *captions* de las arquitecturas multimodales viven en
`<root>/image_set/generated_captions/` y los genera
[`notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb`](notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb),
que los deriva del nombre del concepto THINGS de la carpeta de cada estímulo.

---

## 3. Pipeline completo desde cero

El orden es obligatorio: cada paso produce artefactos que consume el siguiente.
Los mismos guiones sirven para ambas modalidades, y la modalidad la determina el
fichero pasado en `--config`.

### 3.1 Línea de fMRI

```bash
# 0) Resolver sujetos, construir particiones reproducibles y ajustar la
#    normalización solo con entrenamiento
python scripts/00_prepare_dataset.py          --config configs/fMRI/exp01_fmri_to_clip.yaml

# 1) Precomputar los embeddings de imagen con CLIP congelado (objetivos semánticos)
python scripts/01_precompute_clip.py          --config configs/fMRI/exp01_fmri_to_clip.yaml

# 2) EXPERIMENTO 1 — entrenar cerebro → CLIP (no se carga Stable Diffusion)
python scripts/02_train_fmri_to_clip.py       --config configs/fMRI/exp01_fmri_to_clip.yaml

# 3) EXPERIMENTO 2 — ablación de recuperación + modelos de referencia (media y ridge)
python scripts/03_eval_retrieval_ablation.py  --config configs/fMRI/exp02_retrieval_ablation.yaml

# 4) Precomputar los latentes del VAE y ajustar la PCA (solo con entrenamiento):
#    los objetivos estructurales
python scripts/04_precompute_vae_pca.py       --config configs/fMRI/exp03_lowlevel_multitask.yaml

# 5) EXPERIMENTO 3 — decodificador multitarea (cabezas semántica y estructural)
python scripts/05_train_multitask.py          --config configs/fMRI/exp03_lowlevel_multitask.yaml

# 6) EXPERIMENTO 4 — entrenar el adaptador de tokens y generar con SD congelada
python scripts/06_generate_images.py          --config configs/fMRI/exp04_generation_legacy.yaml --train-adapter

# 7) EXPERIMENTO 5 — comparación generativa frente a los controles negativos
python scripts/07_eval_generation_ablation.py --config configs/fMRI/exp05_generation_legacy_ablation.yaml
```

### 3.2 Línea de EEG

Secuencia idéntica con las configuraciones de EEG:

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

*(El nombre del guion `02` es histórico: entrena la modalidad que declare la
configuración.)*

### 3.3 Precómputo adicional para las arquitecturas multimodales

Solo es necesario para las arquitecturas que usan texto y/o control espacial (§7).
Ambas cachés dependen **únicamente de las imágenes estímulo y de la partición**,
nunca de la señal cerebral, así que se calculan una vez y las comparten todas las
configuraciones y todas las variantes de preprocesamiento del EEG.

**Paso 0 — construir los captions.** Los CSV de captions no se distribuyen con los
conjuntos de datos; se generan una vez por modalidad con estos cuadernos, que
escriben en la carpeta `generated_captions/` de la raíz de cada conjunto de datos:

| Modalidad | Cuaderno | Salida |
|---|---|---|
| fMRI | [`notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb`](notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb) | `<root>/auxiliar/generated_captions/<subj>_{train,test}_captions.csv` |
| EEG | [`notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb`](notebooks/EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb) | `<root>/image_set/generated_captions/thingseeg2_{training,test}_image_captions.csv` |

En fMRI el *caption* débil es una lista de los sustantivos principales extraídos de
la descripción COCO de cada estímulo, separados por comas; en EEG es el nombre del
concepto THINGS recuperado de la carpeta de la imagen. Hay que ejecutar el cuaderno
de la modalidad correspondiente antes de `scripts/13`.

**Pasos 1 y 2 — las cachés.**

```bash
# Embeddings de texto (codificador de texto congelado de SD, deduplicados y con hash)
python scripts/13_precompute_text_embeddings.py       --config configs/fMRI/exp04_generation_text_weak.yaml

# Condiciones espaciales de ControlNet (imagen real → VAE → PCA → inversa → decodificación → Canny)
python scripts/14_precompute_controlnet_conditions.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml
```

### 3.4 Herramientas auxiliares

```bash
python scripts/08_sweep_adapter_checkpoints.py  --config <config exp04>   # elegir adaptador por calidad de generación
python scripts/12_sweep_adapter_input_scale.py  --config <config exp04>   # barrido de intensidad de condicionamiento
python scripts/16_sweep_controlnet_scale.py     --config <config exp04>   # barrido de escala del control espacial
python scripts/15_validate_multimodal.py        --config <config exp04>   # comprobaciones en CPU, segundos, sin SD
python scripts/plot_memoria_figures.py                                    # tablas y figuras de la memoria
```

---

## 4. Experimentos y configuraciones finales

El pipeline define cinco experimentos:

| Experimento | Guion | Pregunta que responde |
|---|---|---|
| **Exp 1** | `02_train_fmri_to_clip.py` | ¿Predice la señal cerebral la representación semántica? |
| **Exp 2** | `03_eval_retrieval_ablation.py` | ¿Depende eso de usar la señal *correcta*? (controles y modelos de referencia) |
| **Exp 3** | `05_train_multitask.py` | ¿Predice además estructura de bajo nivel, sin perder semántica? |
| **Exp 4** | `06_generate_images.py` | ¿Pueden las predicciones guiar un generador congelado? |
| **Exp 5** | `07_eval_generation_ablation.py` | ¿Depende la imagen generada de la señal correcta? |

La memoria reporta **siete configuraciones generativas**, todas con
`generation.mode: adapter`. Los códigos siguientes son los empleados en el
documento.

| Código | Modalidad / línea | Arquitectura | Configuración Exp 4 | Configuración Exp 5 |
|---|---|---|---|---|
| **Af** | fMRI | A · solo señal cerebral | `configs/fMRI/exp04_generation_legacy.yaml` | `configs/fMRI/exp05_generation_legacy_ablation.yaml` |
| **Bf** | fMRI | B · texto débil + señal | `configs/fMRI/exp04_generation_text_weak.yaml` | `configs/fMRI/exp05_generation_text_weak_ablation.yaml` |
| **Cf** | fMRI | C · texto + señal + control espacial | `configs/fMRI/exp04_generation_controlnet_weak.yaml` | `configs/fMRI/exp05_generation_controlnet_weak_ablation.yaml` |
| **Ae** | EEG, oficial 63 canales | A · solo señal cerebral | `configs/EEG/exp04_63_generation_legacy.yaml` | `configs/EEG/exp05_63_generation_legacy_ablation.yaml` |
| **Be** | EEG, oficial 63 canales | B · texto débil + señal | `configs/EEG/exp04_63_generation_text_weak.yaml` | `configs/EEG/exp05_63_generation_text_weak_ablation.yaml` |
| **Ce** | EEG, oficial 63 canales | C · texto + señal + control espacial | `configs/EEG/exp04_63_generation_controlnet_weak.yaml` | `configs/EEG/exp05_63_generation_controlnet_weak_ablation.yaml` |
| **Ct** | EEG, preprocesamiento propio, 100–600 ms | C · texto + señal + control espacial | `configs/EEG/exp04_raw_temporal_100_600_generation_controlnet_weak.yaml` | `configs/EEG/exp05_raw_temporal_100_600_generation_controlnet_weak_ablation.yaml` |

Sus contrapartidas de decodificación (Exp 1–3) se comparten por línea:
`exp01_fmri_to_clip`, `exp02_retrieval_ablation` y `exp03_lowlevel_multitask` para
fMRI; `exp01_63_eeg_to_clip`, `exp02_63_retrieval_ablation` y
`exp03_63_lowlevel_multitask` para la línea oficial de EEG; y la familia
`exp0X_raw_temporal_100_600_*` para la línea Ct.

**Un adaptador por (modalidad, arquitectura, modo de texto).** Los adaptadores *no*
se comparten entre modalidades: fMRI y EEG emplean conjuntos de imágenes distintos,
de modo que cada modalidad entrena los suyos. Resultan **tres adaptadores por
modalidad —seis en total— que cubren las siete configuraciones**:

| Adaptador | Entrenado en | Reutilizado por |
|---|---|---|
| fMRI · arquitectura A | Af | — |
| fMRI · arquitectura B | Bf | — |
| fMRI · arquitectura C | Cf | — |
| EEG · arquitectura A | Ae | — |
| EEG · arquitectura B | Be | — |
| EEG · arquitectura C | Ce | **Ct** |

La única reutilización es **Ct tomando el adaptador de Ce**: ambas son EEG,
arquitectura C y texto débil, y difieren *únicamente* en la variante de
preprocesamiento del EEG, que el adaptador nunca ve, puesto que se entrena con
*embeddings* CLIP y latentes del VAE de las imágenes estímulo. Reutilizarlo no es
solo un atajo: mantiene la etapa generativa idéntica entre ambas líneas, de modo
que cualquier diferencia en el Experimento 5 sea atribuible al preprocesamiento y
al decodificador, y no a dos adaptadores entrenados por separado.

---

## 5. Ejemplo completo: reproducir la configuración Cf

Recorrido completo desde un `data/` y un `outputs/` vacíos hasta
`exp05_generation_controlnet_weak_ablation`, la configuración de fMRI con texto y
control espacial.

```bash
# --- Etapa 0: preparación de datos y objetivos congelados ------------------
python scripts/00_prepare_dataset.py     --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/01_precompute_clip.py     --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/04_precompute_vae_pca.py  --config configs/fMRI/exp03_lowlevel_multitask.yaml

# --- Etapa 1: decodificador cerebral ---------------------------------------
python scripts/02_train_fmri_to_clip.py  --config configs/fMRI/exp01_fmri_to_clip.yaml
python scripts/05_train_multitask.py     --config configs/fMRI/exp03_lowlevel_multitask.yaml
```

La arquitectura C necesita el decodificador **multitarea** (Exp 3), porque su
condición espacial se deriva de la predicción estructural.

```bash
# --- Etapa 2: ablación de controles negativos del decodificador ------------
# (a) sobre el decodificador del Exp 1 — el modelo puramente semántico
python scripts/03_eval_retrieval_ablation.py --config configs/fMRI/exp02_retrieval_ablation.yaml

# (b) sobre el decodificador del Exp 3 — el que realmente alimenta la generación.
#     Misma configuración, redirigiendo el experimento de origen y con una carpeta
#     de salida distinta.
python scripts/03_eval_retrieval_ablation.py \
  --config configs/fMRI/exp02_retrieval_ablation.yaml \
  --set evaluation.source_experiment=exp03_fmri_lowlevel_multitask \
        experiment.name=exp02_fmri_retrieval_ablation_exp3
```

La variante **(b)** es la que se reporta en la memoria, de modo que toda la cadena
reportada emplee un único decodificador. La variante (a) se conserva como
comprobación de consistencia; ambas llegan al mismo veredicto cualitativo.

```bash
# --- Etapa 3: cachés multimodales (compartidas, se calculan una vez) -------
# Requisito previo: ejecutar una vez
# notebooks/fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb, para que existan los
# CSV de captions en <root>/auxiliar/generated_captions/.
python scripts/13_precompute_text_embeddings.py       --config configs/fMRI/exp04_generation_controlnet_weak.yaml
python scripts/14_precompute_controlnet_conditions.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml

# --- Etapa 4: entrenar el adaptador de tokens y generar --------------------
python scripts/06_generate_images.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml --train-adapter

# --- Etapa 5: evaluación pareada frente a todos los controles -------------
python scripts/07_eval_generation_ablation.py --config configs/fMRI/exp05_generation_controlnet_weak_ablation.yaml
```

Esto produce las ocho condiciones de la arquitectura C y los contrastes pareados
(`delta_brain`, `delta_text`, `delta_semantic`, `delta_lowlevel` y sus variantes
frente al nulo) con los p-valores de la prueba *t* y de Wilcoxon.

**Reutilizar un adaptador ya entrenado** (para otra variante de preprocesamiento, o
para evitar reentrenarlo):

```bash
python scripts/06_generate_images.py \
  --config configs/EEG/exp04_raw_temporal_100_600_generation_controlnet_weak.yaml \
  --set generation.train_adapter=false \
  --adapter-checkpoint outputs/exp04_63_eeg_generation_controlnet_weak/checkpoints/adapter_best.pt
```

**Prueba de humo rápida** (minutos en lugar de horas; no es un resultado):

```bash
python scripts/06_generate_images.py --config configs/fMRI/exp04_generation_controlnet_weak.yaml --train-adapter \
  --set generation.adapter_epochs=1 generation.adapter_max_train_samples=64 \
        generation.num_samples=4 generation.num_inference_steps=6
```

---

## 6. EEG: preprocesamiento propio desde la señal cruda y ablaciones

Además de los derivados oficiales, la línea de EEG puede partir de los **registros
crudos de 63 canales** y aplicar un pipeline propio parametrizable, que además
habilita una batería de ablaciones de preprocesamiento. Detalles en la §10 de
[`docs/00_documentacion_general.md`](docs/00_documentacion_general.md).

Configuración de referencia (`configs/EEG/preproc/baseline.yaml`): 63 canales,
filtrado paso banda de 0,1 a 100 Hz sobre la señal continua, épocas de −200 a
1000 ms, línea base sobre los 200 ms previos al estímulo, remuestreo a 250 Hz,
recorte semiabierto `[0, 1000)` ms (**exactamente 250 muestras**), blanqueo
multivariante del ruido (MVNN) ajustado solo con las imágenes de entrenamiento y
aplicado **a cada repetición antes de promediar**, promedio de las 4 repeticiones
en entrenamiento y conservación de las 80 repeticiones de prueba →
`63 × 250` por imagen.

```bash
# 1) Construir una variante (una vez por variante y sujeto): requiere el
#    intérprete del entorno virtual
.tfm_fmri_diffusion_3_11/Scripts/python.exe scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/baseline.yaml

# 2) Validarla (formas exactas, ausencia de fuga en el MVNN, particiones
#    disjuntas, lista de canales)
python scripts/10_validate_eeg_preproc.py  --config configs/EEG/preproc/baseline.yaml

# 3) Usarla en los experimentos (hay configuraciones preparadas por variante)
python scripts/00_prepare_dataset.py       --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
python scripts/02_train_fmri_to_clip.py    --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml

# 4) Curva de repeticiones de prueba (R = 1…80): es un protocolo de evaluación,
#    no requiere reentrenar
python scripts/11_eval_test_repetitions.py --config configs/EEG/exp01_raw_baseline_eeg_to_clip.yaml
```

> El paso `09` **requiere el intérprete del entorno virtual del proyecto**: MNE
> está instalado ahí y no en el Python del sistema. La configuración de referencia
> fija `preprocessing.filter.backend: mne` precisamente para que un intérprete
> equivocado falle en un par de segundos con un mensaje accionable, en lugar de
> degradar en silencio a un filtro IIR de scipy que haría esa variante no
> comparable con las demás. La caché de variantes está además protegida por un
> *hash* de configuración: editar un parámetro de preprocesamiento y reejecutar sin
> `--force` se rechaza.

Diez ablaciones de un solo factor se distribuyen como sobrescrituras mínimas en
`configs/EEG/preproc/`: `ablate_mvnn`, `channels_17`, `temporal_100_600`,
`temporal_200_400`, `sampling_100hz`, `frequency_0_5_40`,
`train_independent_trials`, `reference_car`, `baseline_minus100` y
`baseline_none`. Cada una ramifica desde la señal cruda —nunca desde otra
variante—, de modo que el factor manipulado sea realmente lo único que cambia.

Una variante escribe en `data/processed/eeg_preproc/<variante>/<sujeto>/` usando
**el mismo contrato de ficheros que los derivados oficiales**, así que el
*datamodule*, el codificador y los experimentos 1 a 5 la consumen sin cambios.

### Qué reejecutar cuando cambia el preprocesamiento

| Paso | ¿Reejecutar? | Motivo |
|---|---|---|
| `09` (construir la variante) y `00` (metadatos) | **Sí** | El tensor y la normalización son específicos de la variante |
| `01` (CLIP) y `04` (VAE + PCA) | **No** | Dependen solo de las imágenes y de la partición, y siguen alineados por índice |
| Exp 1, Exp 2 y Exp 3 | **Sí** | Cambia la entrada cerebral |
| Entrenamiento del adaptador | **No** | El adaptador proyecta CLIP a latentes del VAE y nunca ve el EEG |
| Generación (Exp 4) y Exp 5 | **Sí** | El decodificador ha cambiado |

Esto se cumple mientras no cambien `dataset.val_ratio` ni `dataset.split_seed`.

---

## 7. Arquitecturas de condicionamiento: texto y control espacial

Stable Diffusion 1.5 permanece congelada. La señal cerebral puede alcanzarla por
tres puertas: la **secuencia de condicionamiento** (atención cruzada), el
**latente inicial** y los **residuos espaciales** que inyecta una red de control.
Sobre eso, `generation.conditioning_architecture` selecciona qué fuentes de
información acompañan a la condición neural:

| Valor | Condición que recibe la U-Net congelada |
|---|---|
| `legacy_adapter` *(por defecto)* | `[K pseudo-tokens neurales]` — solo cerebro, sin *caption* |
| `text_adapter_concat` | `[77 tokens de texto ; K pseudo-tokens]` por atención cruzada |
| `text_adapter_concat_controlnet` | lo anterior **más** los residuos de una ControlNet congelada, derivados de la predicción estructural |

La red de control **no sustituye** a la concatenación: el texto y los
pseudo-tokens siguen entrando por atención cruzada, mientras ella añade residuos
espaciales construidos a partir de `low_pred → PCA inversa → decodificación del
VAE → Canny`. El **adaptador de tokens sigue siendo el único módulo entrenable**.

`generation.mode` es un **eje ortogonal** (`adapter`, `lowlevel_img2img`,
`adapter_lowlevel`) que decide por dónde entra la predicción estructural. Todos
los experimentos reportados usan `adapter`, que es el modo cuyos contrastes son
atribuibles a una única fuente. Como el modo no forma parte de la identidad del
adaptador, cambiarlo no exige reentrenar nada.

**Modos de texto.** `generation.text.mode: none | weak | oracle` se resuelve como
`prompt_categories` / `primary_caption` en fMRI y como `primary_caption` en EEG, de
modo que en EEG **`oracle` equivale a `weak`** (THINGS-EEG2 no dispone de un
*caption* más detallado) y no debe reportarse como una condición más informativa.
`weak` es la *referencia* textual; `oracle` es un techo de rendimiento declarado.

**Los *captions* permutados** se obtienen con un desarreglo de Sattolo con semilla
42, extraído **dentro de cada `(sujeto, partición)` y dentro de la misma familia de
*caption***, usando exactamente la misma función que la permutación cerebral.

**Los puntos de control del adaptador no son intercambiables.** Cada uno almacena
su identidad de condicionamiento (arquitectura, modo de texto, campo de *caption*,
plantilla, `K`, modelo y tipo de condición de la red de control). Cargarlo bajo
otra combinación falla con un error explícito, salvo que se indique
`generation.allow_incompatible_adapter: true`, previsto únicamente para pruebas de
humo.

---

## 8. Detener y reanudar entrenamientos

Los entrenamientos guardan el estado *completo*: modelo, optimizador,
planificador, escalador de gradiente, época, `global_step`, mejor métrica,
contador de parada temprana, estado de los generadores aleatorios, configuración
resuelta y versiones de las bibliotecas.

```bash
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml --resume
#                                                                              ^ reanuda desde last.pt
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml --resume ruta/al/last.pt
```

o bien, desde la configuración:

```yaml
checkpointing:
  resume: auto      # busca checkpoints/last.pt en la carpeta de salida del experimento
```

Los ficheros `last.pt` y `best.pt` nunca se eliminan; solo se borran los
`epoch_XXXX.pt` periódicos que exceden `keep_last_n`. Cada reanudación se anota en
`logs/resume_history.jsonl` y continúa `logs/train_log.csv` sin perder las filas
anteriores.

> Cada punto de control de fMRI ocupa unos 2,7 GB (pesos más estados del
> optimizador). Conviene usar `checkpointing.save_every_n_epochs: 5` y
> `keep_last_n: 1` para ahorrar disco.

---

## 9. Salidas

```
outputs/<experimento>/
  config.yaml                                   configuración resuelta completa
  checkpoints/{last,best,epoch_XXXX}.pt         decodificador, o adapter_{best,last}.pt en el Exp 4
  logs/{train_log.csv, resume_history.jsonl}
  metrics/*.{json,csv}                          métricas del experimento
  figures/*.png                                 curvas, barras por condición, correlación PCA
  embeddings/*.npy   lowlevel/*.npy             predicciones y objetivos
  generated/{real,correct,permuted,zero,...}/*.png
  grids/*.png                                   rejillas comparativas y de mejores/medianos/peores casos
  metadata/{generation_params.json, generation_samples.json}
  report/*.md                                   resumen legible y veredicto
```

Con las arquitecturas multimodales, `generated/` contiene además las condiciones
adicionales (`permuted_text`, `semantic_*`, `lowlevel_*`),
`metrics/generation_deltas.csv` recoge los contrastes pareados con sus p-valores, y
`metadata/generation_samples.json` guarda un registro por (muestra, condición) con
todo lo necesario para reconstruir esa imagen exacta: *prompt* resuelto, condición
cerebral y textual, semillas y escala del control.

Las métricas agregadas usan formato *tidy*: `metric_name, condition, subject_id,
split, value, seed, checkpoint`. El veredicto vive en `metrics/conclusion.json`
(Exp 2) y en `report/exp05_summary.md` (Exp 5).

---

## 10. Interpretación de los controles

| Condición | Qué es | Esperado |
|---|---|---|
| **Correcta** | La respuesta cerebral real de cada imagen | Claramente por encima del azar |
| **Permutada** | Cada muestra recibe la respuesta de *otra* muestra (desarreglo de Sattolo: nunca la suya) | ≈ azar |
| **Nula** | Vector cerebral de ceros, propagado por el decodificador | ≈ azar |
| **Ruido** | Ruido gaussiano de estadística comparable | ≈ azar |

Los niveles de azar dependen del número de candidatos: con `N` candidatos, el
Top-k es aproximadamente `k/N`. En el conjunto de prueba de fMRI (159 candidatos)
el Top-5 de azar es ≈ 3,14 %; en el de EEG (200 candidatos), ≈ 2,50 %. **Nunca hay
que comparar conjuntos entre sí**: cada uno se compara con su propio nivel de azar.

Dos advertencias que el código documenta explícitamente:

- **La similitud coseno media engaña.** Los *embeddings* de CLIP son anisótropos,
  de modo que el modelo de referencia de la media alcanza un coseno en torno a 0,75
  mientras rinde exactamente a nivel de azar en recuperación. La recuperación es la
  métrica honesta.
- **El R² de la cabeza semántica es negativo por diseño** (se optimiza dirección,
  no magnitud). El R² solo es informativo para la rama estructural.

Los veredictos de decodificación (Exp 2) y de generación (Exp 5) se reportan **por
separado**: una señal puede decodificar por encima del azar y, aun así, no guiar
la generación.

---

## 11. Resultados y puntos de control publicados

Lo que se publica no son únicamente los puntos de control: son las **carpetas de
salida completas** de cada experimento, tal y como las genera el sistema, para las
siete configuraciones reportadas. Están alojadas en la carpeta de Drive de la UOC:

> **Enlace:** `https://drive.google.com/drive/folders/1br4gnjzXwemb_9REs0okzS4cmvpgywlx?usp=sharing`

Estructura:

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

- Cada **carpeta de arquitectura** contiene las salidas de sus experimentos 4 y 5.
  El **punto de control del adaptador (`adapter_best.pt`)** está en la carpeta del
  experimento 4.
- Las carpetas de **ejecuciones comunes** contienen los experimentos 1, 2 y 3 de
  esa línea: los decodificadores que alimentan todas sus arquitecturas.

### Qué contiene cada carpeta

Cada subcarpeta es un directorio `outputs/<experimento>/` íntegro, con la
estructura descrita en la §9. En concreto se incluyen:

- **`config.yaml`**, la configuración resuelta completa con la que se ejecutó ese
  experimento. Es el registro autorizado de los parámetros empleados: permite
  comprobar exactamente qué se hizo, sin depender de que los ficheros de
  `configs/` no hayan cambiado después.
- **`metrics/`**, todas las métricas en JSON y CSV: recuperación por condición,
  modelos de referencia, métricas de la rama estructural, métricas de generación
  por condición, contrastes pareados con sus p-valores y el veredicto automático.
- **`figures/`** y **`grids/`**, las figuras y rejillas generadas, incluidas las
  comparativas por condición y los casos mejores, medianos y peores.
- **`generated/`**, las imágenes generadas para **todas** las condiciones del
  experimento, junto con la imagen real de referencia.
- **`embeddings/`** y **`lowlevel/`**, las predicciones y los objetivos en `.npy`,
  que permiten recalcular cualquier métrica sin volver a ejecutar el modelo.
- **`metadata/`**, los parámetros de generación y un registro por (muestra,
  condición) con lo necesario para reconstruir cada imagen concreta.
- **`logs/`** y **`report/`**, el histórico de entrenamiento por época y el resumen
  legible con la conclusión.

### Sobre los puntos de control

Para que el volumen se mantenga razonable, **se han conservado únicamente los
puntos de control `best`**: `best.pt` en los experimentos de decodificación y
`adapter_best.pt` en los de generación. Se han eliminado los `last.pt` y las
instantáneas periódicas `epoch_XXXX.pt`, que en la línea de fMRI ocupan del orden
de 2,7 GB cada una.

La consecuencia práctica es que con este material **se puede evaluar, generar y
reproducir todas las tablas y figuras sin reentrenar nada**, ya que los guiones de
evaluación y de generación usan precisamente el punto de control `best`. Lo que no
es posible es *continuar* un entrenamiento en el punto exacto en que se detuvo,
porque eso requiere el `last.pt` correspondiente.

### Cómo reutilizarlas

Basta copiar cada subcarpeta dentro de `outputs/` conservando el nombre del
experimento —el que figura en `experiment.name` de su configuración, y que también
puede leerse en el `config.yaml` incluido—; los guiones de evaluación y generación
las localizarán automáticamente. Para generar sin reentrenar el adaptador, se pasa
`--set generation.train_adapter=false` junto con `--adapter-checkpoint`.

### También en el propio repositorio

Estas mismas carpetas de salida están **versionadas en este repositorio**, bajo
`outputs/`, de modo que las métricas, las figuras y los informes de todos los
experimentos pueden consultarse directamente en GitHub sin descargar nada. Lo que
`.gitignore` excluye es únicamente la parte pesada o regenerable:

| En el repositorio | Solo en Drive |
|---|---|
| `config.yaml` (configuración resuelta) | `checkpoints/` (pesos de los modelos) |
| `metrics/` (JSON y CSV) | `generated/` (imágenes individuales) |
| `figures/` y `grids/` (PNG) | `embeddings/` y `lowlevel/` (`.npy`) |
| `logs/` (histórico por época) | Cualquier `.npy`, `.npz`, `.pkl`, `.pt` o `.pth` |
| `metadata/` (JSON) y `report/` (Markdown) | |

Conviene destacar que **las rejillas de `grids/` sí están versionadas**: aunque las
imágenes generadas individuales no se suben, las rejillas comparativas —imagen real
frente a cada condición, y los casos mejores, medianos y peores— sí, así que la
comparación cualitativa también puede revisarse desde el repositorio.

En resumen: para **consultar** los resultados basta con el repositorio; para
**reejecutar** la generación o la evaluación, o para reutilizar un decodificador o
un adaptador, hay que descargar las carpetas de Drive.

Los conjuntos de datos **no** se redistribuyen en ninguno de los dos sitios: deben
descargarse de sus fuentes originales (§2).

---

## 12. Estructura del proyecto

```
configs/fMRI        base.yaml + un YAML por experimento (compuestos con `_base_`)
configs/EEG         lo mismo para la línea de EEG, más preproc/ con las variantes del pipeline propio
scripts/            puntos de entrada 00–16 (la modalidad la elige --config) + generación de figuras
notebooks/          versiones visuales: {fMRI,EEG}/00..06 y 30_multimodal_comparison
docs/               documentación técnica completa (véase §14)

src/data            datamodule de fMRI, datamodule y dataset de EEG, normalizadores, particiones, captions, factoría
src/preprocessing   pipeline propio de EEG desde la señal cruda: lector, filtros, epoching, MVNN, control de calidad, constructor de variantes
src/features        embeddings CLIP, latentes del VAE, PCA ajustada solo en entrenamiento, embeddings de texto, carga de características
src/models          codificadores cerebrales (fMRI, EEG), cabezas de predicción, adaptadores, decodificador multitarea
src/losses          coseno, contrastiva InfoNCE, combinación multitarea
src/training        bucles de entrenamiento y validación, puntos de control completos y reanudación
src/evaluation      métricas de recuperación, de embedding y de generación, modelos de referencia, ablaciones, contrastes
src/generation      pipeline de SD congelada, arquitecturas de condicionamiento, condiciones de control, barridos, rejillas
src/utils           configuración, semillas, dispositivo y precisión mixta, registro, rutas, puntos de control, permutaciones
```

**Regla de oro:** toda la lógica vive en `src/`. Los guiones y los cuadernos solo
orquestan e importan de ahí, de modo que no existe implementación duplicada entre
la vía interactiva y la vía por lotes.

---

## 13. Reproducibilidad

- Las semillas están fijadas para Python, NumPy y PyTorch. Las permutaciones de
  los controles negativos usan una semilla estable e independiente del proceso, de
  modo que dos ejecuciones del mismo experimento sortean la misma permutación.
- Las particiones las calcula una **función compartida** que emplean tanto el
  pipeline de preprocesamiento como el *datamodule*, así que los estadísticos
  ajustados «sobre entrenamiento» se ajustan exactamente sobre las imágenes que el
  bucle de entrenamiento llama entrenamiento.
- Los artefactos precalculados están protegidos por *hashes* de configuración:
  cambiar un parámetro sin recalcular falla con un mensaje accionable en lugar de
  mezclar cachés incompatibles en silencio.
- Cada punto de control almacena la configuración resuelta y las versiones de las
  bibliotecas; cada generación guarda los metadatos suficientes para reconstruir
  cada imagen.
- Comprobaciones automáticas: `scripts/10_validate_eeg_preproc.py` (formas
  exactas, fuga en el MVNN, particiones disjuntas) y
  `scripts/15_validate_multimodal.py` (formas, alineación y permutación de los
  *captions*, equivalencia del control desactivado, compatibilidad del adaptador)
  se ejecutan en CPU en segundos.

---

## 14. Documentación

La documentación técnica completa está en un único documento:

**[`docs/00_documentacion_general.md`](docs/00_documentacion_general.md)**. Cubre
el diseño del sistema y sus cinco bloques funcionales, la organización del código,
el **sistema de configuración y cómo escribir configuraciones propias**, el flujo
de datos y los artefactos en disco, los cinco experimentos, ambos bloques de
procesamiento en detalle, el protocolo de evaluación con sus controles y métricas,
la línea de EEG, la reproducibilidad, los costes computacionales y el diagnóstico
de problemas frecuentes.

---

## Cómo citar

Si utilizas este código, cita el Trabajo de Fin de Máster asociado.

> ⬜ **PENDIENTE:** añadir la cita definitiva (autor, título, universidad, año) una
> vez depositada la memoria.
