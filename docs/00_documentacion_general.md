# Documentación técnica del sistema

Documento único de referencia del repositorio. Describe **qué hace el sistema**,
**cómo está organizado el código**, **cómo se configura** y **cómo se ejecuta**, con
el nivel de detalle necesario para entender, reproducir o extender el trabajo.

Para las instrucciones de instalación y los comandos de ejecución paso a paso,
véase el [`README.md`](../README.md) en la raíz del repositorio; este documento se
centra en el diseño y en el funcionamiento interno.

---

## Índice

1. [Qué hace el sistema](#1-qué-hace-el-sistema)
2. [Arquitectura general](#2-arquitectura-general)
3. [Organización del código](#3-organización-del-código)
4. [Sistema de configuración](#4-sistema-de-configuración)
5. [Datos y artefactos en disco](#5-datos-y-artefactos-en-disco)
6. [Los cinco experimentos](#6-los-cinco-experimentos)
7. [Bloque de decodificación cerebral](#7-bloque-de-decodificación-cerebral)
8. [Bloque generativo](#8-bloque-generativo)
9. [Evaluación, controles y métricas](#9-evaluación-controles-y-métricas)
10. [La línea de EEG en detalle](#10-la-línea-de-eeg-en-detalle)
11. [Reproducibilidad y validación del sistema](#11-reproducibilidad-y-validación-del-sistema)
12. [Coste computacional](#12-coste-computacional)
13. [Problemas frecuentes y diagnóstico](#13-problemas-frecuentes-y-diagnóstico)
14. [Configuraciones y experimentos reportados](#14-configuraciones-y-experimentos-reportados)

---

## 1. Qué hace el sistema

El sistema reconstruye la imagen que una persona estaba observando a partir de la
actividad cerebral registrada en ese momento. Recibe una señal de neuroimagen,
estima una representación visual de la imagen vista y utiliza esa estimación como
condicionamiento de un modelo generativo de difusión **congelado**, obteniendo una
reconstrucción aproximada del estímulo.

Funciona sobre dos modalidades de neuroimagen de naturaleza muy distinta:

| Modalidad | Conjunto de datos | Naturaleza de la señal | Codificador |
|---|---|---|---|
| **fMRI** | NSD / Algonauts 2023 | Respuesta espacial por imagen (39 548 vértices en el sujeto empleado) | Perceptrón multicapa residual |
| **EEG** | THINGS-EEG2 | Serie temporal multicanal (`canales × tiempo`) | Red convolucional temporal con agregación por atención |

### 1.1 Las dos etapas, y por qué están separadas

Los modelos generativos actuales producen imágenes plausibles incluso cuando la
información con la que se los condiciona es poco relevante. Por tanto, obtener una
reconstrucción convincente **no demuestra** que la señal cerebral haya contribuido
al resultado, y la inspección visual no permite distinguir ambos casos.

Para evitar esa ambigüedad, el sistema separa explícitamente dos preguntas:

1. **Decodificación** — ¿contiene la señal cerebral información que permita
   recuperar características de la imagen observada? Se mide directamente sobre
   las representaciones predichas, **sin generar ninguna imagen**.
2. **Generación** — ¿es esa información además suficiente para guiar la síntesis?

### 1.2 El principio falsable

Ambas etapas se juzgan con el mismo criterio:

```
señal correcta   →  claramente por encima del azar
señal permutada  →  ≈ azar          (control negativo)
señal nula       →  ≈ azar          (control negativo)
ruido            →  ≈ azar          (control negativo)

se afirma "uso de la señal cerebral" SOLO SI:  correcta ≫ permutada ≈ nula
```

Si la condición correcta no supera claramente a los controles, el sistema lo
declara de forma explícita (`metrics/conclusion.json` en decodificación,
`report/exp05_summary.md` en generación) y el resultado **no** se atribuye a
información cerebral real. Este criterio está implementado en el código, no es una
recomendación de estilo: la función `conclusion_from_summary` de
`src/evaluation/ablation_eval.py` emite el veredicto a partir de las métricas.

---

## 2. Arquitectura general

### 2.1 Los cinco bloques funcionales

- **Bloque 1 — señal de entrada.** Lectura del conjunto de datos,
  preprocesamiento, normalización y particionado reproducible por imagen.
- **Bloque 2 — extracción de características.** Con modelos congelados, obtiene
  desde la *imagen estímulo* los objetivos que el decodificador debe predecir y
  las condiciones auxiliares del generador. No forma parte de la cadena de
  inferencia: es la fuente de supervisión.
- **Bloque 3 — decodificación.** Único bloque cuyo entrenamiento es el objeto del
  trabajo: proyecta la señal cerebral a los espacios de representación visual.
- **Bloque 4 — generación.** Modelo de difusión latente congelado más un módulo de
  adaptación pequeño y entrenable.
- **Bloque 5 — evaluación.** Define las condiciones experimentales (incluidos los
  controles negativos, que actúan **sobre la señal de entrada**), calcula las
  métricas y emite los veredictos.

### 2.2 Qué se entrena y qué está congelado

El sistema reutiliza modelos preentrenados de gran tamaño sin ajustarlos. Esta
decisión reduce el coste, pero sobre todo **permite atribuir los resultados**: si
el generador pudiera ajustarse a los datos, sería imposible distinguir qué parte
del resultado procede de la señal cerebral y cuál de una memorización del conjunto
de estímulos.

| Componente | Parámetros | Estado |
|---|---:|---|
| Codificador visual semántico (CLIP ViT-L/14, rama de imagen) | ≈ 304 M | 🔒 congelado |
| Autocodificador latente del generador (VAE) | ≈ 84 M | 🔒 congelado |
| Red de difusión (U-Net de Stable Diffusion 1.5) | ≈ 860 M | 🔒 congelada |
| Codificador de texto del generador | ≈ 123 M | 🔒 congelado |
| Red de control espacial (ControlNet-Canny) | ≈ 361 M | 🔒 congelada |
| **Codificador cerebral, fMRI** | **237,6 M** | 🔓 entrenable |
| **Codificador cerebral, EEG** | **6,8 M** | 🔓 entrenable |
| Cabezas de predicción (semántica + estructural) | 1,6 M + 1,0 M | 🔓 entrenables |
| **Módulo de adaptación** (77 pseudo-tokens / 8 pseudo-tokens) | **61,4 M / 7,1 M** | 🔓 entrenable |

### 2.3 Un marco único para dos modalidades

Las dos modalidades comparten **todo** el sistema excepto el bloque 1 y el
codificador del bloque 3. Los objetivos visuales, las cabezas, las pérdidas, las
métricas, los controles, la generación y el sistema de puntos de control son los
mismos. Operativamente, la modalidad es una clave de configuración
(`dataset.modality: fmri | eeg`) que selecciona el lector de datos y el codificador
mediante dos funciones factoría:

- `src/data/factory.py::build_datamodule`
- `src/models/multitask_decoder.py::build_model`

Esta separación es deliberada: garantiza que cualquier diferencia observada entre
ambas líneas sea atribuible a la modalidad y no al montaje experimental.

---

## 3. Organización del código

### 3.1 Regla de oro

**Toda la lógica vive en `src/`.** Los guiones de `scripts/` y los cuadernos de
`notebooks/` únicamente orquestan: leen una configuración, llaman a funciones de
`src/` y escriben resultados. Los cuadernos importan **exactamente las mismas
funciones** que los guiones, de modo que no puede existir divergencia entre lo que
se ejecuta en un experimento y lo que se muestra en el análisis interactivo.

Una convención adicional: `torch`, `diffusers`, `open_clip` y `transformers` se
importan **de forma perezosa** dentro de las funciones que los necesitan. Así,
importar un módulo no arrastra dependencias pesadas si no se van a usar, y el
entrenamiento del decodificador nunca carga Stable Diffusion.

### 3.2 Mapa de `src/`

| Paquete | Responsabilidad |
|---|---|
| `src/utils` | Infraestructura transversal: carga de configuración con herencia, semillas, dispositivo y precisión mixta, registro de métricas, rutas centralizadas, puntos de control y permutaciones reproducibles |
| `src/data` | Lectura y preparación de datos. fMRI: lectura por hemisferios, normalización por vértice, *datamodule*. EEG: lectura de derivados, normalización por canal, *datamodule*, partición por imagen. Además: captions y la factoría de modalidad |
| `src/preprocessing` | Pipeline propio de EEG desde la señal cruda: lector del formato original y extracción de eventos, filtros, segmentación en épocas, blanqueo multivariante, control de calidad y constructor de variantes |
| `src/features` | Cálculo de los objetivos con modelos congelados: *embeddings* CLIP, latentes del VAE, ajuste de la PCA solo con entrenamiento, *embeddings* de texto y carga de características |
| `src/models` | Codificador de fMRI, codificador de EEG, cabezas de proyección, adaptadores (por sujeto y de tokens) y el decodificador multitarea con sus funciones de construcción |
| `src/losses` | Pérdida de coseno, pérdida contrastiva InfoNCE simétrica y su combinación multitarea |
| `src/training` | Bucles de entrenamiento y validación, parada temprana, y el núcleo `run_training(use_lowlevel)` compartido por los experimentos 1 y 3 |
| `src/evaluation` | Métricas de recuperación y de *embedding*, modelos de referencia, ablación de controles, métricas de generación y cálculo de los contrastes pareados |
| `src/generation` | Pipeline de difusión congelada, las tres arquitecturas de condicionamiento, condiciones espaciales, entrenamiento del adaptador, generación desde el cerebro, barridos y rejillas de comparación |

Dependencias entre paquetes: `utils` no depende de nadie; `data`, `features`,
`models` y `losses` dependen de `utils`; `training` y `evaluation` dependen de los
anteriores; `generation` depende de todo.

### 3.3 Los guiones de `scripts/`

Numerados por orden de uso. Todos aceptan `--config` y `--set`, y la modalidad la
determina el fichero de configuración.

| Guion | Qué hace | Produce |
|---|---|---|
| `00_prepare_dataset.py` | Resuelve sujetos, construye las particiones y ajusta la normalización solo con entrenamiento | Tabla de metadatos y estadísticos de normalización |
| `01_precompute_clip.py` | Codifica las imágenes con CLIP congelado | Objetivos semánticos en `.npy` |
| `02_train_fmri_to_clip.py` | **Experimento 1**: entrena codificador + cabeza semántica | Puntos de control, métricas, curvas |
| `03_eval_retrieval_ablation.py` | **Experimento 2**: evalúa los controles y los modelos de referencia | Métricas por condición, veredicto |
| `04_precompute_vae_pca.py` | Codifica con el VAE y ajusta la PCA (solo entrenamiento) | Latentes, vectores PCA y modelo de PCA |
| `05_train_multitask.py` | **Experimento 3**: entrena con las dos cabezas | Como el 02, más métricas estructurales |
| `06_generate_images.py` | **Experimento 4**: entrena el adaptador (`--train-adapter`) y genera | Imágenes por condición y metadatos |
| `07_eval_generation_ablation.py` | **Experimento 5**: puntúa las condiciones y calcula los contrastes | Métricas, deltas con p-valores, rejillas |
| `08_sweep_adapter_checkpoints.py` | Compara varios puntos de control del adaptador por calidad de generación | Tabla de márgenes y figura |
| `09_preprocess_eeg_raw.py` | Construye una variante de preprocesamiento propio del EEG | Tensores, metadatos y figuras de control |
| `10_validate_eeg_preproc.py` | Verifica una variante (formas, fugas, particiones) | Informe de comprobaciones |
| `11_eval_test_repetitions.py` | Curva de rendimiento frente al número de repeticiones promediadas | Métricas por número de repeticiones |
| `12_sweep_adapter_input_scale.py` | Barrido de la intensidad de condicionamiento, sin reentrenar | Tabla de márgenes y figura |
| `13_precompute_text_embeddings.py` | Codifica los *prompts* con el codificador de texto congelado | Caché de *embeddings* deduplicada |
| `14_precompute_controlnet_conditions.py` | Calcula las condiciones espaciales de entrenamiento | Mapas de contornos en PNG |
| `15_validate_multimodal.py` | Comprobaciones del condicionamiento multimodal (CPU, segundos) | Informe de comprobaciones |
| `16_sweep_controlnet_scale.py` | Barrido de la escala del control espacial | Tabla de márgenes y figura |
| `plot_adapter_loss_valsim.py` | Curva de pérdida y similitud de validación del adaptador | Figura |
| `plot_margin_vs_epoch.py` | Margen correcto − control frente a época del adaptador | Figura |
| `plot_memoria_figures.py` | Tablas y figuras del capítulo de resultados | CSV y PNG en `outputs/_memoria/` |

### 3.4 Los cuadernos de `notebooks/`

Versiones visuales de cada paso, con inspección de datos y figuras intermedias.
Existen en paralelo para las dos modalidades (`notebooks/fMRI/` y
`notebooks/EEG/`, numerados `00`–`06`), más dos transversales:
`30_multimodal_comparison.ipynb`, que compara ambas líneas frente al azar de cada
una, y `31_real_vs_canny_structural_figure.ipynb`, que ilustra la condición
espacial.

Dos cuadernos tienen una función propia y no son meras versiones visuales: los
que **generan los captions** empleados por las arquitecturas multimodales
(`fMRI/NSD_Algonauts_COCO_Captions_Pipeline.ipynb` y
`EEG/THINGS_EEG2_Folder_Captions_Pipeline.ipynb`). Deben ejecutarse una vez por
modalidad antes del precómputo de *embeddings* de texto.

> ⚠️ Los cuadernos incluyen una primera celda que cambia el directorio de trabajo
> a la raíz del repositorio. **Hay que ejecutarla**: las rutas del proyecto son
> relativas, y si el directorio de trabajo queda en `notebooks/` se crean
> carpetas `data/` y `outputs/` fantasma y la carga de características falla.

---

## 4. Sistema de configuración

### 4.1 Un experimento es un fichero

El sistema es **dirigido por configuración**: un experimento queda completamente
definido por un fichero YAML, sin escribir código. Esto no es solo comodidad, es
una garantía metodológica —reduce el riesgo de que dos experimentos difieran en
algo no documentado— y de reproducibilidad, ya que la configuración efectivamente
resuelta se guarda junto a los resultados (`outputs/<experimento>/config.yaml`).

```
configs/
  fMRI/
    base.yaml                        valores por defecto de la línea de fMRI
    exp01_fmri_to_clip.yaml          un fichero por experimento
    exp02_retrieval_ablation.yaml
    ...
  EEG/
    base_17.yaml, base_63.yaml       valores por defecto según montaje
    exp01_63_eeg_to_clip.yaml
    ...
    preproc/
      baseline.yaml                  variante de referencia del pipeline propio
      ablate_mvnn.yaml, ...          las diez ablaciones de preprocesamiento
```

### 4.2 Herencia con `_base_`

Cada configuración de experimento hereda de una base y **sobrescribe solo lo que
cambia**. La herencia es encadenable, de modo que las configuraciones derivadas
resultan muy cortas y la diferencia respecto a su padre es explícita:

```yaml
# configs/fMRI/exp03_lowlevel_multitask.yaml
_base_: [base.yaml]

experiment:
  name: exp03_fmri_lowlevel_multitask

model:
  use_lowlevel: true                 # activa la cabeza estructural
  lowlevel_head: {output_dim: 512}

losses:
  lambda_lowlevel: 0.25
```

Un ejemplo de cadena real de tres niveles en la línea de EEG:
`exp04_63_generation_controlnet_weak.yaml` → `exp04_63_generation_text_weak.yaml`
→ `exp04_63_generation.yaml` → `base_63.yaml`. Así, la configuración con control
espacial no puede desalinearse del bloque `dataset` de la configuración de texto
de la que deriva.

El cargador está en `src/utils/config.py`. Devuelve un objeto con acceso por punto
(`cfg.training.lr`) y una función `cfg.get("clave.anidada", valor_por_defecto)`.

> ⚠️ Detalle importante de `get`: si una clave existe **con valor nulo**,
> `cfg.get(clave, defecto)` devuelve `None`, no el valor por defecto. Varios
> ficheros usan `null` con el significado de «resuélvelo tú», así que los puntos
> del código que dependen de ello tienen su propia resolución explícita.

### 4.3 Sobrescritura desde la línea de órdenes

Cualquier clave puede modificarse sin editar ficheros:

```bash
python scripts/02_train_fmri_to_clip.py --config configs/fMRI/exp01_fmri_to_clip.yaml \
  --set dataset.subject_selection=all training.batch_size=32 training.lr=5e-5
```

Es el mecanismo recomendado para variaciones puntuales y para redirigir un
experimento sin duplicar su configuración. Ejemplo real, usado para evaluar la
ablación de controles sobre el decodificador multitarea en lugar del semántico:

```bash
python scripts/03_eval_retrieval_ablation.py --config configs/fMRI/exp02_retrieval_ablation.yaml \
  --set evaluation.source_experiment=exp03_fmri_lowlevel_multitask \
        experiment.name=exp02_fmri_retrieval_ablation_exp3
```

Cambiar `experiment.name` es la forma de **no sobrescribir** los resultados
anteriores, ya que la carpeta de salida se deriva de ese nombre.

### 4.4 Los bloques de configuración

| Bloque | Contenido y claves más relevantes |
|---|---|
| `project` | Nombre y **semilla global** (`seed`) |
| `experiment` | `name`: determina la carpeta `outputs/<name>/` |
| `paths` | Directorios de datos, características, particiones y salidas |
| `runtime` | `device` (`auto`/`cuda`/`cpu`), precisión, determinismo |
| `dataset` | `modality`, `root_dir`, `subject_selection`, `test_split`, `val_ratio`, `split_seed`, `normalize`, `captions_dir`; en EEG además `channels`, `source`, `preproc_variant`, `time_window_ms`, `trial_aggregation` |
| `features` | `clip_model` y su preentrenamiento, `vae_model`, `vae_image_size`, `pca_dim`, tamaños de lote del precómputo |
| `model` | `use_lowlevel`, `encoder_type`, y los bloques `fmri_encoder` / `eeg_encoder`, `clip_head`, `lowlevel_head` |
| `training` | `batch_size`, `epochs`, `lr`, `weight_decay`, `scheduler`, `warmup_ratio`, `grad_clip`, `mixed_precision`, `early_stopping_patience`, `num_workers` |
| `losses` | `lambda_cosine`, `lambda_contrastive`, `lambda_lowlevel`, `temperature` |
| `checkpointing` | `monitor` (métrica que elige el mejor modelo), `save_every_n_epochs`, `keep_last_n`, `resume` |
| `evaluation` | `source_experiment`, `checkpoint`, `split`, `conditions`, `run_baselines`, `ridge_alpha`, `noise_std` |
| `generation` | El bloque más extenso: `conditioning_architecture`, `mode`, `text`, `fusion`, `controlnet`, `conditions`, parámetros de muestreo, y todo lo relativo al adaptador |
| `preprocessing` | Solo en `configs/EEG/preproc/`: filtro, referencia, época, línea base, remuestreo, recorte, blanqueo y agregación |

Los valores por defecto están documentados con comentarios en los ficheros
`base.yaml`, que son la referencia autorizada de cada clave.

### 4.5 Cómo crear una configuración nueva

1. **Elige la base correcta.** Para fMRI, `base.yaml`. Para EEG, `base_63.yaml` o
   `base_17.yaml` según el montaje. Si tu variante deriva de otro experimento,
   hereda de ese experimento en lugar de la base, para no repetir bloques.
2. **Pon un `experiment.name` único.** Es lo que determina la carpeta de salida;
   reutilizar un nombre sobrescribe resultados.
3. **Sobrescribe únicamente lo que cambia.** Si el fichero acaba siendo largo,
   probablemente estés heredando de la base equivocada.
4. **Declara las dependencias entre experimentos.** En las configuraciones de
   evaluación, `evaluation.source_experiment` indica de qué experimento se toma el
   punto de control; en las de generación, `generation.decoder_checkpoint` y
   `generation.adapter_checkpoint`.
5. **Comprueba antes de gastar GPU.** Para el condicionamiento multimodal,
   `scripts/15_validate_multimodal.py` verifica en segundos y sobre CPU la
   coherencia de la configuración.

### 4.6 Claves que no pueden cambiarse impunemente

| Si cambias… | Consecuencia |
|---|---|
| `dataset.val_ratio` o `dataset.split_seed` | Cambia la partición: hay que recalcular **todas** las características y reentrenar todo. Es la condición que permite compartir cachés entre variantes |
| `features.clip_model` | Cambia el espacio objetivo: hay que rehacer el precómputo semántico y reentrenar decodificador y adaptador |
| `features.vae_model` o `vae_image_size` | Cambia el espacio latente: hay que rehacer latentes y PCA, y reentrenar la rama estructural y el adaptador |
| `features.pca_dim` | Debe coincidir con `model.lowlevel_head.output_dim` |
| `generation.sd_model` | La dimensión de condicionamiento cambia entre versiones de Stable Diffusion: obliga a reentrenar el adaptador desde cero |
| `generation.conditioning_architecture` o `generation.text.mode` | Definen la identidad del adaptador: un punto de control entrenado con otra combinación **falla al cargarse**, de forma deliberada |
| `generation.mode` | **No** forma parte de la identidad del adaptador: puede cambiarse sin reentrenar |
| `generation.num_tokens` | Forma parte de la identidad del adaptador |
| Variante de preprocesamiento del EEG | Obliga a reconstruir la variante y reentrenar el decodificador, pero **no** a recalcular las características ni a reentrenar el adaptador |

---

## 5. Datos y artefactos en disco

### 5.1 Estructura

```
data/
  processed/
    metadata_<sujeto>.csv              tabla maestra: sujeto, origen, índice, imagen, split
    metadata_<sujeto>.split.json       firma de la partición (invalida la caché si cambia)
    normalization/                     estadísticos ajustados solo con entrenamiento
    pca/                               modelo de PCA y varianza explicada
    eeg_preproc/<variante>/<sujeto>/   variantes del preprocesamiento propio de EEG
  features/
    clip/<modelo>/<sujeto>_<split>.npy       objetivos semánticos
    vae/<modelo>/<sujeto>_<split>_*.npy      latentes y vectores PCA
    text/<modalidad>/<modelo>/<campo>/<hash>/  embeddings de texto deduplicados
    controlnet/<tipo>/<vae>_pca<dim>/          condiciones espaciales en PNG
```

### 5.2 La convención de alineación

Todas las características precalculadas siguen la misma convención: **la fila `k`
de cualquier fichero corresponde al índice `k` dentro del par (sujeto, conjunto)**.
Ese índice es el que la tabla de metadatos asigna a cada imagen.

Es la convención más delicada del sistema, porque romperla no produce un error
sino resultados sin sentido: cada señal quedaría emparejada con la imagen
equivocada. Por eso la política ante datos ausentes es **fallar de forma
explícita** en lugar de descartar filas, ya que un descarte desplazaría todos los
índices posteriores.

En EEG hay un matiz adicional. Las características son **por imagen única**,
mientras que el entrenamiento consume **ensayos individuales**. El *datamodule*
mantiene dos vistas del mismo metadato: una por imagen, que usan el precómputo y
la evaluación, y otra consciente de la agregación, que expande a ensayos y hace
que cada uno reciba el objetivo de *su* imagen.

### 5.3 Salidas de un experimento

```
outputs/<experimento>/
  config.yaml                           configuración resuelta completa
  checkpoints/{last,best,epoch_XXXX}.pt  o adapter_{best,last}.pt en el experimento 4
  logs/{train_log.csv, resume_history.jsonl}
  metrics/*.{json,csv}
  figures/*.png
  embeddings/*.npy   lowlevel/*.npy
  generated/<condición>/*.png
  grids/*.png
  metadata/{generation_params.json, generation_samples.json}
  report/*.md
```

Las métricas agregadas usan formato *tidy*: `metric_name, condition, subject_id,
split, value, seed, checkpoint`. En generación, `metadata/generation_samples.json`
guarda un registro por (muestra, condición) con todo lo necesario para reconstruir
esa imagen exacta: *prompt* resuelto, estado de cada rama, semillas y escala del
control espacial.

### 5.4 Cachés protegidas por firma

Los artefactos costosos están protegidos frente a mezclas silenciosas:

- La **partición** guarda una firma con sus parámetros; si cambian, se reconstruye.
- Las **variantes de preprocesamiento del EEG** guardan un *hash* de su
  configuración; editar un parámetro y reejecutar sin `--force` se **rechaza**.
- La **caché de *embeddings* de texto** se identifica por un *hash* que cubre la
  plantilla, el campo de caption, el tokenizador, el codificador y la longitud
  máxima; si algo cambia, la carga falla con un mensaje accionable.
- Los **puntos de control del adaptador** almacenan su identidad de
  condicionamiento y se niegan a cargarse bajo otra.

---

## 6. Los cinco experimentos

El orden es obligatorio: cada experimento consume artefactos del anterior.

| Experimento | Pregunta | Entradas | Salida principal |
|---|---|---|---|
| **1** | ¿Predice la señal cerebral la representación semántica de la imagen vista? | Señal normalizada + objetivos semánticos | Decodificador entrenado y métricas de recuperación |
| **2** | ¿Depende esa capacidad de usar la señal **correcta**? | Punto de control del experimento 1 o 3 | Métricas por condición, comparación con modelos de referencia y **veredicto** |
| **3** | ¿Predice además estructura de bajo nivel, sin perder semántica? | Lo anterior + objetivos estructurales | Decodificador multitarea y métricas estructurales |
| **4** | ¿Pueden esas predicciones guiar un generador congelado? | Decodificador + adaptador | Imágenes por condición y metadatos |
| **5** | ¿Depende la imagen generada de la señal correcta? | Imágenes del experimento 4 | Métricas por condición, contrastes pareados y **veredicto** |

Los experimentos 1 y 3 **ejecutan la misma función** (`run_training`) con un
booleano distinto: toda la diferencia entre ambos consiste en añadir una segunda
cabeza y un tercer término a la pérdida. El resto —codificador, optimizador,
validación, puntos de control y criterio de mejor modelo— es idéntico, lo que hace
que su comparación sea limpia por construcción.

---

## 7. Bloque de decodificación cerebral

### 7.1 Arquitectura

```
señal cerebral ──▶ codificador ──▶ representación intermedia (2048) ──┬──▶ cabeza semántica  ──▶ 768
                (según modalidad)                                     └──▶ cabeza estructural ──▶ 512
```

El contrato entre codificador y cabezas es una única cifra: la dimensión
intermedia, fijada en 2048 en ambas modalidades. Es lo que permite intercambiar el
codificador sin tocar nada más del sistema, incluido el módulo de adaptación del
bloque generativo.

**Codificador de fMRI.** La señal es un vector plano de decenas de miles de
componentes sin estructura secuencial, por lo que se emplea un perceptrón
multicapa con conexiones residuales. Su regularización incluye un mecanismo
específico: el descarte aleatorio de **vértices completos** durante el
entrenamiento, que impide depender de un subconjunto reducido de localizaciones.

**Codificador de EEG.** La señal posee estructura temporal, por lo que se emplea
una red convolucional temporal con descarte de canales completos, convoluciones
temporales y **agregación temporal por atención**. Esta última hace la
arquitectura **invariante al número de muestras temporales**: la misma red acepta
250, 125, 100 o 50 muestras sin modificación, que es precisamente lo que exigen
las ablaciones de ventana temporal y de frecuencia de muestreo.

Las cabezas son proyecciones deliberadamente pequeñas, para que el trabajo
representacional recaiga en el tronco compartido y ambas tareas se resuelvan desde
una representación común.

### 7.2 Función de pérdida

```
L = λ_cos · (1 − cos(pred, objetivo)) + λ_con · InfoNCE(pred, objetivo) [ + λ_low · MSE(estructural) ]
```

El **término de coseno** empuja la predicción hacia la dirección del objetivo, lo
que es coherente con que la información semántica de estos *embeddings* resida en
su dirección; no penaliza la magnitud, decisión consciente con una consecuencia
que reaparece en el bloque generativo (§8.5).

El **término contrastivo** es el responsable del buen rendimiento en recuperación:
dentro de cada lote construye la matriz de similitudes de todos contra todos y
exige que cada predicción se parezca a *su* imagen **y se diferencie de las
demás**. La distinción importa: con solo el coseno, el modelo puede reducir la
pérdida prediciendo un vector genérico parecido para todas las imágenes —
exactamente el comportamiento del modelo de referencia de la media, que alcanza
similitud coseno alta y rendimiento de azar en recuperación.

El **error cuadrático medio** supervisa la rama estructural cuando está activa.

### 7.3 Aprendizaje multitarea

Se suma un único escalar y se retropropaga una sola vez. Por la topología del
grafo, cada cabeza recibe gradiente solo de su propio término, mientras que el
**tronco recibe la señal combinada de los tres**; ese acoplamiento a través del
tronco compartido es el único mecanismo por el que una tarea puede influir en la
otra, y los pesos de la pérdida regulan la mezcla.

### 7.4 Protocolo de entrenamiento

Idéntico en ambas modalidades y en ambas configuraciones: optimizador AdamW con
planificador coseno y calentamiento, precisión mixta, recorte de norma de
gradiente, techo alto de épocas con **parada temprana**, y —decisión metodológica
relevante— **selección del mejor modelo por métrica de recuperación en validación,
no por la pérdida**, porque esta puede seguir descendiendo mientras la capacidad
discriminativa se degrada. El criterio se mantiene también en la configuración
multitarea, de modo que la rama estructural no pueda condicionar la selección a
costa de la semántica.

La validación de cada época acumula todas las predicciones del conjunto y calcula
la recuperación contra el conjunto **completo** de candidatos, no contra el lote.
Como corolario, un mismo modelo obtiene cifras distintas en validación y prueba
por la simple diferencia en el número de candidatos: **los conjuntos no se
comparan entre sí**, cada uno se compara con su propio azar.

### 7.5 Puntos de control y reanudación

Cada punto de control guarda el estado **completo**: pesos, optimizador,
planificador, escalador de gradiente, época, paso global, mejor métrica, contador
de parada temprana, estados de los generadores aleatorios, configuración resuelta
y versiones de bibliotecas. La reanudación (`--resume`, o `checkpointing.resume:
auto`) continúa el registro sin perder historial y anota el evento en
`logs/resume_history.jsonl`. Los ficheros `last.pt` y `best.pt` nunca se eliminan.

---

## 8. Bloque generativo

### 8.1 El generador congelado y sus tres puertas

En el modelo de difusión conviven **dos espacios que no deben confundirse**: el
**espacio latente**, que es el lienzo sobre el que opera la difusión y que el
autocodificador convierte en píxeles, y el **espacio de condicionamiento**, que es
la secuencia de vectores que la red consulta por atención cruzada y que en el uso
convencional produce el codificador de texto.

La información cerebral puede entrar por **tres puertas**, todas accesibles sin
modificar el generador:

1. **La secuencia de condicionamiento** — vía semántica.
2. **El latente inicial** — vía estructural, partiendo de una reconstrucción
   aproximada en lugar de ruido puro.
3. **Los residuos espaciales** que inyecta una red de control adjunta — vía
   estructural explícita.

### 8.2 El módulo de adaptación

El decodificador produce **un único vector** de 768 dimensiones, mientras que el
generador espera una **secuencia** de vectores en el espacio de salida de un
codificador de texto. Aunque las dimensiones coincidan numéricamente, son espacios
distintos. El módulo de adaptación es el traductor: convierte el vector semántico
en *K* pseudo-tokens que ocupan el lugar de un *prompt*.

Es una red de dos capas y **el único módulo entrenable de la etapa generativa**.
Su entrenamiento merece enunciarse sin ambigüedad, porque es la parte que más se
malinterpreta: **se entrena sin ninguna señal cerebral**. Es una tarea *imagen →
imagen* con el objetivo propio del modelo de difusión —se toma la representación
semántica **real** de una imagen y su latente **real**, se añade ruido, y se pide a
la red congelada que estime ese ruido usando como única pista los pseudo-tokens—.
En inferencia, en cambio, el módulo recibe la representación **predicha desde el
cerebro**.

De esta propiedad se deriva una consecuencia de gran valor experimental: como el
adaptador no depende de la señal cerebral, **un mismo adaptador sirve para todas
las variantes de preprocesamiento** de una modalidad. Reutilizarlo no solo ahorra
la etapa más costosa, sino que refuerza la comparabilidad, ya que con el módulo
idéntico y congelado cualquier diferencia es atribuible al preprocesamiento y al
decodificador.

Los adaptadores **no se comparten entre modalidades**, porque los conjuntos de
imágenes son distintos.

### 8.3 Modos de generación

`generation.mode` decide por qué puerta entra la información estructural:

| Modo | Vía semántica | Vía estructural | Punto de partida |
|---|---|---|---|
| `adapter` | Sí | No | Ruido puro |
| `lowlevel_img2img` | No | Sí (inicialización) | Reconstrucción de la predicción estructural |
| `adapter_lowlevel` | Sí | Sí (inicialización) | Reconstrucción de la predicción estructural |

Todos los experimentos reportados usan `adapter`, que es el modo cuyos contrastes
son atribuibles a una única fuente. En los otros dos, permutar la señal cerebral
altera simultáneamente ambas ramas, de modo que el contraste resultante mide la
contribución cerebral agregada.

### 8.4 Arquitecturas de condicionamiento

Eje **independiente** del anterior, seleccionado con
`generation.conditioning_architecture`:

| Valor | Condición que recibe la red congelada |
|---|---|
| `legacy_adapter` *(por defecto)* | `[K pseudo-tokens]` — solo cerebro, sin caption |
| `text_adapter_concat` | `[77 tokens de texto ; K pseudo-tokens]` por concatenación |
| `text_adapter_concat_controlnet` | lo anterior **más** los residuos de una red de control congelada |

La atención cruzada del generador no depende de la longitud de la secuencia, por
lo que concatenar es legítimo y no requiere modificar el modelo. La red de control
**no sustituye** a la concatenación: texto y pseudo-tokens siguen entrando por
atención cruzada mientras ella añade residuos espaciales construidos a partir de
la predicción estructural.

**Modos de texto.** `generation.text.mode: none | weak | oracle`. El *prompt*
débil es la condición de referencia; el detallado es un techo de rendimiento
declarado, porque un texto muy informativo puede dominar el condicionamiento. En
EEG **`oracle` equivale a `weak`**, ya que el conjunto de datos no dispone de una
descripción más detallada que el nombre del concepto.

**Identidad del adaptador.** Cada punto de control almacena su arquitectura, modo
de texto, campo de caption, plantilla, número de pseudo-tokens y ajustes del
control espacial. Cargarlo bajo otra combinación **falla con un error explícito**,
salvo que se pida lo contrario para una prueba de humo. Esto impide el error más
sutil posible en este sistema: reutilizar un adaptador de otra arquitectura y
obtener resultados aparentemente válidos pero sin sentido.

### 8.5 Dos detalles que conviene conocer

**La norma del vector predicho no está calibrada.** Ningún término de la pérdida
del decodificador supervisa la magnitud, así que deriva entre ejecuciones y, dado
que el adaptador es casi equivariante a la escala de su entrada, actúa como una
intensidad de condicionamiento incontrolada. Hay dos mitigaciones: una calibración
externa (`generation.rescale_clip_pred`) y una **invariancia por construcción**
(`generation.adapter_normalize_input`), en la que el adaptador normaliza su entrada
y la intensidad pasa a ser un parámetro explícito y ajustable en inferencia. Los
experimentos reportados usan la segunda, con lo que el problema queda eliminado.

**La pérdida del adaptador no predice la calidad de generación.** Se calcula sobre
**un único instante** aleatorio del proceso de difusión, mientras que generar
encadena decenas de pasos con guiado; se comprobó empíricamente que puntos de
control con pérdida igual o menor podían generar peor. Por eso el sistema ofrece
tres mecanismos alternativos de selección: promediar la pérdida sobre varios
instantes, evaluar periódicamente la **calidad de generación** sobre un
subconjunto reservado, y barrer los puntos de control guardados puntuando por el
**margen** entre la condición correcta y el mejor control negativo — porque una
mejora igual en todas las condiciones no aporta evidencia de uso de señal cerebral.

---

## 9. Evaluación, controles y métricas

### 9.1 Controles negativos

Se construyen **sobre la señal cerebral, antes del codificador**, dejando
constante todo lo demás (arquitectura, pesos, imágenes, semillas):

- **Permutada** — cada muestra recibe la señal de **otra**, mediante una
  permutación sin puntos fijos que garantiza que ninguna recibe la suya. Es el
  control más informativo, porque conserva la distribución estadística de la señal
  y destruye únicamente su correspondencia con la imagen.
- **Nula** — vector de ceros; mide el sesgo de base del sistema.
- **Ruido** — gaussiano de estadística comparable; distingue «ausencia de señal»
  de «señal sin estructura».

En las arquitecturas multimodales se extienden a las demás fuentes: **texto
permutado** (permutación dentro del mismo conjunto y la misma familia de caption),
**texto genérico** y **estructura permutada o desactivada**.

Dos precisiones que evitan malentendidos: la condición **nula semántica** es una
señal *cerebral* nula que se propaga por el decodificador y produce una
representación concreta, no una condición nula en la entrada del generador; y la
**estructura desactivada** es la red de control a escala cero, no un mapa de
contornos vacío.

Las permutaciones son **reproducibles entre procesos**, con una semilla derivada
de forma estable del nombre de la condición.

### 9.2 Espacio de condiciones

Una condición experimental es una **terna**: (estado del texto, estado de la rama
semántica, estado de la rama estructural). Esa representación es la que permite
expresar toda la matriz de controles sin duplicar el código de generación. El
conjunto disponible depende de la arquitectura: las arquitecturas sin texto o sin
control espacial simplemente no tienen esas columnas.

### 9.3 Métricas

**En decodificación**, la métrica principal es la **recuperación**: se compara la
representación predicha con las de todas las imágenes candidatas y se registra la
posición de la correcta. Su virtud es que tiene un **nivel de azar calculable**
(aproximadamente `k/N` con `N` candidatos), lo que la hace interpretable sin
referencias externas.

Dos advertencias documentadas en el código:

- **La similitud coseno absoluta engaña.** Los *embeddings* empleados son
  anisótropos, de modo que el modelo de referencia de la media alcanza un coseno
  elevado y rinde exactamente a nivel de azar en recuperación.
- **El coeficiente de determinación de la cabeza semántica es negativo por
  diseño**, ya que se optimiza dirección y no magnitud. Solo es informativo para
  la rama estructural.

Los resultados se contrastan con dos modelos de referencia obligatorios: el
**predictor de la media**, que materializa el azar, y una **regresión lineal
regularizada**, que es una referencia exigente.

**En generación**, las imágenes generadas y las reales se codifican con el mismo
modelo congelado y se comparan mediante **similitud semántica** (métrica
principal), recuperación entre generadas y reales, error de píxel y, opcionalmente,
similitud estructural y distancia perceptual.

### 9.4 Contrastes pareados

El resultado generativo no se reporta como valores absolutos, sino como
**diferencias pareadas**. Como todas las condiciones comparten imágenes y **semilla
de difusión**, la diferencia se calcula muestra a muestra y luego se promedia, lo
que elimina la variabilidad debida a la imagen y a la trayectoria de muestreo.
Cada contraste se acompaña de una prueba *t* pareada y de una prueba de Wilcoxon.

Los contrastes definidos separan la contribución de cada fuente: cerebro
(frente a permutada y frente a nula), texto, rama semántica y rama estructural.
**No todos son atribuibles a una única fuente en todas las arquitecturas**: en la
arquitectura con control espacial, permutar el cerebro altera a la vez las dos
ramas, de modo que ese contraste mide la contribución agregada, mientras que los
contrastes por rama sí son limpios.

### 9.5 Política de reporte

Las conclusiones de **decodificación y generación se reportan por separado**,
porque un buen rendimiento en recuperación no implica que la generación dependa de
la señal. Y **los resultados negativos se reportan como tales**, con independencia
de la calidad visual de las imágenes. El veredicto automático compara la condición
correcta con el **mejor** de los controles —no con su media, lo que sobrestimaría
la ventaja— y conviene acompañarlo siempre del tamaño del efecto.

---

## 10. La línea de EEG en detalle

### 10.1 Tratamiento de las repeticiones

Cada imagen se presenta varias veces (cuatro en entrenamiento, ochenta en prueba).
El sistema adopta tratamientos distintos según la etapa, por razones diferentes:

- **En entrenamiento** se usan las repeticiones individuales: cada par
  (imagen, repetición) es un ejemplo independiente, lo que multiplica los ejemplos
  y actúa como aumento de datos con ruido realista.
- **En evaluación y ablaciones** se trabaja siempre a nivel de imagen, promediando
  repeticiones. Esto eleva la relación señal-ruido y, sobre todo, garantiza que los
  candidatos de la recuperación sean **imágenes únicas** y no ensayos duplicados,
  que inflarían el resultado.

### 10.2 Dos rutas de preprocesamiento

**Ruta A — derivados publicados.** Consume los ficheros preprocesados que
acompañan al conjunto de datos, en cualquiera de sus dos montajes, con
normalización por canal ajustada solo con entrenamiento.

**Ruta B — pipeline propio desde la señal cruda.** Reconstruye la cadena completa
a partir de los registros continuos de 63 canales a 1 kHz. Sus etapas, en orden:
control de calidad no destructivo; recuperación de eventos desde el canal de
disparo; filtrado paso banda de fase cero **sobre la señal continua**; referencia
opcional; segmentación en épocas con convención de intervalo semiabierto;
corrección de línea base; remuestreo con protección antialias; recorte temporal;
**blanqueo multivariante del ruido**; y agregación de repeticiones.

Dos detalles metodológicos determinantes:

- La **partición por imagen se calcula antes** de ajustar el blanqueo, de modo que
  este se estima exactamente sobre las imágenes que el entrenamiento considera de
  entrenamiento.
- El blanqueo se aplica **a cada repetición individual, antes de promediarlas**,
  porque estima la covarianza del ruido y el promediado es precisamente lo que la
  destruye.

La salida usa **el mismo contrato de ficheros que los derivados oficiales**, de
modo que el *datamodule*, el codificador y los cinco experimentos la consumen sin
cambios.

> ⚠️ El paso de preprocesamiento requiere el intérprete del entorno virtual del
> proyecto, donde está instalado MNE. La configuración de referencia fija el
> *backend* del filtro a MNE precisamente para que un intérprete equivocado falle
> en un par de segundos con un mensaje accionable, en lugar de degradar en
> silencio a otro filtro y producir una variante no comparable con las demás.

### 10.3 Ablaciones de preprocesamiento

Diez variantes de un solo factor, bajo tres reglas: **un factor por variante**;
**ramificación desde la señal cruda**, de modo que ninguna variante se construya
sobre otra y el factor manipulado sea realmente el único que cambia; y **recálculo
de los estadísticos dependientes** cuando el factor altera la representación.

Los factores manipulados son: presencia del blanqueo, número de electrodos,
ventana temporal (dos variantes), frecuencia de muestreo, banda de frecuencia,
agregación de repeticiones en entrenamiento, referencia y ventana de línea base
(dos variantes).

### 10.4 Qué reejecutar al cambiar de preprocesamiento

| Paso | ¿Reejecutar? | Motivo |
|---|---|---|
| Construcción de la variante y preparación de metadatos | **Sí** | El tensor y la normalización son específicos de la variante |
| Precómputo semántico y estructural | **No** | Dependen solo de las imágenes y de la partición |
| Experimentos 1, 2 y 3 | **Sí** | Cambia la entrada cerebral |
| Entrenamiento del adaptador | **No** | El adaptador nunca ve la señal cerebral |
| Generación y experimento 5 | **Sí** | Cambia el decodificador |

Se mantiene mientras no cambien la proporción de validación ni la semilla de
partición.

---

## 11. Reproducibilidad y validación del sistema

- **Semillas** fijadas para Python, NumPy y PyTorch, y semillas estables para las
  permutaciones de los controles.
- **Partición compartida**: la calcula una única función usada tanto por el
  preprocesamiento como por el *datamodule*, lo que elimina la posibilidad de que
  un estadístico se ajuste sobre un conjunto distinto del que el entrenamiento
  llama entrenamiento.
- **Configuración resuelta y versiones de bibliotecas** guardadas en cada punto de
  control y en cada carpeta de salida.
- **Metadatos de generación** suficientes para reconstruir cualquier imagen.
- **Verificación automática del propio sistema**, que es lo que permite afirmar que
  las comparaciones son limpias:
  - `scripts/10_validate_eeg_preproc.py` comprueba las formas exactas de cada
    variante (incluida la convención semiabierta, que evita errores de una
    muestra), la **ausencia de fuga** en el blanqueo —alterar los datos reservados
    no debe modificar la matriz estimada—, la disyunción de las particiones y el
    recuento de repeticiones.
  - `scripts/15_validate_multimodal.py` comprueba la retrocompatibilidad, las
    formas del condicionamiento, la alineación y permutación de los captions, la
    equivalencia exacta entre desactivar el control espacial y no usarlo, y la
    compatibilidad de los puntos de control del adaptador.
  - Existen además patrones de prueba de humo sin GPU con datos sintéticos, en los
    que el veredicto automático **debe** declarar ausencia de uso de señal
    cerebral: confirman que la maquinaria de evaluación no produce falsos
    positivos.

---

## 12. Coste computacional

El sistema está dimensionado para una única estación con GPU de gama media (16 GB
de memoria de vídeo) y 32 GB de RAM. Las decisiones que lo hacen viable son el
precómputo de todas las salidas de modelos congelados, la precisión mixta y la
segmentación de la atención y del autocodificador durante la generación.

| Etapa | Coste aproximado |
|---|---|
| Preparación de datos y particiones | segundos |
| Precómputo semántico | minutos |
| Precómputo estructural | decenas de minutos; es el artefacto más grande en disco |
| Construcción de una variante de preprocesamiento de EEG | decenas de minutos, ≈ 2 GB por sujeto y variante |
| Entrenamiento del decodificador | decenas de minutos, con parada temprana |
| Evaluación con ablaciones | ≈ 1 minuto |
| **Entrenamiento del adaptador** | **horas** — etapa dominante |
| Generación y evaluación generativa | minutos por condición |

Que la etapa dominante sea el adaptador es lo que hace tan valiosa su propiedad de
independencia respecto a la señal cerebral: reutilizarlo entre variantes es lo que
convierte el estudio de ablaciones en algo abordable.

> ⚠️ Cada punto de control del decodificador de fMRI ocupa del orden de 2,7 GB
> (pesos más estados del optimizador). Conviene ajustar la frecuencia de guardado
> y el número de puntos retenidos para no llenar el disco.

---

## 13. Problemas frecuentes y diagnóstico

| Síntoma | Causa habitual y solución |
|---|---|
| «No se encuentran características precalculadas» en un cuaderno | El directorio de trabajo no es la raíz del repositorio. Ejecuta la primera celda del cuaderno, que lo corrige |
| El preprocesamiento de EEG falla inmediatamente mencionando el *backend* del filtro | Se está usando el intérprete equivocado; MNE está en el entorno virtual del proyecto |
| Se rechaza reutilizar una variante de preprocesamiento | Su firma de configuración cambió. Es intencionado: reejecuta con `--force` si de verdad quieres reconstruirla |
| Error al cargar un punto de control del adaptador | Su identidad de condicionamiento no coincide con la configuración actual. Es intencionado: usa el adaptador correspondiente |
| La carga de *embeddings* de texto falla | Cambió la plantilla, el campo de caption o el tokenizador. Reejecuta el precómputo de texto |
| Falta memoria de vídeo al generar | Reduce el tamaño de lote de generación, activa la descarga a CPU o baja el número de muestras |
| Un resultado parece demasiado bueno | Comprueba el nivel de azar del conjunto y revisa los controles. La recuperación depende del número de candidatos, así que validación y prueba no son comparables entre sí |

---

## 14. Configuraciones y experimentos reportados

Los resultados de la memoria corresponden a **siete configuraciones generativas**,
todas con `generation.mode: adapter`, y a sus experimentos de decodificación
asociados.

| Código | Modalidad y línea | Arquitectura de condicionamiento |
|---|---|---|
| **Af** | fMRI | A · solo señal cerebral |
| **Bf** | fMRI | B · texto débil + señal |
| **Cf** | fMRI | C · texto + señal + control espacial |
| **Ae** | EEG, derivado oficial de 63 canales | A · solo señal cerebral |
| **Be** | EEG, derivado oficial de 63 canales | B · texto débil + señal |
| **Ce** | EEG, derivado oficial de 63 canales | C · texto + señal + control espacial |
| **Ct** | EEG, preprocesamiento propio, ventana 100–600 ms | C · texto + señal + control espacial |

Se emplean **seis adaptadores**, tres por modalidad —uno por arquitectura—, ya que
no se comparten entre modalidades. La única reutilización es **Ct tomando el
adaptador de Ce**: ambas son EEG, arquitectura C y texto débil, y difieren
únicamente en la variante de preprocesamiento, que el adaptador nunca ve. Esa
reutilización mantiene la etapa generativa idéntica entre ambas líneas, de modo que
cualquier diferencia sea atribuible al preprocesamiento y al decodificador.

La correspondencia exacta entre estos códigos y los ficheros de configuración, así
como los comandos de ejecución, están en la sección 4 del
[`README.md`](../README.md).
