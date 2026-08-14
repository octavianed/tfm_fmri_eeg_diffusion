# Documento 1 — Objetivos del proyecto y experimentos

> Proyecto: **decodificación y reconstrucción visual a partir de fMRI** (NSD Algonauts 2023),
> como línea nueva del TFM (que hasta ahora usaba EEG). Este documento explica *qué* se
> quiere conseguir, *qué significa cada concepto* y *qué hace cada experimento* y cómo se
> relacionan con el código. El Documento 2 cubre la parte técnica (código, ejecución,
> salidas, métricas, tiempos y resultados).

---

## 1. La idea en una frase

Queremos comprobar, de forma **honesta y falsable**, si la actividad cerebral (fMRI) que se
registra mientras una persona mira una imagen contiene información suficiente para (a)
**predecir una representación visual** de esa imagen y (b) **reconstruir** una imagen
parecida — y hacerlo sin engañarnos a nosotros mismos.

---

## 2. Motivación: por qué está diseñado así

En la línea de EEG del TFM apareció un problema clásico en reconstrucción cerebro→imagen:
**la condición cerebral apenas afectaba a la salida**. Es decir, se generaban imágenes,
pero al sustituir la señal real por una señal incorrecta o nula se obtenían resultados
parecidos. Eso significa que el modelo **no estaba usando de verdad la señal cerebral**;
se apoyaba en el generador (Stable Diffusion) y en priors del propio modelo.

Para evitar repetir ese error, este proyecto **separa explícitamente dos problemas** que
suelen mezclarse:

1. **Decodificación cerebral** (¿la fMRI predice una representación de la imagen real?).
   Es medible con métricas objetivas y **no** necesita generar imágenes.
2. **Generación visual** (usar esas predicciones para guiar un generador **congelado**).

Y sobre todo, introduce **controles negativos** en todas las fases: si la señal correcta no
supera claramente a una señal permutada o nula, **no se afirma** que el modelo use
información cerebral real. Esta es la diferencia fundamental respecto al enfoque EEG previo.

---

## 3. Glosario: qué significa cada cosa

### Datos y cerebro
- **fMRI (resonancia magnética funcional)**: mide de forma indirecta la actividad neuronal
  a través del flujo sanguíneo (señal BOLD). Aquí **no** es una serie temporal por canales
  (como el EEG), sino **una respuesta espacial por imagen**: un vector de amplitudes.
- **Vóxel / vértice**: unidad espacial de la medida. NSD Algonauts entrega la respuesta
  proyectada a la superficie cortical (vértices) del atlas *fsaverage*, restringida a la
  corteza visual (“challenge space”). Para `subj01` son **19 004 vértices** en el hemisferio
  izquierdo (LH) y **20 544** en el derecho (RH) → **39 548 “features” de fMRI por imagen**.
- **NSD (Natural Scenes Dataset)**: dataset de fMRI 7T de 8 sujetos viendo imágenes
  naturales de COCO. **Algonauts 2023** es su versión preprocesada para un reto público.
- **Hemisferios (lh, rh)**: se cargan los dos y se **concatenan** en un único vector por
  imagen.
- **Split train/val/test**: partición de los datos. La normalización y cualquier ajuste
  (PCA, baselines) se hacen **solo con train** para evitar *leakage* (contaminación).
- **Normalización por vóxel**: a cada vértice se le resta su media y se divide por su
  desviación típica, calculadas **solo en train**. Deja la fMRI centrada (media ≈ 0,
  desviación ≈ 1) y comparable entre vértices.

### Representaciones visuales (los “objetivos” a predecir)
- **Embedding CLIP de imagen**: CLIP es un modelo (congelado) que convierte una imagen en
  un vector (aquí de **768 dimensiones**, modelo `ViT-L-14`) que captura su **contenido
  semántico de alto nivel** (“qué hay en la escena”). Es el objetivo principal a predecir
  desde la fMRI.
- **Latente del VAE de Stable Diffusion**: el VAE (congelado) comprime una imagen 512×512 a
  un tensor pequeño `[4, 64, 64]` (16 384 números) que captura **estructura de bajo nivel**
  (composición, color, layout). 
- **PCA (Análisis de Componentes Principales)**: reduce esos 16 384 números a un vector de
  **512 dimensiones** quedándose con las direcciones de mayor varianza. Es el objetivo
  “low-level” a predecir. La PCA se **ajusta solo con train**.

### Generación
- **Stable Diffusion (SD) congelado**: generador de imágenes. **No se reentrena**. Se le
  guía con las representaciones predichas desde la fMRI.
- **TokenAdapter**: pequeño módulo (lo **único** entrenable en generación) que traduce el
  embedding CLIP predicho a los “pseudo-tokens” que la U-Net congelada de SD espera como
  condición (`prompt_embeds`). 
- **img2img**: variante de generación que parte de una imagen/latente inicial (aquí, el
  latente de bajo nivel reconstruido desde la fMRI) en lugar de ruido puro.

### Métricas y evaluación
- **Retrieval (recuperación)**: dada la predicción de la fMRI para una imagen, la comparamos
  (por coseno) con los embeddings CLIP de **todas** las imágenes candidatas de test y miramos
  en qué **puesto (rank)** queda la correcta.
  - **Top-1 / Top-5 / Top-10**: fracción de casos en los que la imagen correcta está entre
    las 1 / 5 / 10 más parecidas. Cuanto mayor, mejor.
  - **mean rank / median rank**: puesto medio/mediano de la imagen correcta (1 = perfecto).
  - **azar**: con `N` candidatos, Top-k por azar ≈ `k/N`; rank medio por azar ≈ `N/2`.
- **Cosine similarity (coseno)**: parecido de dirección entre predicción y objetivo.
  ⚠️ Los embeddings CLIP son **anisótropos** (todos apuntan a una zona común), así que un
  coseno “alto” puede ser engañoso; el **retrieval** es la métrica honesta.
- **InfoNCE (pérdida contrastiva)**: durante el entrenamiento empuja a que la predicción de
  cada fMRI se parezca a **su** imagen y se diferencie del resto del lote. Es lo que produce
  buen retrieval.
- **Pearson por componente / R²**: calidad de regresión de la predicción (útil sobre todo
  para la rama low-level).

### Controles (lo más importante del proyecto)
- **fMRI correcto**: la respuesta cerebral real asociada a cada imagen.
- **fMRI permutado**: a cada muestra se le da la fMRI de **otra** muestra (nunca la suya),
  con un *derangement* de Sattolo (permutación sin puntos fijos). Debe rendir **como el azar**.
- **fMRI cero**: vector de ceros. Debe rendir **como el azar**.
- **fMRI ruido**: ruido gaussiano con estadística similar. Control adicional.
- **Baselines**: modelos simples de referencia:
  - **media**: predice siempre el embedding medio de train (→ retrieval a nivel de azar).
  - **ridge**: regresión lineal regularizada fMRI→CLIP. En fMRI suele ser un baseline
    **fuerte**; si tu modelo no lo supera, no aporta gran cosa.

---

## 4. La hipótesis falsable (el corazón del proyecto)

Todo se organiza alrededor de esta comparación:

```
fMRI correcto  →  representación predicha claramente mejor que el azar
fMRI permutado →  ~ azar
fMRI cero      →  ~ azar

Condición de éxito:   correcto  ≫  permutado ≈ cero
```

- **Si se cumple**: hay evidencia de que la fMRI aporta información específica de cada
  imagen; el modelo está “leyendo” el cerebro.
- **Si NO se cumple**: aunque se generen imágenes bonitas, **no** se puede atribuir el
  resultado a la señal cerebral, y así debe indicarse en la memoria. El código lo deja
  escrito automáticamente (`conclusion.json`, `report/*.md`).

Esto convierte “¿funciona?” en una pregunta con respuesta objetiva, en lugar de una
impresión visual.

---

## 5. Los cinco experimentos

El orden es **obligatorio**: cada uno prepara al siguiente. Para cada uno indico la pregunta
que responde, qué produce y a qué parte del código corresponde.

### Experimento 1 — fMRI → embedding CLIP
- **Pregunta**: ¿la fMRI contiene información suficiente para aproximar el embedding CLIP
  (semántico) de la imagen vista?
- **Qué hace**: entrena un `fMRIEncoder` + `CLIPHead` para predecir, desde el vector de
  fMRI, el embedding CLIP de la imagen. Stable Diffusion **no** se carga.
- **Pérdida**: coseno (1 − cos) + InfoNCE contrastiva.
- **Código**: `scripts/02_train_fmri_to_clip.py` / `notebooks/01_...` →
  `src/training/train_clip_decoder.py` → `run_training(use_lowlevel=False)`.
  Usa `src/data` (datos), `src/models` (red), `src/losses` (pérdidas),
  `src/evaluation/retrieval_metrics.py` (validación).
- **Éxito**: retrieval en test claramente por encima del azar y del baseline de media.

### Experimento 2 — Retrieval y ablaciones (correcto / permutado / cero)
- **Pregunta**: ¿la mejora **depende** de usar la fMRI correcta, o el modelo rinde igual con
  señal incorrecta/nula?
- **Qué hace**: carga el modelo de Exp1 y evalúa retrieval en test bajo las condiciones
  correcto/permutado/cero (+ruido), y compara con baselines media y ridge.
- **Código**: `scripts/03_eval_retrieval_ablation.py` / `notebooks/02_...` →
  `src/evaluation/ablation_eval.py` (`evaluate_ablation`, `conclusion_from_summary`) +
  `src/evaluation/baselines.py`.
- **Éxito**: `correcto ≫ permutado ≈ cero`. **Este es el experimento decisivo.**

### Experimento 3 — Rama low-level (PCA de latentes VAE) + multitarea
- **Pregunta**: ¿la fMRI puede además predecir información visual de **bajo nivel**
  (estructura/color) complementaria a la semántica?
- **Qué hace**: añade una `LowLevelHead` que predice el vector PCA del latente del VAE, y
  entrena un modelo **multitarea** (CLIP + low-level). Antes hay que precomputar latentes
  VAE + PCA.
- **Pérdida**: coseno + InfoNCE (CLIP) + `lambda_lowlevel` · MSE (low-level).
- **Código**: `scripts/04_precompute_vae_pca.py` (features) y
  `scripts/05_train_multitask.py` / `notebooks/03_...` →
  `src/training/train_multitask_decoder.py` → `run_training(use_lowlevel=True)`.
- **Éxito**: mantiene o mejora el retrieval CLIP **y** predice el low-level mejor que un
  baseline medio (R² > 0, Pearson > 0).

### Experimento 4 — Generación con Stable Diffusion congelado
- **Pregunta**: ¿las representaciones decodificadas desde la fMRI pueden guiar la generación
  de una imagen visualmente relacionada con la que se vio?
- **Qué hace**: entrena el **TokenAdapter** (único módulo entrenable; U-Net/VAE/CLIP
  congelados) y genera imágenes a partir de los embeddings predichos, para las condiciones
  correcto/permutado/cero, con semillas fijas.
- **Código**: `scripts/06_generate_images.py` / `notebooks/04_...` →
  `src/generation/sd_pipeline.py` (`train_token_adapter`, `FrozenSDGenerator`) +
  `src/generation/generate_from_fmri.py`.
- **Éxito**: las imágenes con fMRI correcta deberían parecerse más a la real que las de los
  controles (se mide en Exp5).

### Experimento 5 — Comparación generativa final
- **Pregunta**: ¿la **calidad/contenido** de la imagen generada cambia de forma
  significativa al usar la fMRI correcta frente a permutada/cero?
- **Qué hace**: calcula métricas de similitud (CLIP, opcional SSIM/LPIPS) entre generada y
  real por condición, hace tests estadísticos y crea rejillas comparativas y de casos
  mejores/medios/peores.
- **Código**: `scripts/07_eval_generation_ablation.py` / `notebooks/05_...` →
  `src/evaluation/generation_metrics.py` + `src/generation/make_grids.py`.
- **Éxito**: la condición correcta supera a los controles en similitud CLIP y muestra
  diferencias cualitativas; si no, se interpreta como que el generador no usa suficientemente
  la fMRI.

---

## 6. El pipeline completo (visión de conjunto)

```
                IMAGEN REAL (estímulo)
                 │                        │
   CLIP (congelado)                VAE SD (congelado)
   embedding 768  ─ objetivo alto        latente [4,64,64] ─ PCA(train) ─ objetivo bajo (512)
                 ▲                        ▲
                 │  (se aprenden a predecir)
                 │
   fMRI [39.548] ─► fMRIEncoder ─► h ─┬─► CLIPHead ─► embedding predicho   (Exp1, Exp3)
                                      └─► LowLevelHead ─► PCA predicho      (Exp3)

   EVALUACIÓN (Exp2):  retrieval correcto/permutado/cero  →  ¿correcto ≫ controles?

   GENERACIÓN (Exp4):  embedding predicho ─► TokenAdapter ─► SD congelado ─► imagen
                       (opcional) PCA predicho ─► inverse PCA ─► latente ─► img2img

   COMPARACIÓN (Exp5): imagen real vs correcto/permutado/cero  →  similitud CLIP
```

- **Alto nivel (CLIP)** = “qué” hay en la imagen. **Bajo nivel (VAE/PCA)** = “cómo” está
  compuesta. Un buen sistema idealmente acierta ambos.
- Todo lo caro (CLIP, VAE, SD) va **congelado** y sus salidas se **precomputan** a disco,
  para que sea viable en 16 GB de VRAM / 32 GB de RAM.

---

## 7. Qué se puede concluir para el TFM

- Exp1/Exp2 permiten afirmar (con métricas) **si** la fMRI de un sujeto codifica el contenido
  visual de forma recuperable, y **cuánto** por encima del azar y de un baseline lineal.
- Exp3 añade la dimensión de bajo nivel y comprueba que no se rompe la semántica.
- Exp4/Exp5 aportan la parte **cualitativa/generativa**, pero **subordinada** a que Exp2 haya
  dado positivo: solo si `correcto ≫ controles` tiene sentido atribuir las reconstrucciones a
  la señal cerebral.

En resumen, el proyecto está diseñado para que la conclusión del TFM sea **defendible**: no
“hemos generado imágenes”, sino “hemos demostrado, con controles, que la fMRI aporta
información visual específica, y la hemos usado para guiar una reconstrucción”.

*(Los valores concretos que ya has obtenido y su interpretación están en el Documento 2,
sección 8.)*
