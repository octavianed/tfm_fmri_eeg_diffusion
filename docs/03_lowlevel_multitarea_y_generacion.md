# Documento 3 — Bajo nivel (VAE+PCA), Experimento 3 (multitarea), métricas y modos de generación

> Amplía los Documentos 1 y 2 con el detalle de: (a) qué hace el precómputo VAE+PCA, (b) qué
> es el Experimento 3 y su pérdida multitarea, (c) qué significan las métricas (Pearson, R²…),
> (d) qué hacen los tres `mode` de generación, y (e) con qué configuración (y cuántas épocas)
> entrenar cada experimento.

---

## 1. Paso 4 — Precomputar latentes VAE + PCA (`04_precompute_vae_pca.py`)

### 1.1 ¿Qué se quiere conseguir?

Hasta aquí (Exp1/Exp2) solo predecimos el **embedding CLIP**, que captura el **contenido
semántico** ("qué hay en la imagen"). Pero una imagen también tiene **información de bajo
nivel**: estructura espacial, composición, colores, dónde están las cosas. El objetivo del
Paso 4 es construir un **objetivo numérico de bajo nivel** que la fMRI pueda intentar predecir,
para luego (Exp3) comprobar si la fMRI también codifica esa parte.

Ese objetivo se deriva del **VAE (autoencoder variacional) de Stable Diffusion**, que es la
red que SD usa internamente para comprimir imágenes. No lo entrenamos: lo usamos **congelado**
como un "extractor de rasgos de bajo nivel".

### 1.2 Qué hace el script, paso a paso

El script encadena dos etapas (dos funciones de `src/features/`):

**Etapa A — Codificar imágenes a latentes VAE** (`precompute_vae_latents.py`)
1. Carga el VAE de SD-1.5 (congelado).
2. Para cada imagen (train/val/test), la redimensiona a **512×512** y la normaliza al rango
   `[-1, 1]` (lo que el VAE espera).
3. La pasa por el **encoder del VAE**, que devuelve un **latente** de forma `[4, 64, 64]`
   (4 canales, 64×64 = la imagen "comprimida" ×8 en cada dimensión). Se toma la media de la
   distribución latente y se multiplica por el factor de escala de SD (`0.18215`), que es la
   convención estándar de Stable Diffusion.
4. Ese latente se **aplana** a un vector de `4·64·64 = 16 384` números y se guarda.
   - Salida: `data/features/vae/<modelo>/subjNN_{train,val,test}_latents.npy` (float32
     `[N, 16384]`). En subj01, el de train ocupa ~**550 MB**.
   - También un `subjNN_vae_meta.json` con el factor de escala y la forma del latente (se
     necesita luego para "deshacer" la PCA en generación, ver §4).

**Etapa B — Ajustar PCA solo con train** (`fit_vae_pca.py`)
1. 16 384 dimensiones son demasiadas para que la fMRI las prediga directamente (y muchas son
   redundantes). Aplicamos **PCA (Análisis de Componentes Principales)**, que encuentra las
   `pca_dim = 512` **direcciones de mayor varianza** y proyecta cada latente a un vector de
   512 números.
2. La PCA se **ajusta usando solo el split de train** (para evitar *leakage*); luego se usa
   esa misma PCA para **proyectar** train, val y test.
   - Salida: `data/features/vae/<modelo>/subjNN_{split}_pca.npy` (float32 `[N, 512]`) — este
     es el **objetivo de bajo nivel** que la fMRI aprenderá a predecir en Exp3.
   - Modelo PCA: `data/processed/pca/subjNN_vae_pca_model.pkl` (contiene la PCA + forma del
     latente + factor de escala + varianza explicada).
   - `subjNN_vae_pca_model.evr.json`: varianza explicada por componente (para la figura del
     notebook 03).

### 1.3 Varianza explicada (cómo leerla)

La PCA no conserva toda la información: se queda con la de mayor varianza. La **varianza
explicada acumulada** dice qué fracción de la variabilidad original capturan las primeras K
componentes. En subj01 (512 componentes de 16 384):

| componentes | varianza acumulada |
|---|---|
| 10 | 29% |
| 100 | 47% |
| 256 | 53% |
| 512 | **58,5%** |

Interpretación: las primeras componentes concentran la mayor parte de la señal "gruesa"
(estructura/color global); el resto de varianza (41%) es detalle de alta frecuencia que la PCA
truncada descarta. Es un compromiso razonable: bastante información de bajo nivel en un vector
manejable de 512 números.

### 1.4 Coste

- **Tiempo**: ~10–25 min en GPU (el VAE a 512 px es más pesado que CLIP) + ~1 min de PCA.
- **Disco**: los latentes ocupan bastante (~550 MB/sujeto en train). Si te preocupa el
  espacio, puedes borrar los `*_latents.npy` **después** de ajustar la PCA y de entrenar el
  adapter (la PCA `*_pca.npy` y el `.pkl` son mucho más pequeños). Ojo: el entrenamiento del
  adapter (Exp4) **sí** usa los `*_latents.npy`, así que no los borres antes de eso.

---

## 2. Experimento 3 — Multitarea (`05_train_multitask.py` / notebook 03)

### 2.1 ¿Qué se quiere conseguir?

Responder: **¿la fMRI puede predecir, además del contenido semántico (CLIP), la información
visual de bajo nivel (estructura/color)?** Y hacerlo **sin estropear** lo que ya funcionaba
(el retrieval CLIP).

### 2.2 Qué cambia respecto al Experimento 1

Es **el mismo modelo y el mismo bucle de entrenamiento** que Exp1, pero el `fMRIEncoder` ahora
tiene **dos cabezas** en lugar de una:

```
fMRI ─► fMRIEncoder ─► h ─┬─► CLIPHead     ─► embedding CLIP predicho   (alto nivel, como Exp1)
                          └─► LowLevelHead ─► vector PCA predicho (512)  (bajo nivel, NUEVO)
```

Se activa con `model.use_lowlevel: true` en el config, y la función es la misma:
`run_training(use_lowlevel=True)` en `src/training/train_multitask_decoder.py`.

### 2.3 La pérdida multitarea

Se combinan tres términos (definidos en `src/losses/multitask_losses.py`):

```
L_total =  λ_cosine      · (1 − cos(clip_pred, clip_target))     # dirección del embedding CLIP
        +  λ_contrastive · InfoNCE(clip_pred, clip_target)        # que cada fMRI "encuentre" su imagen
        +  λ_lowlevel    · MSE(low_pred, low_target)              # ajuste del vector PCA de bajo nivel
```

- Los dos primeros términos son **exactamente** los de Exp1 (aprenden el CLIP).
- El tercero es nuevo: un **error cuadrático medio (MSE)** entre el vector PCA predicho y el
  real. `λ_lowlevel` (por defecto **0,25**) controla **cuánto peso** damos al bajo nivel:
  - más alto (p. ej. 0,5) → prioriza el bajo nivel, puede bajar un poco el retrieval CLIP;
  - más bajo (p. ej. 0,1) → apenas afecta al CLIP, pero aprende menos bajo nivel.

El **criterio del "mejor" checkpoint sigue siendo el retrieval CLIP** (`monitor: val_top5`),
porque el objetivo primario es no perder la semántica.

### 2.4 Qué salidas produce (además de las de Exp1)

- `metrics/{val,test}_lowlevel_metrics.json` — métricas de la rama de bajo nivel (ver §3).
- `lowlevel/{val,test}_low_{pred,target}.npy` — vectores PCA predichos y reales.
- `figures/pca_component_correlation_{val,test}.png` — correlación de Pearson por componente.

### 2.5 Cómo interpretar el resultado (con tus números reales, subj01 test)

- **Retrieval CLIP**: Top-5 **89,3%** en Exp3 vs 86,8% en Exp1 → añadir la rama de bajo nivel
  **no perjudicó** (incluso mejoró levemente) la semántica. ✔ Objetivo cumplido.
- **Rama de bajo nivel**: **R² = 0,225 (>0)**, Pearson medio = 0,052. Es decir: la fMRI
  **sí** predice algo de bajo nivel (mejor que el baseline de la media), aunque de forma
  **mucho más débil** que la semántica. Esto es **esperable y honesto**: reconstruir
  estructura/color desde fMRI es más difícil y ruidoso que reconocer el "qué".

### 2.6 ¿Cómo funciona un solo modelo con dos cabezas? (tronco compartido y gradiente conjunto)

Esta es la parte conceptual clave y explica por qué añadir la rama VAE pudo **mejorar también**
la predicción CLIP.

**Un solo tronco, dos cabezas.** El `forward` del modelo
(`src/models/multitask_decoder.py`) es literalmente:

```python
def forward(self, fmri, subject=None):
    h = self.encode(fmri, subject)              # TRONCO compartido: fMRI [B,39548] -> h [B,2048]
    out = {"h": h, "clip": self.clip_head(h)}       # cabeza 1 lee de h
    if self.low_head is not None:
        out["low"] = self.low_head(h)               # cabeza 2 lee del MISMO h
    return out
```

Ambas cabezas parten del **mismo** vector `h`. El `fMRIEncoder` (la parte grande, ~230M de
parámetros) es el **mismo** que en Exp1; solo se le cuelga una segunda cabeza pequeña
(`LowLevelHead`, una MLP `h`→512):

```
                          ┌─► CLIPHead    ─► clip_pred (768)
fMRI ─► fMRIEncoder ─► h ─┤
   (TRONCO, compartido)    └─► LowLevelHead ─► low_pred (512)
```

Por eso decimos "mismo bloque y mismo bucle": `run_training` no cambia; solo se activa
`use_lowlevel=True`, que añade la segunda cabeza y el tercer término de la pérdida.

**La pérdida es CONJUNTA (un solo escalar, un solo `backward`).** No se entrena cada cabeza por
separado ni en pasos alternos:

```python
total = λ_cosine·L_cos + λ_contrastive·L_infonce + λ_lowlevel·L_mse   # un escalar
total.backward()                                                       # una sola retropropagación
```

Ahora bien, **cada parámetro recibe gradiente solo de los términos de los que depende** (es
automático, por la topología del grafo de cómputo):

| Parámetros | Reciben gradiente de… | Por qué |
|---|---|---|
| `clip_head` | solo `L_cos` + `L_infonce` | `low_pred` no depende de `clip_head` |
| `low_head` | solo `L_mse` | los términos CLIP no dependen de `low_head` |
| **`fMRIEncoder` (tronco)** | **de los TRES términos (la suma)** | `h` alimenta a **ambas** cabezas |

Es decir: las **cabezas** están desacopladas entre sí (cada una se ajusta con su término), pero
el **tronco** recibe una señal **combinada**. Los `λ` reescalan cuánto pesa cada término en esa
suma; en particular controlan la mezcla de gradientes que llega al tronco. Ese acoplamiento **a
través del tronco compartido** es el mecanismo por el que una tarea puede influir en la otra.

**Por qué añadir la rama VAE puede ayudar al CLIP** (efecto de tarea auxiliar / regularización):

1. **Regularización contra el sobreajuste (la razón principal aquí).** El tronco es enorme
   (~230M parámetros) frente a ~8 857 muestras de train; con una sola cabeza CLIP tiende a
   memorizar atajos que ajustan train pero no generalizan (se vio: el mejor `val` estaba en la
   época 3). Exigirle **además** predecir el bajo nivel invalida esos atajos y empuja a `h` a
   capturar información visual más general → mejor generalización también del CLIP.
2. **Compartir "fuerza estadística".** Ambos objetivos vienen de la misma imagen; aprender a
   extraer estructura/color puede ayudar al encoder a limpiar el ruido de la fMRI y quedarse
   con rasgos estables, algunos útiles también para la semántica.
3. **Gradiente denso y bien condicionado.** El MSE de bajo nivel (sobre vectores PCA centrados)
   aporta una señal estable que complementa a la de CLIP (más direccional), guiando mejor al
   encoder, sobre todo al principio.

**Matiz honesto (importante para la memoria).** La mejora fue **pequeña**: test Top‑5 86,8%→89,3%
y Top‑1 47,2%→52,2%. Con solo **159 muestras de test**, cada imagen vale ~0,63 puntos, así que
ese +5 en Top‑1 son ~8 imágenes: es **consistente con "no perjudica y probablemente ayuda un
poco"**, pero con margen de ruido. Para afirmarlo con solidez conviene (a) comprobar que también
se mantiene/mejora en `val` (984 candidatos, más estable) y/o (b) repetir con 2–3 semillas y
comparar medias. Lo que **sí** puedes afirmar con confianza es lo esencial del Exp3: **la rama
de bajo nivel no rompe la semántica** (R² low‑level > 0 y retrieval CLIP igual o mejor).

---

## 3. Métricas explicadas (con Pearson en detalle)

Recordatorio: para la cabeza **CLIP** la métrica honesta es el **retrieval** (Top-k, rank);
el coseno absoluto y el R² del CLIPHead engañan (ver Documento 2 §7). Para la rama
**low-level** (objetivo PCA, entrenado con MSE) las métricas de regresión **sí** son
informativas. Las que se calculan (`src/evaluation/embedding_metrics.py`):

- **MSE (error cuadrático medio)** = media de `(predicho − real)²`. Cuanto menor, mejor.
  Sensible a la escala de los datos, así que su valor absoluto solo tiene sentido comparándolo
  con un baseline.
- **MAE (error absoluto medio)** = media de `|predicho − real|`. Como el MSE pero menos
  sensible a valores extremos.
- **R² (coeficiente de determinación)** = `1 − SS_res/SS_tot`, donde `SS_res` es el error del
  modelo y `SS_tot` la varianza de los datos.
  - **R² = 1**: predicción perfecta. **R² = 0**: igual que predecir siempre la media (inútil).
    **R² < 0**: peor que predecir la media.
  - Para el bajo nivel, **R² > 0 ya es un resultado válido** (predice mejor que la media). Tu
    0,225 significa que explica ~22% de la varianza del objetivo PCA.
- **Pearson por componente** = para **cada una** de las 512 dimensiones PCA, la **correlación
  de Pearson** entre la columna predicha y la real, **a lo largo de las N imágenes de test**.
  - La correlación de Pearson `r` mide si dos series **suben y bajan juntas**, en `[−1, 1]`:
    `r ≈ 1` (varían igual), `r ≈ 0` (sin relación lineal), `r ≈ −1` (opuestas). No depende de
    la escala.
  - Intuición aquí: para la componente PCA número *j*, ¿cuando su valor real es alto en una
    imagen, el modelo también predice un valor alto? Si sí → `r` positivo → la fMRI capta esa
    dirección de bajo nivel.
  - **`mean_pearson`** = media de esas 512 correlaciones; **`median_pearson`** = mediana.
    Suelen ser bajos porque **las primeras componentes** (las de más varianza) se predicen
    mejor y las últimas casi nada; la figura `pca_component_correlation_*.png` muestra
    justamente esa curva decreciente. Un `mean_pearson` **positivo** (tu 0,052) indica señal
    real; lo importante es que las **primeras** componentes tengan `r` claramente > 0.
- **mean_cosine (bajo nivel)** = coseno medio entre vector PCA predicho y real. Complementario;
  no es la métrica principal aquí.

**Qué se considera bueno (bajo nivel)**: R² > 0 y Pearson medio > 0 (y creciente hacia las
primeras componentes). No esperes valores altos: el bajo nivel desde fMRI es intrínsecamente
difícil. Lo esencial es que **no dañe** el retrieval CLIP.

---

## 4. Experimento 4 — Los tres modos de generación (`generation.mode`)

La condición fMRI puede llegar a Stable Diffusion (congelado) por dos vías:
- **Semántica (alto nivel)**: el embedding CLIP predicho → `TokenAdapter` → `prompt_embeds`
  (los "tokens" que guían la U-Net). El adapter es lo **único entrenable**.
- **Estructural (bajo nivel)**: el vector PCA predicho → se "deshace" la PCA → un latente VAE
  aproximado → se usa como **imagen inicial de img2img** (SD parte de ahí en vez de ruido).

El parámetro `generation.mode` decide **cuál(es)** de esas vías se usan:

### 4.1 `adapter` (Opción B) — solo semántica  *(por defecto)*
- **Qué usa**: solo el embedding CLIP predicho (a través del TokenAdapter).
- **Entrenamiento**: **necesita** el TokenAdapter entrenado (`train_adapter: true` o
  `--train-adapter`). El decoder puede ser Exp1 o Exp3 (solo se usa su cabeza CLIP).
- **Cómo genera**: **text2img** normal (parte de ruido), guiado por los tokens del adapter.
- **Efecto**: la imagen intenta reflejar el **contenido** que la fMRI predijo; la composición
  la "inventa" SD. Es la vía conceptualmente más limpia (semántica pura).
- **Cuándo usarlo**: opción principal y más robusta; es la que has entrenado.

### 4.2 `lowlevel_img2img` (Opción C) — solo estructura
- **Qué usa**: solo el vector PCA de bajo nivel predicho. **No** usa el adapter.
- **Entrenamiento**: **no** requiere entrenar el adapter (puedes poner `train_adapter: false`).
  **Requiere** un decoder con cabeza de bajo nivel → usa el checkpoint de **Exp3**.
- **Cómo genera**: reconstruye un latente desde la PCA predicha, lo decodifica a una imagen
  inicial y hace **img2img** desde ahí, **sin guía semántica** (el prompt es nulo). El
  parámetro `strength` (0,8 por defecto) controla cuánto se aparta SD de esa inicialización
  (1,0 = la ignora; 0,0 = la deja casi intacta).
- **Efecto**: la imagen refleja la **estructura/colores** que la fMRI predijo, pero puede no
  tener un contenido semántico coherente. Sirve para ver "cuánta forma" captó la fMRI.
- **Cuándo usarlo**: para aislar la contribución de bajo nivel, o si no quieres entrenar el
  adapter.

### 4.3 `adapter_lowlevel` (B + C) — estructura + semántica
- **Qué usa**: **ambas**: la PCA predicha como inicialización img2img **y** el embedding CLIP
  predicho (vía adapter) como guía semántica.
- **Entrenamiento**: necesita el adapter entrenado **y** un decoder de Exp3 (cabeza de bajo
  nivel).
- **Cómo genera**: img2img partiendo del latente de bajo nivel, guiado por los tokens del
  adapter.
- **Efecto**: combina "dónde/cómo" (bajo nivel) con "qué" (semántica). En principio la
  reconstrucción más completa, a costa de depender de que ambas ramas sean buenas.
- **Cuándo usarlo**: cuando la rama de bajo nivel (Exp3) da resultados razonables y quieres la
  reconstrucción más fiel.

### 4.4 Tabla resumen

| mode | usa CLIP (adapter) | usa PCA (img2img) | entrena adapter | decoder necesario | generación |
|---|---|---|---|---|---|
| `adapter` | Sí | No | Sí | Exp1 o Exp3 | text2img (desde ruido) |
| `lowlevel_img2img` | No | Sí | No | **Exp3** | img2img (sin prompt) |
| `adapter_lowlevel` | Sí | Sí | Sí | **Exp3** | img2img + prompt |

> Nota: en todos los modos, la comparación de Exp5 (correcto/permutado/cero) se hace igual;
> lo que cambia es **por dónde entra** la señal fMRI en el generador.

---

## 5. Configuraciones recomendadas de entrenamiento (por experimento)

Todo se puede fijar en el `expNN.yaml` o por CLI con `--set clave=valor`. Los valores por
defecto de `base.yaml` ya son razonables; abajo, recomendaciones y para qué tocar cada cosa.

### 5.1 Experimento 1 — fMRI→CLIP (`configs/exp01_fmri_to_clip.yaml`)
```yaml
training:
  epochs: 100                 # el early stopping lo corta antes (en tu caso ~época 13)
  early_stopping_patience: 10 # súbelo a 15 si quieres explorar más antes de parar
  lr: 1.0e-4
  weight_decay: 0.01
  batch_size: 64              # baja a 32 si te falta VRAM
  scheduler: cosine
  warmup_ratio: 0.02
  mixed_precision: true       # true en GPU
checkpointing:
  monitor: val_top5           # métrica que elige best.pt
  save_every_n_epochs: 5      # ↑ para ahorrar disco (cada ckpt ≈2,7 GB)
  keep_last_n: 1
```
- **Épocas**: 100 es un techo; converge y sobreajusta pronto, así que en la práctica para
  antes. No necesitas más de 100. Si el `best` sale muy temprano (época 3–5, como en tu caso),
  puedes probar más regularización (`fmri_encoder.dropout` 0,3, `voxel_dropout` 0,15) o `lr`
  algo menor (5e-5) para ver si mejora la validación.

### 5.2 Experimento 3 — Multitarea (`configs/exp03_lowlevel_multitask.yaml`)
```yaml
model:
  use_lowlevel: true
  lowlevel_head: { output_dim: 512 }   # debe coincidir con features.pca_dim
features:
  pca_dim: 512                # 1024 capta más varianza de bajo nivel (cabeza más grande)
losses:
  lambda_cosine: 1.0
  lambda_contrastive: 1.0
  lambda_lowlevel: 0.25       # 0.1 si el retrieval CLIP se resiente; 0.5 para priorizar bajo nivel
training:
  epochs: 100                 # igual que Exp1; early stopping lo corta
checkpointing:
  monitor: val_top5           # sigue mandando el retrieval CLIP
```
- **Épocas**: como Exp1 (~100 con early stopping). El resto igual que Exp1.

### 5.3 Paso 4 — Precómputo VAE+PCA (usa `configs/exp03_...`)
```yaml
features:
  vae_image_size: 512
  vae_batch_size: 16          # baja a 8 si te falta VRAM al codificar con el VAE
  pca_dim: 512
```
- No hay "épocas": es determinista. Solo se ejecuta una vez por sujeto.

### 5.4 Experimento 4 — Generación / adapter (`configs/exp04_generation.yaml`)
```yaml
generation:
  mode: adapter               # adapter | lowlevel_img2img | adapter_lowlevel (ver §4)
  train_adapter: true         # false si mode=lowlevel_img2img (no hace falta adapter)
  adapter_epochs: 15          # 15–40: más = mejor reconstrucción, pero 1–4 h de entrenamiento
  adapter_batch_size: 4       # ajustado a 16 GB @512px; baja si hay OOM
  adapter_lr: 1.0e-4
  adapter_resume: auto        # reanuda adapter_last.pt si existe
  num_samples: 16             # nº de imágenes a generar por condición (súbelo para más ejemplos)
  num_inference_steps: 50     # pasos de difusión (30–50 razonable)
  guidance_scale: 3.0         # 2–5; más alto = más "obediente" a la condición, menos diverso
  strength: 0.8               # solo modos img2img: cuánto se aparta de la init de bajo nivel
  # decoder: por defecto usa outputs/exp03.../best.pt (tiene ambas cabezas); si no, exp01
```
- **Épocas del adapter (`adapter_epochs`)**: es el único "entrenamiento" de Exp4 y **lo más
  caro** (pérdida de difusión con la U-Net congelada). Guía:
  - **5–10**: prueba rápida para cerrar el pipeline (reconstrucciones burdas).
  - **15–30**: equilibrio razonable (tu config está en 15).
  - **40+**: solo si tienes tiempo y buscas mejor calidad. Recuerda `adapter_resume: auto`
    para poder ir acumulando épocas en varias sesiones.
- **Épocas de generación**: no aplica; la generación no entrena, solo `num_inference_steps`
  (pasos del muestreador de difusión por imagen).

### 5.5 Experimento 5 — Comparación (`configs/exp05_generation_ablation.yaml`)
- No entrena nada: solo mide y hace rejillas. Opcionalmente activa `compute_ssim`/`compute_lpips`
  (requieren `scikit-image`/`lpips` instalados).

### 5.6 Regla general sobre épocas
Para Exp1/Exp3 **no fijes un número bajo de épocas "a mano"**: pon un techo alto (100) y deja
que el **early stopping** (por `val_top5`) elija el mejor momento y guarde `best.pt`. Para el
adapter (Exp4) sí eliges tú el número de épocas según tu presupuesto de tiempo.

---

## 6. El problema de la pérdida del adapter y dos palancas para mitigarlo

### 6.1 Por qué la pérdida de entrenamiento del adapter engaña

El adapter (Exp4) se entrena con una **pérdida de difusión de un solo paso**: para cada imagen
se elige **un timestep `t` aleatorio** (de 0 a 999), se añade ruido según ese `t`, y se pide a
la U-Net congelada que adivine ese ruido a partir de los tokens del adapter. La pérdida es el
MSE entre el ruido real y el predicho.

Generar una imagen es un proceso **muy distinto**: se parte de ruido puro y se aplica la U-Net
**muchas veces seguidas** (p. ej. 50 pasos), acumulando cada predicción, y en cada paso se hace
*classifier-free guidance* (CFG): dos pasadas (con y sin condición) que se combinan
amplificando su diferencia por `guidance_scale`. Por eso la pérdida de entrenamiento es un
**proxy ruidoso e imperfecto** de la calidad real de generación:

- Cada época evalúa cada imagen en **un solo `t`** de los 1000 posibles, y la dificultad
  depende muchísimo de qué `t` toque (poco ruido = fácil; casi todo ruido = casi imposible).
  Eso mete varianza "de sorteo" en la media por época → la curva oscila sin bajar.
- El entrenamiento **nunca** comprueba "¿genera bien tras 50 pasos + CFG partiendo de ruido?".
  Un adapter puede tener una pérdida de un paso similar (o menor) y aun así generar **peor**.

Esto es exactamente lo que se observó empíricamente (subj01): checkpoints con época más avanzada
y pérdida ~igual generaban peor que checkpoints muy tempranos; y con SD-2.1 la pérdida quedó
plana ~0,155 durante 25 épocas sin que la calidad mejorara. **Conclusión práctica: no elijas el
checkpoint del adapter por `best_loss`; evalúa la calidad de generación.** El notebook
`06_adapter_checkpoint_sweep.ipynb` / `scripts/08_sweep_adapter_checkpoints.py` hacen justo eso
(generar con varios checkpoints y puntuar con similitud CLIP). Las dos palancas siguientes
atacan el problema **durante** el entrenamiento.

### 6.2 Palanca 1 — Promediar la pérdida sobre varios timesteps

```yaml
generation:
  adapter_timesteps_per_sample: 1   # 1 = comportamiento original (off). Prueba 2-4.
```
En vez de evaluar cada muestra en un único `t`, la evalúa en N timesteps aleatorios y **promedia**
la pérdida. El adapter se calcula **una sola vez** por batch (solo se repiten las pasadas por la
U-Net congelada). Efecto: la pérdida por época es **menos ruidosa** y algo más fiable como señal.
Coste: N pasadas extra de U-Net por batch → más VRAM y tiempo (con N=2-4 es asumible en 16 GB).
Limitación: **no** representa los 50 pasos + CFG; solo reduce el ruido de la métrica que ya usas.

### 6.3 Palanca 2 — Elegir el checkpoint por calidad de generación (no por la pérdida)

```yaml
generation:
  adapter_eval_enabled: false          # master switch (off por defecto)
  adapter_eval_every_n_epochs: 5       # cada cuántas épocas evaluar
  adapter_eval_num_samples: 6          # nº de imágenes held-out a generar en la eval
  adapter_eval_steps: 25               # pasos de difusión en la eval (menos = más rápido)
  adapter_eval_split: val              # split de donde salen las imágenes held-out
  adapter_eval_guidance_scale: null    # null -> usa generation.guidance_scale
  adapter_select_by: auto              # auto | loss | clip_sim
```
Cada N épocas genera unas pocas imágenes held-out con el adapter actual y las puntúa por
**similitud CLIP** (generada vs real), y guarda `adapter_best.pt` según **esa** métrica en vez
de la pérdida. Es el mismo principio que `monitor: val_top5` en Exp1/Exp3, pero "validar" aquí
significa **generar de verdad y medir**.

Detalles de diseño:
- Evalúa la **tarea propia del adapter** (embedding CLIP real → imagen), usando los embeddings
  CLIP **ya precomputados** del split held-out. No depende del decoder de fMRI ni de las
  condiciones correcto/permutado/cero (eso es cosa del *sweep* y de Exp5, aguas abajo).
- **Reutiliza la U-Net de entrenamiento** para generar (no carga una segunda copia de ~1,7 GB;
  solo añade el VAE) y fuerza la ruta semántica pura (`mode='adapter'`) durante la eval.
- `val_clip_sim`/`best_val_sim` se registran en `logs/adapter_train_log.csv` y en el checkpoint
  (se restauran al reanudar).
- `adapter_select_by: auto` → `clip_sim` si la eval está activa, `loss` si no. Puedes forzar
  `loss` (para tener la métrica CLIP en el CSV pero seguir eligiendo por pérdida) o `clip_sim`.
- Coste: cada época de evaluación es bastante más lenta (genera `num_samples` imágenes con
  `adapter_eval_steps` pasos). Sube `adapter_eval_every_n_epochs` o baja `num_samples`/`steps`
  para acelerar.

### 6.4 Las cuatro combinaciones (para diseñar experimentos)

| `adapter_timesteps_per_sample` | `adapter_eval_enabled` | efecto |
|---|---|---|
| `1` | `false` | comportamiento original (ninguna mejora) |
| `>1` | `false` | pérdida menos ruidosa; `best.pt` por pérdida |
| `1` | `true` | `best.pt` por calidad de generación (CLIP); pérdida normal |
| `>1` | `true` | ambas a la vez |

### 6.5 ¿Y optimizar directamente la calidad de imagen en la pérdida?

Sí existe (métodos tipo DRaFT / DDPO / Diffusion-DPO: optimizar contra una "recompensa" calculada
sobre la imagen final), pero implica **retropropagar a través de los ~50 pasos de muestreo**
(~100× más caro por paso y mucha más memoria) y suele usarse como **ajuste fino** sobre un modelo
ya bueno, no como entrenamiento principal — inviable para el hardware/tiempo de un TFM. La
palanca 2 es la alternativa práctica: seguir entrenando con la pérdida barata pero **seleccionar**
el checkpoint por calidad de generación medida.
