# Documento 6 — El TokenAdapter y el proceso de generación, en detalle

> Documento de referencia para entender **qué es exactamente el TokenAdapter**, **con qué se
> entrena** y **cómo se genera una imagen** a partir de la señal cerebral. Complementa el
> Documento 3 (§4, modos de generación) entrando en el detalle de espacios, formas y código.
>
> **Parte 1** — el TokenAdapter: qué espacios conecta y cómo se entrena.
> **Parte 2** — el proceso de generación completo (predicción → CLIP → VAE → imagen).

---

# Aclaración previa: los «tokens VAE» no existen

Esta es la confusión que hay que deshacer antes de nada. En Stable Diffusion intervienen **dos
espacios distintos** que no se deben mezclar:

| | **Espacio latente del VAE** | **Espacio de condición (cross-attention)** |
|---|---|---|
| Qué es | El **lienzo**: la imagen comprimida | Las **instrucciones**: «qué pintar» |
| Forma (SD-1.5, 512×512) | `[4, 64, 64]` = 16 384 números | `[77, 768]` = 77 tokens de 768 dims |
| Quién lo produce | El **encoder del VAE** (imagen → latente) | Normalmente el **text encoder de CLIP** (prompt → tokens) |
| Papel en la difusión | Es **lo que se desruidiza** | Es **lo que guía** el desruidizado |
| Argumento de la U-Net | `sample` (el `z_t` ruidoso) | `encoder_hidden_states` |

La llamada real a la U-Net, tal cual está en el código
([`sd_pipeline.py`](../src/generation/sd_pipeline.py)):

```python
pred = unet(zt, t, encoder_hidden_states=cond).sample
#           ↑                            ↑
#     latente VAE ruidoso        tokens de condición
```

Con esto ya se puede responder a la duda de raíz:

> **SD no espera «tokens VAE».** Espera **dos cosas separadas**: un latente del VAE (el lienzo) y
> una secuencia de tokens de condición (las instrucciones). **El TokenAdapter solo produce lo
> segundo.** El latente del VAE aparece en su entrenamiento únicamente como *aquello que hay que
> desruidizar* — es la supervisión, nunca la salida del adapter.

---

# Parte 1 — El TokenAdapter

## 1.1. ¿Por qué hace falta un adapter?

Stable Diffusion 1.5 fue entrenado para obedecer a **texto**: el usuario escribe un prompt, el
text encoder de CLIP lo convierte en `[77, 768]` y la U-Net lee esa secuencia por
cross-attention.

Nosotros no tenemos texto. Tenemos, predicho desde el cerebro, **un único vector CLIP de imagen
de 768 dimensiones**. Y aquí está el problema:

- CLIP de imagen produce **un vector** `[768]` que resume la imagen completa.
- SD espera **una secuencia** `[77, 768]` en el espacio de salida del *text* encoder.

Aunque el número 768 coincida en ambos casos (casualidad de SD-1.5 con `ViT-L-14`), **son
espacios diferentes**: uno es «resumen global de una imagen», el otro es «secuencia de estados
ocultos de un codificador de texto». No se pueden intercambiar directamente.

El TokenAdapter es el **traductor** entre ambos. Es la **Opción B** de la especificación (§10.3):

```
emb. CLIP de imagen [768]  ──TokenAdapter──►  pseudo-tokens [77, 768]  ──►  U-Net congelada
```

Se llaman *pseudo*-tokens porque no corresponden a ninguna palabra: son vectores que ocupan el
sitio donde iría un prompt y que la U-Net interpreta como si lo fueran. Entran por la **API
pública** `prompt_embeds` de diffusers, sin tocar la U-Net.

## 1.2. Arquitectura exacta

De [`src/models/adapters.py`](../src/models/adapters.py):

```python
class TokenAdapter(nn.Module):
    """CLIP image embedding [B, clip_dim] -> SD tokens [B, num_tokens, cross_dim]."""
    def __init__(self, clip_dim, cross_dim=768, num_tokens=77, hidden_dim=1024, dropout=0.0):
        self.net = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),              # 768 -> 1024
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * cross_dim),# 1024 -> 77*768 = 59 136
        )

    def forward(self, clip_emb):
        out = self.net(clip_emb)
        return out.view(-1, self.num_tokens, self.cross_dim)   # [B, 77, 768]
```

Es una MLP de dos capas: proyecta a 1024, y luego a `77 × 768 = 59 136` valores que se
**reorganizan** (`view`) en 77 tokens de 768. Unos **61 M de parámetros**, casi todos en la
segunda capa — pequeño frente a los ~860 M de la U-Net, y el **único módulo entrenable** de toda
la etapa de generación.

`cross_dim` no está fijado a mano: se lee del propio modelo
(`cross_dim = int(unet.config.cross_attention_dim)`), que es 768 en SD-1.5 y **1024 en SD-2.1**.
Por eso cambiar de backbone obliga a reentrenar el adapter desde cero.

## 1.3. Con qué se entrena (esto es lo importante)

Los pares de entrenamiento los monta
[`_load_adapter_training_data`](../src/generation/sd_pipeline.py):

```python
clip    = load_split_features(cfg, subj, split, "clip")   # CLIP(imagen real)   [N, 768]
latents = np.load(vae_latent_path(cfg, subj, split))      # VAE(la MISMA imagen)[N, 16384]
```

Es decir, para cada imagen del split de **train**:

- **entrada**: el embedding CLIP de la **imagen real** (precomputado en el paso `01`);
- **objetivo**: el latente VAE de **esa misma imagen** (precomputado en el paso `04`, guardado
  aplanado como `z = vae.encode(x).mean × 0.18215`, la escala estándar de SD).

⚠️ **El cerebro no aparece en ningún punto del entrenamiento del adapter.** No se carga ningún
checkpoint del decoder, ni se lee el tensor EEG/fMRI. La tarea es puramente **imagen → imagen**.

## 1.4. El bucle de entrenamiento, paso a paso

Para un lote de imágenes (código real en
[`train_token_adapter`](../src/generation/sd_pipeline.py)):

```python
cond_in = clip_t[idx]                                   # 1) CLIP real         [B, 768]
z0      = lat_t[idx]                                    # 2) latente VAE real  [B, 4, 64, 64]
cond    = adapter(cond_in)                              # 3) pseudo-tokens     [B, 77, 768]
noise   = torch.randn_like(z0)                          # 4) ruido ε
t       = torch.randint(0, 1000, (B,))                  # 5) timestep aleatorio
zt      = noise_sched.add_noise(z0, noise, t)           # 6) latente ruidoso
target  = noise                                          #    (prediction_type='epsilon')
pred    = unet(zt, t, encoder_hidden_states=cond).sample# 7) U-Net CONGELADA predice el ruido
loss    = mse(pred, target)                             # 8) MSE
loss.backward()                                          # 9) solo se actualiza el adapter
```

Leído en palabras:

> Se toma la imagen real, se la comprime al latente del VAE, se le **añade ruido** en un instante
> `t` aleatorio del proceso de difusión, y se le pide a la U-Net **congelada** que adivine ese
> ruido, dándole como única pista los tokens que ha producido el adapter. Si los tokens describen
> bien la imagen, la U-Net acierta más y la pérdida baja.

O sea: el adapter aprende a **escribir, en el idioma de SD, una descripción de la imagen a partir
de su embedding CLIP**, suficientemente buena como para ayudar a reconstruirla.

### Quién está congelado y quién entrena

| Componente | Estado | Papel |
|---|---|---|
| **TokenAdapter** (~61 M) | 🔓 **entrenable** | traduce CLIP → tokens |
| U-Net de SD (~860 M) | 🔒 congelada | predice el ruido (juez fijo) |
| VAE | 🔒 congelado | ya usado en el paso `04` (los latentes vienen precomputados) |
| CLIP | 🔒 congelado | ya usado en el paso `01` (los embeddings vienen precomputados) |
| Decoder cerebral (Exp1/3) | ❌ **no interviene** | — |

Como CLIP y VAE ya se aplicaron en los precómputos, el entrenamiento del adapter **solo necesita
la U-Net en memoria** — de ahí que quepa en 16 GB de VRAM.

## 1.5. Por qué la pérdida del adapter engaña

La pérdida es un **MSE de difusión de un solo paso**: cada imagen se evalúa en **un** `t` de los
1000 posibles, y la dificultad depende enormemente de qué `t` toque (poco ruido = fácil; casi
todo ruido = casi imposible). Generar, en cambio, son ~50 pasos encadenados + CFG.

Consecuencia práctica, verificada empíricamente en este proyecto: **checkpoints con pérdida igual
o menor pueden generar peor.** Por eso existen dos palancas opcionales (Documento 3 §6):

- `generation.adapter_timesteps_per_sample > 1`: promedia la pérdida sobre N timesteps por
  muestra → señal menos ruidosa (cuesta N pasadas extra de U-Net).
- `generation.adapter_eval_enabled: true`: cada N épocas **genera** unas imágenes held-out y elige
  `adapter_best.pt` por **similitud CLIP**, no por pérdida. Ojo: esa evaluación también usa
  embeddings CLIP **reales** del split held-out, no predichos desde el cerebro.

Y como red de seguridad, se guardan *snapshots* periódicos `epoch_XXXX.pt` para poder barrerlos
después con `scripts/08_sweep_adapter_checkpoints.py`.

## 1.6. De qué depende el adapter (y de qué no)

| Si cambias… | ¿Reentrenar? | Motivo |
|---|---|---|
| Preprocesado del EEG (cualquier variante) | **No** | no toca CLIP ni el VAE |
| `features.clip_model` | **Sí** | cambia el espacio de entrada |
| `generation.sd_model` (1.5 → 2.1) | **Sí** | `cross_attention_dim` 768 → 1024 |
| `features.vae_model` / `vae_image_size` | **Sí** | cambian los latentes objetivo |
| `val_ratio` / `split_seed` / sujetos | **Sí** (en rigor) | cambian los pares de entrenamiento |
| Modalidad fMRI ↔ EEG | **Sí** | distinto conjunto de imágenes (COCO/NSD vs THINGS) |

Dentro de una misma modalidad, **todas las variantes de preprocesado comparten un único adapter**
(ver Documento 5 §6). Para reutilizarlo de verdad hay que pedirlo explícitamente, porque el
script `06` lo reentrenaría en la carpeta del nuevo experimento:

```bash
--set generation.train_adapter=false \
--adapter-checkpoint outputs/<exp_previo>/checkpoints/adapter_best.pt
```

Esto además **refuerza la comparabilidad** entre ablaciones: con el mismo adapter congelado,
cualquier diferencia en Exp5 es atribuible al preprocesado y al decoder, no a dos adapters
distintos.

---

# Parte 2 — El proceso de generación

## 2.1. La cadena completa

```
                    ┌── ENTRENADO ANTES (Exp1/Exp3) ──┐   ┌─ ENTRENADO ANTES (Exp4) ─┐
  EEG/fMRI  ──►  decoder  ──►  ĈLIP predicho [768] ──► TokenAdapter ──► tokens [77,768]
  [C,T]/[V]                         │                                        │
                                    │                                        ▼
                                    │                            U-Net congelada (×50 pasos, CFG)
                                    │                                        │
                                    │  (opcional, modos low-level)           ▼
                                    └─► P̂CA [512] ─inversa─► latente ─► ruido ──► latente final
                                                                                   │
                                                                        VAE decoder (congelado)
                                                                                   ▼
                                                                            imagen 512×512
```

Nótese el contraste con la Parte 1: **aquí el adapter recibe un embedding CLIP *predicho*, no el
real**. Fue entrenado con reales y en inferencia se le dan predichos. Ese salto tiene dos
componentes que conviene no confundir:

- **la dirección** es peor (el decoder no acierta el embedding exacto: coseno ~0,3–0,6) →
  irreducible, es el propio problema de decodificación;
- **la norma** está descalibrada → **esto sí se puede corregir**, y hay dos mecanismos
  implementados para ello (§2.3 bis y §2.3 ter).

## 2.2. Paso 1 — Del cerebro al embedding CLIP predicho

[`predict_condition_embeddings`](../src/generation/generate_from_fmri.py):

```python
mats = load_subject_matrices(cfg, datamodule, subj, split, want=("fmri",))
fin  = make_condition_input(mats.fmri, cond, rng, noise_std)   # correct|permuted|zero|noise
out  = model(batch, subject=...)      # decoder de Exp1/Exp3
# out["clip"] -> [N, 768]   (y out["low"] -> [N, 512] si el modelo tiene rama low-level)
```

Dos detalles importantes:

1. **La señal entra ya normalizada y a nivel de imagen** (en EEG, promediada sobre repeticiones).
2. **Las condiciones de ablación se construyen aquí**, sobre la señal cerebral, *antes* del
   decoder: `correct` (real), `permuted` (derangement de Sattolo — nunca su propia señal), `zero`
   (ceros), `noise` (gaussiano). Todas las condiciones comparten las mismas imágenes y la misma
   semilla de generación, para que la comparación sea limpia.

## 2.3. Paso 2 — Del embedding a los tokens de condición

```python
generator.load_adapter(int(meta["clip_dim"]), adapter_ck)   # adapter entrenado, en eval()
...
with torch.no_grad():
    tokens = self.adapter(clip_embeds)     # [B, 77, 768]
```

El adapter va **congelado** (`no_grad`) y `clip_embeds` es el vector predicho. Coherencia
importante: los embeddings CLIP se guardan **sin normalizar** en el paso `01`, y el `CLIPHead`
también emite vectores sin normalizar → ambos lados usan la misma convención.

⚠️ Hay un desajuste que conviene conocer: el `CLIPHead` se optimiza por **dirección** (coseno +
InfoNCE), no por magnitud, así que la **norma** del vector predicho no queda calibrada respecto a
la de los reales con los que se entrenó el adapter. **Ya no es una limitación inevitable**: está
medida y tiene dos mitigaciones implementadas —§2.3 bis (opción A, sin reentrenar) y §2.3 ter
(opción B, que la elimina por construcción)—. Ambas están **desactivadas por defecto**, así que
con la configuración por defecto el desajuste sigue presente; actívalas a conciencia.

### 2.3 bis. La calibración de la norma (`generation.rescale_clip_pred`)

**El problema, medido.** Ningún término de la pérdida de Exp1/Exp3 ve la norma (coseno e InfoNCE
normalizan; `lambda_nmse` = 0), así que la norma de `clip_pred` es un **parámetro libre que nada
supervisa**. Los embeddings CLIP reales, en cambio, viven en una cáscara estrechísima:
**19,46 ± 0,90** (~4,6 % de variación). Medido sobre los runs de este repo, el ratio
`‖pred‖/‖real‖` va de **0,54 a 1,39** según el experimento.

Como el `TokenAdapter` resulta ser **casi equivariante a escala**
(`‖adapter(2x)‖ / ‖2·adapter(x)‖ = 1,042`), el efecto no es semántico sino de **intensidad**: con
dirección perfecta y norma equivocada, los tokens cambian un 36,6 % en magnitud pero mantienen
coseno **0,9993**. En la práctica funciona como un `guidance_scale` efectivo descontrolado, y
distinto en cada run.

**La opción implementada.** `generation.rescale_clip_pred` (por defecto **`none`**, no altera nada
de lo ya ejecutado):

| valor | efecto |
|---|---|
| `none` | comportamiento histórico |
| `train_median` | proyecta cada `clip_pred` sobre la esfera de radio = mediana de `‖CLIP real‖` del split **train** (sin leakage) |
| un float | usa ese radio tal cual → sirve como **palanca de intensidad de condicionamiento** |

Se aplica justo en la frontera del adapter (`FrozenSDGenerator._prompt_embeds`), así que afecta
solo a los modos que usan adapter, y queda registrado en `generation_params.json`. La evaluación
en bucle del adapter (*feature 2*) **no** se toca: ya recibe embeddings reales.

**Resultado del A/B (sub-08, adapter `exp04_63`, n = 32, mismas imágenes y semillas):**

| caso (ratio) | `correcto` A→B | p (Wilcoxon) |
|---|---|---|
| official-63 (1,39, sobre-escalado) | 0,6195 → 0,6102 (**−0,0093**) | **0,008** |
| raw-baseline (0,58, sub-escalado) | 0,5963 → 0,6037 (+0,0073) | 0,184 |

Y sobre la **comparabilidad entre variantes** (que era el objetivo):

| | \|clip_sim(official-63) − clip_sim(raw-baseline)\| |
|---|---|
| sin calibrar | **0,0232** |
| con `train_median` | **0,0065** (3,6× menor) |

**Conclusión honesta:**

1. **No mejora la calidad absoluta.** En el run sobre-escalado la **empeora de forma
   significativa**. Coherente con el mecanismo: una norma mayor = condicionamiento más fuerte, y
   eso subía la similitud CLIP. Calibrar a la baja eliminó ese beneficio accidental.
2. **Sí hace lo que se diseñó**: elimina el confusor de escala y hace comparables las variantes
   (0,023 → 0,007). Sin calibrar, una parte de la ventaja aparente de `official-63` era solo
   condicionamiento más intenso, no mejor decodificación.
3. **Contexto que limita todo lo anterior**: los márgenes `correcto − permutado` son
   **+0,010 … +0,016 y NO significativos** (p = 0,065–0,278) en las cuatro combinaciones. Es
   decir, en este montaje la generación **no demuestra dependencia clara del EEG**, así que
   ninguna de estas diferencias debe presentarse como mejora de reconstrucción.

**Recomendación de uso**: mantener `none` para los números de calidad, y usar `train_median`
**cuando se comparen variantes de preprocesado entre sí** (Exp5 sobre las ablaciones), reportando
que se hizo. Y, dado que un valor mayor ayudó, la intensidad de condicionamiento es un
hiperparámetro sin explorar: se puede barrer pasando floats a `rescale_clip_pred` (o tocando
`guidance_scale`).

> Todo esto es **un sujeto, un adapter y n = 32**: indicativo, no concluyente.

### 2.3 ter. Opción B — adapter invariante a escala por construcción

La opción A calibra la norma *fuera* del adapter. La **opción B** elimina el problema de raíz:
hace que el adapter **ignore la norma por construcción**, normalizando su entrada dentro del
propio `forward` ([`adapters.py`](../src/models/adapters.py)):

```python
def forward(self, clip_emb):
    if self.normalize_input:
        clip_emb = F.normalize(clip_emb, dim=-1) * self.input_scale
    out = self.net(clip_emb)
    return out.view(-1, self.num_tokens, self.cross_dim)
```

Con `normalize_input=True` se cumple `adapter(x) == adapter(k·x)` para todo `k > 0`
(verificado: `max|a(x) − a(2x)| = 0`). La deriva arbitraria de la norma del decoder deja de poder
actuar como intensidad de condicionamiento descontrolada.

**Dos claves de config** (ambas OFF por defecto, no alteran nada existente):

| clave | efecto |
|---|---|
| `generation.adapter_normalize_input` | `true` = adapter invariante a escala. **Requiere reentrenar el adapter** (se entrena con embeddings normalizados) |
| `generation.adapter_input_scale` | intensidad de condicionamiento en inferencia (entrenamiento siempre a 1.0). **Se barre sin reentrenar** |

**Por qué se normaliza *dentro* del módulo** (y no en el pipeline): así el flag viaja con el
módulo y no puede haber desajuste. Un adapter entrenado con entradas normalizadas y ejecutado sin
normalizar (o al revés) produciría condicionamiento basura de forma **silenciosa**. Protecciones
implementadas y verificadas:

1. El flag se **guarda en cada checkpoint** del adapter (`"normalize_input"`).
2. `load_adapter` lo lee **del checkpoint**, que manda sobre el config, y avisa si discrepan.
3. Los checkpoints antiguos (sin la clave) se interpretan como `False` → **sin regresión**.
4. Reanudar un entrenamiento con el flag cambiado **falla con un error explícito** en lugar de
   mezclar los dos regímenes.
5. La evaluación en bucle (*feature 2*) funciona sin cambios: recibe el propio objeto adapter, así
   que la normalización viaja dentro.

**Cómo usarla:**

```bash
# 1) reentrenar el adapter en modo invariante (nuevo experiment.name) — 1-4 h
python scripts/06_generate_images.py --config configs/EEG/exp04_63_generation.yaml \
  --train-adapter --set generation.adapter_normalize_input=true \
  --set experiment.name=exp04_63_eeg_generation_normadapter

# 2) elegir la intensidad probando varios valores, SIN reentrenar (minutos)
python scripts/12_sweep_adapter_input_scale.py \
  --config configs/EEG/exp04_63_generation.yaml \
  --adapter-checkpoint outputs/exp04_63_eeg_generation_normadapter/checkpoints/adapter_best.pt \
  --scales 0.6 0.8 1.0 1.2 1.4 1.8 --num-samples 8

# 3) generar en serio con el valor elegido
python scripts/06_generate_images.py --config configs/EEG/exp04_63_generation.yaml \
  --set generation.adapter_normalize_input=true --set generation.adapter_input_scale=1.4 \
  --set generation.train_adapter=false \
  --adapter-checkpoint outputs/exp04_63_eeg_generation_normadapter/checkpoints/adapter_best.pt
```

**El paso 2 explicado** («barrer» = probar varios valores y comparar). `adapter_input_scale` es un
simple multiplicador aplicado **en inferencia** dentro del `forward`; los pesos entrenados no
cambian. Así que es como `guidance_scale`: se prueban varios valores reutilizando el mismo
checkpoint. `scripts/12_sweep_adapter_input_scale.py` carga el adapter y las predicciones **una
sola vez**, genera con cada valor, puntúa como el Experimento 5 (similitud CLIP correcto /
permutado / cero) y escribe en `outputs/<exp>/input_scale_sweep/`:
`input_scale_sweep_summary.csv`, `input_scale_sweep_margins.csv`, `sweep_params.json`,
`input_scale_sweep.png` (calidad y **margen** frente a la escala) y las rejillas por valor.

Se elige por **margen (correcto − mejor control)**, no por `correct` a secas: un `correct` más alto
con un control igual de alto no significa nada. El script falla **antes de gastar GPU** si el
adapter no es invariante a escala (con `normalize_input=False` el parámetro se ignora y todos los
valores darían imágenes idénticas) o si el modo no carga el adapter.

**Qué esperar (y qué no).** La opción B garantiza **comparabilidad** entre variantes de
preprocesado sin depender de una calibración externa, y convierte la intensidad de
condicionamiento en un hiperparámetro explícito. Pero **fija esa intensidad a 1.0 por defecto**, y
el A/B de §2.3 bis mostró que una intensidad *mayor* que la nominal subía la similitud CLIP: por
tanto **conviene barrer `adapter_input_scale`** (y/o `guidance_scale`) tras reentrenar, o se
perderá ese efecto beneficioso. La sensibilidad exacta de la palanca solo puede medirse **con el
adapter ya reentrenado** (en un adapter sin entrenar la salida está dominada por los sesgos y la
palanca parece más débil de lo que es).

| | Opción A (`rescale_clip_pred`) | Opción B (`adapter_normalize_input`) |
|---|---|---|
| Reentrenar | **No** | **Sí**, el adapter |
| Invariancia | aproximada (proyecta a un radio fijo) | **exacta, por construcción** |
| Palanca de intensidad | el propio radio (float) | `adapter_input_scale` |
| Riesgo de desajuste | ninguno | **cubierto**: el flag va en el checkpoint |

⚠️ Si `adapter_normalize_input: true`, la opción A queda **sin efecto** (normalizar deshace
cualquier reescalado); el código avisa con un `WARNING`.

## 2.4. Paso 3 — Los tres modos de generación

`generation.mode` decide por qué vía entra la información cerebral:

### `adapter` (Opción B) — solo semántica, **por defecto**

- Usa **solo** el embedding CLIP predicho, vía el adapter.
- **text2img**: se parte de **ruido puro** y la U-Net desruidiza guiada por los tokens.
- La composición la «inventa» SD; lo que aporta el cerebro es el **contenido**.

### `lowlevel_img2img` (Opción C) — solo estructura

- **No usa el adapter** (los tokens pasan a ser un embedding de ceros → incondicional).
- Usa el vector PCA predicho. [`lowlevel_init_images`](../src/generation/generate_from_fmri.py):

```python
lat = inverse_pca_to_latent(bundle, low_vectors[i:i+1])   # PCA [512] -> latente [4,64,64]
...
imgs = vae.decode(z / vae.config.scaling_factor).sample    # latente -> imagen inicial
```

  Es decir: se **deshace la PCA** para reconstruir un latente aproximado del VAE, se **decodifica
  a imagen**, y esa imagen se usa como punto de partida de **img2img**.
- `strength` (0,8 por defecto) controla cuánto se aparta SD de esa inicialización: `1.0` la
  ignora, `0.0` la deja casi intacta.

### `adapter_lowlevel` (B + C) — estructura + semántica

- Init img2img desde el latente de bajo nivel **y** guía semántica por los tokens del adapter.
- Requiere adapter entrenado **y** un decoder con rama low-level (Exp3).

| modo | usa CLIP (adapter) | usa PCA (img2img) | decoder necesario | parte de |
|---|---|---|---|---|
| `adapter` | Sí | No | Exp1 o Exp3 | ruido |
| `lowlevel_img2img` | No | Sí | **Exp3** | imagen del latente predicho |
| `adapter_lowlevel` | Sí | Sí | **Exp3** | imagen del latente predicho |

## 2.5. Paso 4 — El muestreo (y el prompt negativo)

```python
prompt_embeds = self._prompt_embeds(clip_embeds)          # tokens del adapter
negative      = self._empty_text_embeds(num)              # prompt negativo
generator     = torch.Generator(device).manual_seed(seed) # semilla fija
images = self.pipe(prompt_embeds=..., negative_prompt_embeds=...,
                   num_inference_steps=50, guidance_scale=3.0, generator=generator).images
```

- **CFG (classifier-free guidance)**: en cada paso se hacen dos pasadas (condicionada y no
  condicionada) y se combinan amplificando la diferencia,
  `pred = uncond + guidance_scale · (cond − uncond)`. Con `guidance_scale: 3.0`.
- **Prompt negativo**: por defecto `generation.load_text_encoder: false`, así que SD se carga
  **sin** el text encoder de CLIP (que solo servía para el prompt vacío, y que `transformers` 5.x
  rechaza al cargar SD-1.5). El negativo pasa a ser un **embedding de ceros**, coherente con el
  diseño de prompt vacío.
- **Nunca se usan captions** de la imagen real: taparían el aporte cerebral (spec §20).
- El VAE decoder cierra el proceso convirtiendo el latente final en píxeles.

Todo queda anotado en `metadata/generation_params.json` (modo, condiciones, split, nº de muestras,
semillas, `guidance_scale`, pasos, `strength`, modelo SD, checkpoints usados, `image_ids`).

## 2.6. Paso 5 — Qué se guarda y cómo se evalúa (Exp5)

```
outputs/<exp>/generated/real/<image_id>.png        # el estímulo real
outputs/<exp>/generated/correct/<image_id>.png     # generado con señal correcta
outputs/<exp>/generated/permuted/<image_id>.png    # control
outputs/<exp>/generated/zero/<image_id>.png        # control
outputs/<exp>/metadata/generation_params.json
```

`scripts/07_eval_generation_ablation.py` codifica reales y generadas con CLIP y calcula
**similitud CLIP**, retrieval generada↔real, MSE de píxeles y, opcionalmente, SSIM/LPIPS; monta
las rejillas `[real | correcto | permutado | cero]` y los casos mejores/medianos/peores, y hace
tests estadísticos emparejados.

El criterio de éxito es el mismo principio rector del proyecto:

```
correcto  ≫  permutado ≈ cero
```

Si no se cumple, **no se atribuye** la reconstrucción a la señal cerebral, aunque las imágenes
sean bonitas — y así debe escribirse en la memoria.

---

## Resumen en una tabla

| | Parte 1 (entrenar el adapter) | Parte 2 (generar) |
|---|---|---|
| Entrada al adapter | CLIP de la **imagen real** | CLIP **predicho** desde el cerebro |
| Papel del latente VAE | **objetivo**: es lo que se desruidiza | **salida**: se decodifica a imagen (y, en modos low-level, también entrada inicial) |
| U-Net | congelada, actúa de juez | congelada, muestrea ~50 pasos con CFG |
| Señal cerebral | **no interviene** | la aporta el decoder de Exp1/Exp3 |
| Se entrena | solo el TokenAdapter | **nada** (todo congelado) |
| Norma de la entrada | casi constante (~19,5) | descalibrada, salvo que se active §2.3 bis/ter |

Y la idea que resuelve la duda inicial: **el TokenAdapter no produce latentes del VAE**. Produce
la *condición* (77 tokens) que la U-Net congelada necesita para saber qué pintar; el latente del
VAE es el lienzo sobre el que se pinta, y en el entrenamiento del adapter solo se usa como
referencia para medir si esa condición era buena.
