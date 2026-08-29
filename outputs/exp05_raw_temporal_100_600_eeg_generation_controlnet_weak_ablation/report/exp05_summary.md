# Experiment 5 — comparación generativa

- Arquitectura: `text_adapter_concat_controlnet`
- Texto: `weak` (campo `primary_caption`, plantilla `Image of {caption}`)
- ControlNet: `lllyasviel/sd-controlnet-canny` (canny, escala 0.5)
- Muestras: 200 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
        condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
          correct              0.714100                0.725664      0.405      0.590        0.136183   0.102366    0.756305
         permuted              0.715186                0.726833      0.435      0.610        0.135319   0.097069    0.754974
             zero              0.766331                0.773859      0.570      0.720        0.109926   0.126187    0.730403
    permuted_text              0.596140                0.593753      0.000      0.030        0.135549   0.099930    0.781972
semantic_permuted              0.717050                0.733330      0.430      0.635        0.136206   0.102318    0.753386
    semantic_zero              0.715578                0.723340      0.445      0.615        0.115086   0.106833    0.737218
lowlevel_permuted              0.717063                0.731285      0.415      0.615        0.136653   0.097878    0.755786
    lowlevel_zero              0.764897                0.773485      0.575      0.745        0.108833   0.129857    0.734470
```

## Deltas

```
                 delta positive          negative          metric                                                                  question     value     t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada? -0.001086 8.284899e-01     4.206395e-01 200
    delta_correct_zero  correct              zero clip_similarity                       ¿la señal cerebral correcta supera al control nulo? -0.052231 4.732855e-18     1.182647e-17 200
            delta_text  correct     permuted_text clip_similarity    con cerebro correcto, ¿aporta el caption correcto frente al permutado?  0.117959 9.711358e-41     1.238978e-28 200
        delta_semantic  correct semantic_permuted clip_similarity                        con ControlNet correcta, ¿aporta el CLIP cerebral? -0.002950 2.729094e-01     1.587475e-01 200
   delta_semantic_zero  correct     semantic_zero clip_similarity         con ControlNet correcta, ¿aporta el CLIP cerebral frente al nulo? -0.001479 7.082067e-01     7.547642e-01 200
        delta_lowlevel  correct lowlevel_permuted clip_similarity      con pseudo-tokens correctos, ¿aporta la predicción VAE-PCA cerebral? -0.002963 5.227489e-01     5.812794e-01 200
   delta_lowlevel_zero  correct     lowlevel_zero clip_similarity con pseudo-tokens correctos, ¿aporta la ControlNet frente a desactivarla? -0.050797 4.704826e-17     7.671437e-17 200
           delta_brain  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada? -0.001086 8.284899e-01     4.206395e-01 200
     delta_joint_brain  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada? -0.001086 8.284899e-01     4.206395e-01 200
```

## Conclusión

- correcto: 0.7141
- mejor control: 0.7663
- margen: -0.0522

**La generación con señal cerebral correcta NO supera claramente a permutado/cero: no se puede atribuir la reconstrucción a la señal cerebral real.**

> Recordatorio metodológico: con un caption informativo (oracle) es esperable que `correcto ≈ permutado`; eso no implica que el decoder falle, sino que el texto puede dominar el condicionamiento (§38). Compara siempre el mismo delta entre modos de texto.

## Tests pareados

```
{'correct_vs_permuted': {'n': 200, 'mean_diff': -0.0010861949622631073, 'std_diff': 0.07081337771296504, 't_stat': -0.2169239339550069, 't_pvalue': 0.8284898548643155, 'wilcoxon_pvalue': 0.42063947371889066}, 'correct_vs_zero': {'n': 200, 'mean_diff': -0.05223118916153908, 'std_diff': 0.0772883388987046, 't_stat': -9.557205801503029, 't_pvalue': 4.732854844500654e-18, 'wilcoxon_pvalue': 1.182646652947994e-17}, 'correct_vs_permuted_text': {'n': 200, 'mean_diff': 0.1179592576622963, 'std_diff': 0.09790971517353396, 't_stat': 17.03810308280572, 't_pvalue': 9.711358376281803e-41, 'wilcoxon_pvalue': 1.2389782738406249e-28}, 'correct_vs_semantic_permuted': {'n': 200, 'mean_diff': -0.002950349897146225, 'std_diff': 0.03795084524131994, 't_stat': -1.0994286982961383, 't_pvalue': 0.2729093939923552, 'wilcoxon_pvalue': 0.15874749522053144}, 'correct_vs_semantic_zero': {'n': 200, 'mean_diff': -0.0014787861704826356, 'std_diff': 0.05579794256036062, 't_stat': -0.3748022529475859, 't_pvalue': 0.7082066593429169, 'wilcoxon_pvalue': 0.7547642355486297}, 'correct_vs_lowlevel_permuted': {'n': 200, 'mean_diff': -0.0029634861648082734, 'std_diff': 0.065459214312855, 't_stat': -0.6402463534234405, 't_pvalue': 0.5227489394897652, 'wilcoxon_pvalue': 0.5812793692598309}, 'correct_vs_lowlevel_zero': {'n': 200, 'mean_diff': -0.050797272473573685, 'std_diff': 0.07801741746982284, 't_stat': -9.207968424675995, 't_pvalue': 4.7048255870401804e-17, 'wilcoxon_pvalue': 7.671437250666649e-17}}
```
