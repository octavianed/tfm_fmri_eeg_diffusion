# Experiment 5 — comparación generativa

- Arquitectura: `text_adapter_concat_controlnet`
- Texto: `weak` (campo `primary_caption`, plantilla `Image of {caption}`)
- ControlNet: `lllyasviel/sd-controlnet-canny` (canny, escala 0.5)
- Muestras: 200 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
        condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
          correct              0.717740                0.725924      0.420      0.620        0.113708   0.098147    0.735326
         permuted              0.719193                0.722007      0.440      0.590        0.112048   0.102275    0.740586
             zero              0.763313                0.776177      0.565      0.745        0.109175   0.127696    0.732746
    permuted_text              0.589446                0.591711      0.015      0.030        0.116945   0.099016    0.766424
semantic_permuted              0.716953                0.725285      0.405      0.595        0.114042   0.098842    0.735107
    semantic_zero              0.720450                0.726719      0.450      0.595        0.125760   0.090038    0.747976
lowlevel_permuted              0.721895                0.730367      0.445      0.625        0.112098   0.102034    0.738926
    lowlevel_zero              0.767077                0.778595      0.575      0.740        0.110618   0.126723    0.731166
```

## Deltas

```
                 delta positive          negative          metric                                                                  question     value     t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada? -0.001453 7.496703e-01     9.309640e-01 200
    delta_correct_zero  correct              zero clip_similarity                       ¿la señal cerebral correcta supera al control nulo? -0.045573 1.713502e-19     8.605199e-18 200
            delta_text  correct     permuted_text clip_similarity    con cerebro correcto, ¿aporta el caption correcto frente al permutado?  0.128294 6.382757e-46     3.632035e-30 200
        delta_semantic  correct semantic_permuted clip_similarity                        con ControlNet correcta, ¿aporta el CLIP cerebral?  0.000787 7.264399e-01     9.980529e-01 200
   delta_semantic_zero  correct     semantic_zero clip_similarity         con ControlNet correcta, ¿aporta el CLIP cerebral frente al nulo? -0.002710 4.902888e-01     3.678619e-01 200
        delta_lowlevel  correct lowlevel_permuted clip_similarity      con pseudo-tokens correctos, ¿aporta la predicción VAE-PCA cerebral? -0.004155 3.333974e-01     1.424728e-01 200
   delta_lowlevel_zero  correct     lowlevel_zero clip_similarity con pseudo-tokens correctos, ¿aporta la ControlNet frente a desactivarla? -0.049337 1.157911e-22     1.236191e-20 200
           delta_brain  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada? -0.001453 7.496703e-01     9.309640e-01 200
     delta_joint_brain  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada? -0.001453 7.496703e-01     9.309640e-01 200
```

## Conclusión

- correcto: 0.7177
- mejor control: 0.7633
- margen: -0.0456

**La generación con señal cerebral correcta NO supera claramente a permutado/cero: no se puede atribuir la reconstrucción a la señal cerebral real.**

> Recordatorio metodológico: con un caption informativo (oracle) es esperable que `correcto ≈ permutado`; eso no implica que el decoder falle, sino que el texto puede dominar el condicionamiento (§38). Compara siempre el mismo delta entre modos de texto.

## Tests pareados

```
{'correct_vs_permuted': {'n': 200, 'mean_diff': -0.0014530614018440247, 'std_diff': 0.06431409147698607, 't_stat': -0.31951615800776945, 't_pvalue': 0.749670315491367, 'wilcoxon_pvalue': 0.9309639632270188}, 'correct_vs_zero': {'n': 200, 'mean_diff': -0.045573449283838274, 'std_diff': 0.06410315181935355, 't_stat': -10.054199868822684, 't_pvalue': 1.713502263767762e-19, 'wilcoxon_pvalue': 8.605199366021795e-18}, 'correct_vs_permuted_text': {'n': 200, 'mean_diff': 0.12829406023025514, 'std_diff': 0.09664532416282617, 't_stat': 18.77330347031161, 't_pvalue': 6.382756822074656e-46, 'wilcoxon_pvalue': 3.632035390852315e-30}, 'correct_vs_semantic_permuted': {'n': 200, 'mean_diff': 0.0007870118319988251, 'std_diff': 0.03176749620117729, 't_stat': 0.3503589957207855, 't_pvalue': 0.7264398618613627, 'wilcoxon_pvalue': 0.9980528920316095}, 'correct_vs_semantic_zero': {'n': 200, 'mean_diff': -0.002709906846284866, 'std_diff': 0.05545097068574403, 't_stat': -0.69113073538477, 't_pvalue': 0.49028881767805294, 'wilcoxon_pvalue': 0.3678618878402944}, 'correct_vs_lowlevel_permuted': {'n': 200, 'mean_diff': -0.004155495166778564, 'std_diff': 0.06060692390237383, 't_stat': -0.9696511957446359, 't_pvalue': 0.33339742812866113, 'wilcoxon_pvalue': 0.1424727716267651}, 'correct_vs_lowlevel_zero': {'n': 200, 'mean_diff': -0.04933665722608566, 'std_diff': 0.06272700979909103, 't_stat': -11.123209921014595, 't_pvalue': 1.15791077226629e-22, 'wilcoxon_pvalue': 1.2361910463293077e-20}}
```
