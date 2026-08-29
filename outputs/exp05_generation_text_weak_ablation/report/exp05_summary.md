# Experiment 5 — comparación generativa

- Arquitectura: `text_adapter_concat`
- Texto: `weak` (campo `prompt_categories`, plantilla `Image of {caption}`)
- ControlNet: desactivada
- Muestras: 159 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
    condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
      correct              0.725355                0.729931   0.257862   0.628931        0.125217   0.137109    0.716194
     permuted              0.623630                0.614037   0.056604   0.182390        0.129226   0.122222    0.760024
         zero              0.609365                0.603363   0.088050   0.213836        0.130947   0.104713    0.732351
permuted_text              0.639398                0.635610   0.050314   0.207547        0.127360   0.130026    0.735767
```

## Deltas

```
                 delta positive      negative          metric                                                               question    value     t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct      permuted clip_similarity                     ¿la señal cerebral correcta supera a la permutada? 0.101725 1.201849e-26     6.026222e-21 159
    delta_correct_zero  correct          zero clip_similarity                    ¿la señal cerebral correcta supera al control nulo? 0.115990 1.043351e-31     1.976128e-23 159
            delta_text  correct permuted_text clip_similarity con cerebro correcto, ¿aporta el caption correcto frente al permutado? 0.085958 1.632168e-28     1.657401e-22 159
           delta_brain  correct      permuted clip_similarity                     ¿la señal cerebral correcta supera a la permutada? 0.101725 1.201849e-26     6.026222e-21 159
     delta_joint_brain  correct      permuted clip_similarity                     ¿la señal cerebral correcta supera a la permutada? 0.101725 1.201849e-26     6.026222e-21 159
```

## Conclusión

- correcto: 0.7254
- mejor control: 0.6236
- margen: +0.1017

**La generación con señal cerebral correcta supera a los controles: hay evidencia de que la señal cerebral influye en la reconstrucción.**

> Recordatorio metodológico: con un caption informativo (oracle) es esperable que `correcto ≈ permutado`; eso no implica que el decoder falle, sino que el texto puede dominar el condicionamiento (§38). Compara siempre el mismo delta entre modos de texto.

## Tests pareados

```
{'correct_vs_permuted': {'n': 159, 'mean_diff': 0.101724866051344, 'std_diff': 0.0989257876544638, 't_stat': 12.96630317578239, 't_pvalue': 1.2018491356213722e-26, 'wilcoxon_pvalue': 6.026222013261482e-21}, 'correct_vs_zero': {'n': 159, 'mean_diff': 0.11599026854683019, 'std_diff': 0.09867817821194212, 't_stat': 14.821733256989702, 't_pvalue': 1.0433505801758682e-31, 'wilcoxon_pvalue': 1.9761278193754127e-23}, 'correct_vs_permuted_text': {'n': 159, 'mean_diff': 0.08595750523063372, 'std_diff': 0.0794183617110434, 't_stat': 13.647762007498137, 't_pvalue': 1.6321681063178185e-28, 'wilcoxon_pvalue': 1.6574005478044546e-22}}
```
