# Experiment 5 — comparación generativa

- Arquitectura: `text_adapter_concat`
- Texto: `weak` (campo `primary_caption`, plantilla `Image of {caption}`)
- ControlNet: desactivada
- Muestras: 200 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
    condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
      correct              0.680221                0.667665      0.255      0.390        0.132937   0.059469    0.722905
     permuted              0.668688                0.652166      0.200      0.335        0.133489   0.058385    0.728078
         zero              0.656168                0.644470      0.160      0.240        0.091266   0.204018    0.824678
permuted_text              0.603174                0.604237      0.005      0.035        0.135065   0.056493    0.738881
```

## Deltas

```
                 delta positive      negative          metric                                                               question    value     t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct      permuted clip_similarity                     ¿la señal cerebral correcta supera a la permutada? 0.011533 3.581244e-02     3.023316e-02 200
    delta_correct_zero  correct          zero clip_similarity                    ¿la señal cerebral correcta supera al control nulo? 0.024053 4.758706e-03     3.313719e-02 200
            delta_text  correct permuted_text clip_similarity con cerebro correcto, ¿aporta el caption correcto frente al permutado? 0.077046 2.075932e-29     1.472416e-25 200
           delta_brain  correct      permuted clip_similarity                     ¿la señal cerebral correcta supera a la permutada? 0.011533 3.581244e-02     3.023316e-02 200
     delta_joint_brain  correct      permuted clip_similarity                     ¿la señal cerebral correcta supera a la permutada? 0.011533 3.581244e-02     3.023316e-02 200
```

## Conclusión

- correcto: 0.6802
- mejor control: 0.6687
- margen: +0.0115

**La generación con señal cerebral correcta supera a los controles: hay evidencia de que la señal cerebral influye en la reconstrucción.**

> Recordatorio metodológico: con un caption informativo (oracle) es esperable que `correcto ≈ permutado`; eso no implica que el decoder falle, sino que el texto puede dominar el condicionamiento (§38). Compara siempre el mismo delta entre modos de texto.

## Tests pareados

```
{'correct_vs_permuted': {'n': 200, 'mean_diff': 0.011532683372497559, 'std_diff': 0.07717266215970853, 't_stat': 2.113400883098722, 't_pvalue': 0.03581244173868863, 'wilcoxon_pvalue': 0.03023315799007624}, 'correct_vs_zero': {'n': 200, 'mean_diff': 0.024052579402923584, 'std_diff': 0.11914054449190432, 't_stat': 2.8550720618858403, 't_pvalue': 0.0047587055549213435, 'wilcoxon_pvalue': 0.033137190030337764}, 'correct_vs_permuted_text': {'n': 200, 'mean_diff': 0.0770463189482689, 'std_diff': 0.08169830696515144, 't_stat': 13.336867462142406, 't_pvalue': 2.0759316326346978e-29, 'wilcoxon_pvalue': 1.4724158742751246e-25}}
```
