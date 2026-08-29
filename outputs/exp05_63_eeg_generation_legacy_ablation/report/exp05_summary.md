# Experiment 5 — comparación generativa

- Arquitectura: `legacy_adapter`
- Texto: `none`
- ControlNet: desactivada
- Muestras: 200 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
  correct              0.597126                0.595826      0.015      0.035        0.166965   0.029605    0.812929
 permuted              0.592241                0.590134      0.000      0.025        0.169425   0.026751    0.818967
     zero              0.606470                0.609926      0.005      0.015        0.079340   0.276694    0.912838
```

## Deltas

```
                 delta positive negative          metric                                            question     value  t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct permuted clip_similarity  ¿la señal cerebral correcta supera a la permutada?  0.004885  0.089824         0.129042 200
    delta_correct_zero  correct     zero clip_similarity ¿la señal cerebral correcta supera al control nulo? -0.009344  0.021550         0.031272 200
           delta_brain  correct permuted clip_similarity  ¿la señal cerebral correcta supera a la permutada?  0.004885  0.089824         0.129042 200
     delta_joint_brain  correct permuted clip_similarity  ¿la señal cerebral correcta supera a la permutada?  0.004885  0.089824         0.129042 200
```

## Conclusión

- correcto: 0.5971
- mejor control: 0.6065
- margen: -0.0093

**La generación con señal cerebral correcta NO supera claramente a permutado/cero: no se puede atribuir la reconstrucción a la señal cerebral real.**

## Tests pareados

```
{'correct_vs_permuted': {'n': 200, 'mean_diff': 0.0048852570354938505, 'std_diff': 0.040529586641272576, 't_stat': 1.7046304509403845, 't_pvalue': 0.08982402728367238, 'wilcoxon_pvalue': 0.1290417007876884}, 'correct_vs_zero': {'n': 200, 'mean_diff': -0.00934381142258644, 'std_diff': 0.057044022680926766, 't_stat': -2.316481940972348, 't_pvalue': 0.021549995477027403, 'wilcoxon_pvalue': 0.031271525016122466}}
```
