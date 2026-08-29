# Experiment 5 — comparación generativa

- Arquitectura: `legacy_adapter`
- Texto: `none`
- ControlNet: desactivada
- Muestras: 159 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
  correct              0.685062                0.678084   0.138365   0.433962        0.131609   0.131560    0.732165
 permuted              0.580204                0.583356   0.025157   0.056604        0.138750   0.117634    0.782097
     zero              0.582376                0.575309   0.006289   0.031447        0.127440   0.088576    0.749675
```

## Deltas

```
                 delta positive negative          metric                                            question    value     t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct permuted clip_similarity  ¿la señal cerebral correcta supera a la permutada? 0.104858 1.508376e-29     1.815750e-21 159
    delta_correct_zero  correct     zero clip_similarity ¿la señal cerebral correcta supera al control nulo? 0.102687 1.123502e-32     1.348984e-23 159
           delta_brain  correct permuted clip_similarity  ¿la señal cerebral correcta supera a la permutada? 0.104858 1.508376e-29     1.815750e-21 159
     delta_joint_brain  correct permuted clip_similarity  ¿la señal cerebral correcta supera a la permutada? 0.104858 1.508376e-29     1.815750e-21 159
```

## Conclusión

- correcto: 0.6851
- mejor control: 0.5824
- margen: +0.1027

**La generación con señal cerebral correcta supera a los controles: hay evidencia de que la señal cerebral influye en la reconstrucción.**

## Tests pareados

```
{'correct_vs_permuted': {'n': 159, 'mean_diff': 0.10485830640642897, 'std_diff': 0.09426531458528302, 't_stat': 14.026505294564565, 't_pvalue': 1.5083758261400438e-29, 'wilcoxon_pvalue': 1.815749688049639e-21}, 'correct_vs_zero': {'n': 159, 'mean_diff': 0.10268650358577944, 'std_diff': 0.08529641245665302, 't_stat': 15.180328284226892, 't_pvalue': 1.1235018807918888e-32, 'wilcoxon_pvalue': 1.348983665448377e-23}}
```
