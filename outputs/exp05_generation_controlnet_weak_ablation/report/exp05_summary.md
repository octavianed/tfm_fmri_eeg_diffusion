# Experiment 5 — comparación generativa

- Arquitectura: `text_adapter_concat_controlnet`
- Texto: `weak` (campo `prompt_categories`, plantilla `Image of {caption}`)
- ControlNet: `lllyasviel/sd-controlnet-canny` (canny, escala 0.5)
- Muestras: 159 | semilla de difusión: 123 | pasos: 50 | CFG: 3.0

## Métricas por condición

```
        condition  mean_clip_similarity  median_clip_similarity  clip_top1  clip_top5  mean_pixel_mse  mean_ssim  mean_lpips
          correct              0.700190                0.704940   0.169811   0.408805        0.113369   0.133086    0.739737
         permuted              0.655941                0.655186   0.094340   0.257862        0.117689   0.117367    0.763856
             zero              0.648515                0.637166   0.094340   0.364780        0.132839   0.097767    0.747739
    permuted_text              0.602980                0.601810   0.018868   0.081761        0.116802   0.117043    0.755861
semantic_permuted              0.658848                0.660960   0.106918   0.264151        0.117551   0.115988    0.758016
    semantic_zero              0.657304                0.653829   0.106918   0.276730        0.115560   0.097904    0.751779
lowlevel_permuted              0.689002                0.684473   0.150943   0.402516        0.113859   0.134133    0.742409
    lowlevel_zero              0.716558                0.724346   0.238994   0.572327        0.125900   0.129504    0.728678
```

## Deltas

```
                 delta positive          negative          metric                                                                  question     value     t_pvalue  wilcoxon_pvalue   n
delta_correct_permuted  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada?  0.044248 1.177651e-10     1.403738e-09 159
    delta_correct_zero  correct              zero clip_similarity                       ¿la señal cerebral correcta supera al control nulo?  0.051675 6.870193e-13     5.821785e-12 159
            delta_text  correct     permuted_text clip_similarity    con cerebro correcto, ¿aporta el caption correcto frente al permutado?  0.097209 1.293219e-27     6.110037e-22 159
        delta_semantic  correct semantic_permuted clip_similarity                        con ControlNet correcta, ¿aporta el CLIP cerebral?  0.041341 3.041485e-10     7.682460e-10 159
   delta_semantic_zero  correct     semantic_zero clip_similarity         con ControlNet correcta, ¿aporta el CLIP cerebral frente al nulo?  0.042886 1.090449e-11     1.052726e-10 159
        delta_lowlevel  correct lowlevel_permuted clip_similarity      con pseudo-tokens correctos, ¿aporta la predicción VAE-PCA cerebral?  0.011188 8.428504e-03     2.699921e-02 159
   delta_lowlevel_zero  correct     lowlevel_zero clip_similarity con pseudo-tokens correctos, ¿aporta la ControlNet frente a desactivarla? -0.016368 3.407802e-04     5.830216e-04 159
           delta_brain  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada?  0.044248 1.177651e-10     1.403738e-09 159
     delta_joint_brain  correct          permuted clip_similarity                        ¿la señal cerebral correcta supera a la permutada?  0.044248 1.177651e-10     1.403738e-09 159
```

## Conclusión

- correcto: 0.7002
- mejor control: 0.6559
- margen: +0.0442

**La generación con señal cerebral correcta supera a los controles: hay evidencia de que la señal cerebral influye en la reconstrucción.**

> Recordatorio metodológico: con un caption informativo (oracle) es esperable que `correcto ≈ permutado`; eso no implica que el decoder falle, sino que el texto puede dominar el condicionamiento (§38). Compara siempre el mismo delta entre modos de texto.

## Tests pareados

```
{'correct_vs_permuted': {'n': 159, 'mean_diff': 0.044248158733050026, 'std_diff': 0.08084503518093693, 't_stat': 6.901451037532396, 't_pvalue': 1.177650884510521e-10, 'wilcoxon_pvalue': 1.4037384714845396e-09}, 'correct_vs_zero': {'n': 159, 'mean_diff': 0.05167463693603779, 'std_diff': 0.08327962054592813, 't_stat': 7.824151631200627, 't_pvalue': 6.870192670676714e-13, 'wilcoxon_pvalue': 5.821785129677739e-12}, 'correct_vs_permuted_text': {'n': 159, 'mean_diff': 0.09720923821881132, 'std_diff': 0.09202845971479023, 't_stat': 13.319378135865, 't_pvalue': 1.2932189635558197e-27, 'wilcoxon_pvalue': 6.110036851933665e-22}, 'correct_vs_semantic_permuted': {'n': 159, 'mean_diff': 0.04134119690964057, 'std_diff': 0.07751699450750757, 't_stat': 6.724882219316032, 't_pvalue': 3.04148542806278e-10, 'wilcoxon_pvalue': 7.682460316833752e-10}, 'correct_vs_semantic_zero': {'n': 159, 'mean_diff': 0.042886021564591606, 'std_diff': 0.0737263815764976, 't_stat': 7.334852792270534, 't_pvalue': 1.090448997221707e-11, 'wilcoxon_pvalue': 1.0527264374860414e-10}, 'correct_vs_lowlevel_permuted': {'n': 159, 'mean_diff': 0.011187798946908434, 'std_diff': 0.052877931752844554, 't_stat': 2.6678951366421613, 't_pvalue': 0.008428504425622861, 'wilcoxon_pvalue': 0.026999206316116593}, 'correct_vs_lowlevel_zero': {'n': 159, 'mean_diff': -0.016368073206277763, 'std_diff': 0.05635916861212071, 't_stat': -3.6621113303062747, 't_pvalue': 0.00034078020036359843, 'wilcoxon_pvalue': 0.0005830215936856789}}
```
