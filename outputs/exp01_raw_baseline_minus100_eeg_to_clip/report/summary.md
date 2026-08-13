# exp01_raw_baseline_minus100_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_baseline_minus100_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.002 / 0.019 / 0.030
- mean cosine: 0.437

## test
- retrieval Top-1/5/10: 0.010 / 0.040 / 0.075
- mean cosine: 0.457

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.