# exp01_raw_baseline_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_baseline_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.004 / 0.021 / 0.034
- mean cosine: 0.340

## test
- retrieval Top-1/5/10: 0.005 / 0.055 / 0.090
- mean cosine: 0.317

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.