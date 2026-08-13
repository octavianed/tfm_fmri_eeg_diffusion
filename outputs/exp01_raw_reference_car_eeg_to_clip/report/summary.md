# exp01_raw_reference_car_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_reference_car_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.002 / 0.022 / 0.034
- mean cosine: 0.433

## test
- retrieval Top-1/5/10: 0.005 / 0.030 / 0.085
- mean cosine: 0.423

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.