# exp01_raw_temporal_200_400_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_temporal_200_400_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.006 / 0.023 / 0.038
- mean cosine: 0.367

## test
- retrieval Top-1/5/10: 0.035 / 0.100 / 0.180
- mean cosine: 0.330

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.