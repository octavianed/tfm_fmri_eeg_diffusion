# exp01_raw_temporal_100_600_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_temporal_100_600_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.005 / 0.025 / 0.044
- mean cosine: 0.327

## test
- retrieval Top-1/5/10: 0.035 / 0.120 / 0.165
- mean cosine: 0.296

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.