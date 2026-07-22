# exp01_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.005 / 0.022 / 0.035
- mean cosine: 0.626

## test
- retrieval Top-1/5/10: 0.020 / 0.080 / 0.130
- mean cosine: 0.592

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.