# exp01_raw_ablate_mvnn_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_ablate_mvnn_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.002 / 0.011 / 0.016
- mean cosine: 0.572

## test
- retrieval Top-1/5/10: 0.015 / 0.090 / 0.165
- mean cosine: 0.621

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.