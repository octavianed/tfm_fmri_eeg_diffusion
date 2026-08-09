# exp01_63_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_63_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.004 / 0.028 / 0.050
- mean cosine: 0.623

## test
- retrieval Top-1/5/10: 0.005 / 0.045 / 0.100
- mean cosine: 0.579

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.