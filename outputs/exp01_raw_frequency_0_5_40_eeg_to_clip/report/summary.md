# exp01_raw_frequency_0_5_40_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_frequency_0_5_40_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.001 / 0.013 / 0.021
- mean cosine: 0.305

## test
- retrieval Top-1/5/10: 0.015 / 0.035 / 0.055
- mean cosine: 0.276

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.