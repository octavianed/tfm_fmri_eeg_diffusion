# exp01_raw_train_independent_trials_eeg_to_clip — summary

- **Objetivo:** fMRI -> CLIP embedding
- **Sujetos:** sub-01
- **Modelo:** fMRIEncoder(hidden=None, out=None) + CLIPHead
- **Config:** `outputs\exp01_raw_train_independent_trials_eeg_to_clip\config.yaml`

## val
- retrieval Top-1/5/10: 0.004 / 0.015 / 0.025
- mean cosine: 0.636

## test
- retrieval Top-1/5/10: 0.005 / 0.020 / 0.055
- mean cosine: 0.617

**Conclusion preliminar:** confirmar en el Experimento 2 (retrieval ablation) que fMRI correcto supera claramente a permutado/cero antes de atribuir el resultado a la senal cerebral.